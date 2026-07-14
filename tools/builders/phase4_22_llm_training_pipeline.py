"""Builder for Notebook 22 — LLM Training Pipeline.

Run:  python3 tools/builders/phase4_22_llm_training_pipeline.py
Emits: notebooks/phase4_nlp_llms/22_llm_training_pipeline.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # 22 · LLM Training Pipeline
    ### Phase 4 — Modern NLP and LLMs · *ML/AI Senior Mastery Curriculum*

    > How does a language model go from a random initialisation to writing code, passing
    > exams, and following nuanced instructions? This notebook traces the **four-stage
    > pipeline** used to build every modern LLM: **pre-training** (learn language from
    > text) → **continual pre-training** (adapt to a domain) → **supervised fine-tuning
    > / SFT** (learn to follow instructions) → **alignment** (RLHF or DPO — make it
    > helpful and safe). We implement each stage from first principles before showing
    > the production tooling.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **BPE tokenisation** from scratch: why subword tokenisation, the merge algorithm,
      vocabulary size tradeoffs.
    - **Pre-training objective**: causal language modelling (next-token prediction
      cross-entropy), teacher forcing, compute budgeting via Chinchilla scaling laws
      (Notebook 19).
    - **Data curation** for pre-training: deduplication, quality filtering, domain
      mixing, data poisoning risks.
    - **Supervised fine-tuning (SFT)**: instruction-response templates (Alpaca, ChatML,
      Llama-2), loss masking on prompt tokens.
    - **RLHF** (PPO): reward model training, the PPO loop, KL penalty against the
      reference model.
    - **DPO** (Direct Preference Optimisation): no reward model needed, just preference
      pairs — the Bradley-Terry model and the DPO loss.
    - **LoRA** (Low-Rank Adaptation): freeze base model, add $BA$ adapters to attention
      projections, train <1% of parameters.

    **Why it matters**
    - Understanding the training pipeline is prerequisite to: fine-tuning LLMs for
      your domain (Notebook 41 MLOps), evaluating them (Notebooks 38–40), and
      diagnosing production failures (Notebook 24 Hallucinations).

    **Typical interview questions**
    - "Explain BPE tokenisation."
    - "What is the difference between SFT and RLHF?"
    - "Why is DPO simpler than RLHF? What does it sacrifice?"
    - "How does LoRA work and why does it avoid catastrophic forgetting?"
    """),

    md(r"""
    ## 2 · Historical Motivation

    **GPT (2018): pre-training works.** Radford et al. showed that a Transformer
    trained on next-token prediction on BooksCorpus could be fine-tuned to beat
    task-specific models on diverse NLP benchmarks. The key insight: language
    modelling is a universal pre-training objective.

    **GPT-2 and scaling (2019).** Scaled to 1.5B parameters, the model showed
    surprising zero-shot capability. The paper introduced the idea that LLMs may be
    "multitask learners" without task-specific fine-tuning.

    **GPT-3 and in-context learning (2020).** 175B parameters trained on 300B tokens.
    Introduced few-shot learning purely via the prompt — no gradient update. But the
    raw model was difficult to use: it completed text rather than following instructions.

    **InstructGPT and RLHF (2022).** Ouyang et al. (OpenAI) showed that fine-tuning
    GPT-3 with SFT + RLHF on human-labelled data produced a much more helpful and
    safer model despite being 100× smaller. This established the **SFT → RLHF**
    recipe.

    **Stanford Alpaca (2023)** democratised instruction tuning: fine-tune LLaMA-7B on
    52K GPT-4-generated instruction-response pairs for ~$600. Spawned the instruction-
    tuning cottage industry.

    **DPO (Rafailov et al., 2023)** simplified RLHF: derive that the optimal RLHF
    policy implicitly defines a reward, so you can optimise it directly from preference
    pairs without ever training a reward model. Eliminates the unstable PPO loop.

    **LoRA (Hu et al., 2021)** made fine-tuning affordable: instead of updating all
    $d \times k$ parameters of a weight matrix, add two low-rank matrices $A$ ($d
    \times r$) and $B$ ($r \times k$) and train only those. $r=8$ or $r=16$ is typical;
    for a 7B model this reduces trainable parameters from 7B to ~4M.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The four-stage pipeline.**

    ```mermaid
    flowchart LR
        PT["1. Pre-training\nTrillions of tokens\nNext-token prediction\n(Chinchilla budget)"]
        CPT["2. Continual Pre-training\nDomain corpus\n(optional, cheap)"]
        SFT["3. SFT\nInstruction-response pairs\n(10K-1M examples)\nLoss masked on prompt"]
        ALIGN["4. Alignment\nRLHF (PPO+reward model)\nor DPO (preference pairs)"]
        PT --> CPT --> SFT --> ALIGN
    ```

    **BPE intuition.** Start with character-level tokens. Find the most frequent pair
    of adjacent tokens and merge them into a new token. Repeat until the vocabulary
    reaches the target size. Result: common words → single tokens; rare/OOV words →
    subword pieces; characters → last-resort fallback. "tokenization" might become
    ["token", "ization"] or ["tok", "en", "ization"] depending on training corpus.

    **LoRA intuition.** A pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$
    stores everything the model learned about language. Fine-tuning wants to add a
    small correction $\Delta W$. LoRA assumes $\Delta W = BA$ where $B \in \mathbb{R}^{
    d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d,k)$. Instead
    of storing $dk$ floats for $\Delta W$, we store only $r(d+k)$ — with $r=8$,
    $d=k=4096$, that's 65K vs 16M. The hypothesis: the *task adaptation* of a large
    model lives in a low-dimensional subspace.

    **DPO intuition.** RLHF trains a reward model $r_\phi$ and then optimises the
    policy with PPO. DPO notices that the optimal policy under the RLHF objective is:
    $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\,\exp(r(x,y)/\beta)$. Plugging this
    back in, we can express the reward implicitly via the policy itself, eliminating
    the need for $r_\phi$ entirely — you just need preference pairs $(y_w, y_l)$ per
    prompt $x$.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from collections import Counter, defaultdict

    rng = np.random.default_rng(42)
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    CORPUS_SAMPLE = [
        "the cat sat on the mat",
        "the cat ate the rat",
        "tokenization splits words into subwords",
        "low rank adaptation is efficient",
        "language models are trained on tokens",
        "the dog ran after the cat",
        "supervised fine tuning follows pre training",
        "reinforcement learning from human feedback",
    ]
    print(f"Sample corpus: {len(CORPUS_SAMPLE)} sentences")
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 BPE tokenisation

    **Algorithm.** Given a corpus:
    1. Initialise vocabulary as character-level tokens + `<EOS>`.
    2. Represent each word as a sequence of characters separated by spaces (plus a
       word-boundary marker `</w>` at the end).
    3. Count all adjacent symbol pairs across the corpus.
    4. Merge the most frequent pair → new symbol. Add to vocabulary.
    5. Repeat steps 3–4 until target vocab size is reached or no pairs remain.

    **Encoding a new word** (after training): greedily apply the learned merge rules
    in order. Any character not in the vocabulary maps to `<UNK>` (or is handled by
    byte-level BPE as in GPT-4).

    ### 4.2 Pre-training loss (causal LM)

    For a sequence $x_1, x_2, \dots, x_T$ and model parameters $\theta$:
    $$J_{\text{CLM}} = -\frac{1}{T-1} \sum_{t=1}^{T-1}
      \log p_\theta(x_{t+1} \mid x_1, \dots, x_t)$$
    This is exactly the cross-entropy from Notebook 02 applied position-by-position,
    with a **causal mask** preventing the model from seeing future tokens (Notebook 19).

    ### 4.3 SFT loss (instruction masking)

    For an instruction-response pair (prompt $P$, response $R$), we concatenate to
    form $[P; R]$ and apply CLM loss only on the response tokens:
    $$J_{\text{SFT}} = -\frac{1}{|R|} \sum_{t \in R} \log p_\theta(x_t \mid x_{<t})$$
    The prompt tokens contribute to the context but NOT to the loss. This prevents the
    model from "forgetting" how to follow format while learning to generate responses.

    ### 4.4 RLHF objective

    Train a reward model $r_\phi$ to predict human preference:
    $$J_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[
      \log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\right]$$
    (Bradley-Terry preference model). Then optimise the policy $\pi_\theta$:
    $$J_{\text{PPO}} = \mathbb{E}[r_\phi(x,y)] - \beta\,\mathrm{KL}\!\left[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right]$$
    The KL term prevents the policy from drifting too far from the supervised baseline.

    ### 4.5 DPO loss

    Directly optimise preference pairs without a reward model:
    $$J_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(
      \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
      - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
    Intuition: increase the log-probability of $y_w$ (winner) relative to the
    reference model, while decreasing it for $y_l$ (loser).

    ### 4.6 LoRA

    Replace a forward pass $h = xW_0$ with:
    $$h = xW_0 + x\underbrace{BA}_{\Delta W},\quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k}$$
    Initialise $B=0$, $A \sim \mathcal{N}(0, \sigma^2)$ so $\Delta W = 0$ at start
    (no change to pre-trained behaviour). Scale by $\alpha/r$ (hyperparameter $\alpha$).
    Only $A$ and $B$ are trained; $W_0$ is frozen.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    ### 5a BPE tokenisation from scratch
    """),

    code(r"""
    # 5a. BPE implementation from scratch in pure Python.
    def get_pairs(vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(pair, vocab):
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = freq
        return new_vocab

    def learn_bpe(corpus_sentences, num_merges=30):
        # Build initial vocab: characters + </w> as word boundary.
        vocab = Counter()
        for sent in corpus_sentences:
            for word in sent.split():
                word_str = " ".join(list(word)) + " </w>"
                vocab[word_str] += 1
        merges = []
        for i in range(num_merges):
            pairs = get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            vocab = merge_vocab(best, vocab)
        return vocab, merges

    # Train BPE on our sample corpus.
    bpe_vocab, bpe_merges = learn_bpe(CORPUS_SAMPLE, num_merges=25)
    print(f"BPE merges learned: {len(bpe_merges)}")
    print(f"Final BPE tokens (sample):")
    for token, freq in sorted(bpe_vocab.items(), key=lambda x: -x[1])[:8]:
        print(f"  '{token}'  freq={freq}")
    """),

    code(r"""
    # 5a.2 Encode a new sentence using learned BPE merge rules.
    def apply_bpe(word, merges):
        chars = list(word) + ["</w>"]
        tokens = chars[:]
        for pair in merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def tokenise_bpe(sentence, merges):
        result = []
        for word in sentence.lower().split():
            result.extend(apply_bpe(word, merges))
        return result

    test_sentences = [
        "the cat sat",
        "tokenization is important",
        "low rank finetuning",
    ]
    for s in test_sentences:
        tokens = tokenise_bpe(s, bpe_merges)
        print(f"  '{s}' -> {tokens}")
    """),

    md(r"""
    ### 5b Pre-training loss (causal LM cross-entropy)
    """),

    code(r"""
    # 5b. Implement the causal LM loss from scratch.
    def softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def causal_lm_loss(logits, targets):
        # logits: (T, V), targets: (T,) integer token ids.
        # Loss on all T tokens (pre-training).
        T, V = logits.shape
        probs = softmax(logits)
        log_probs = np.log(probs[np.arange(T), targets] + 1e-12)
        return -log_probs.mean()

    # Simulate: tiny 8-token sequence, 10-token vocabulary, random logits.
    V_sim, T_sim = 10, 8
    logits_random = rng.normal(0, 1, (T_sim, V_sim))
    targets_sim   = rng.integers(0, V_sim, T_sim)
    loss_random   = causal_lm_loss(logits_random, targets_sim)
    # Expected loss at random init ~ log(V).
    print(f"Random init loss: {loss_random:.3f}  (expected ≈ log({V_sim}) = {np.log(V_sim):.3f})")
    # Simulate near-perfect logits.
    logits_good = np.eye(V_sim)[targets_sim] * 10
    loss_good   = causal_lm_loss(logits_good, targets_sim)
    print(f"Near-perfect logits loss: {loss_good:.3f}")
    """),

    md(r"""
    ### 5c SFT loss — mask the prompt, train only on response tokens
    """),

    code(r"""
    # 5c. SFT loss: loss only on response tokens, prompt tokens masked out.
    def sft_loss(logits, targets, prompt_len):
        # prompt_len: number of tokens to EXCLUDE from loss.
        T, V = logits.shape
        assert prompt_len < T
        probs = softmax(logits)
        # Only compute loss on response tokens (from prompt_len onwards).
        resp_logits  = logits[prompt_len:]
        resp_targets = targets[prompt_len:]
        resp_probs   = softmax(resp_logits)
        log_probs    = np.log(resp_probs[np.arange(len(resp_targets)), resp_targets] + 1e-12)
        return -log_probs.mean()

    # Simulate prompt (4 tokens) + response (4 tokens).
    logits_sft  = rng.normal(0, 1, (8, V_sim))
    targets_sft = rng.integers(0, V_sim, 8)
    loss_clm_all = causal_lm_loss(logits_sft, targets_sft)
    loss_sft_only = sft_loss(logits_sft, targets_sft, prompt_len=4)
    print(f"CLM loss (all 8 tokens):       {loss_clm_all:.3f}")
    print(f"SFT loss (response 4 tokens):  {loss_sft_only:.3f}")
    print("The SFT loss only penalises wrong predictions on the RESPONSE side.")
    print("Prompt tokens provide context but don't contribute gradients.")
    """),

    md(r"""
    ### 5d DPO loss from scratch
    """),

    code(r"""
    # 5d. DPO loss from scratch (batched).
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def log_prob_sequence(logits, targets):
        # Sum of log probs over a response sequence.
        probs = softmax(logits)
        return np.sum(np.log(probs[np.arange(len(targets)), targets] + 1e-12))

    def dpo_loss(logits_w, logits_l, logits_w_ref, logits_l_ref,
                 targets_w, targets_l, beta=0.1):
        # w = winner (preferred), l = loser (rejected).
        lp_w     = log_prob_sequence(logits_w,     targets_w)
        lp_l     = log_prob_sequence(logits_l,     targets_l)
        lp_w_ref = log_prob_sequence(logits_w_ref, targets_w)
        lp_l_ref = log_prob_sequence(logits_l_ref, targets_l)
        # Log-ratio of policy vs reference.
        ratio_w = lp_w - lp_w_ref
        ratio_l = lp_l - lp_l_ref
        return -np.log(sigmoid(beta * (ratio_w - ratio_l)))

    # Simulate: 4 response tokens, 10 vocab.
    T_dpo = 4
    # "Good" policy: high logits for winner tokens, low for loser.
    targets_w_dpo = rng.integers(0, V_sim, T_dpo)
    targets_l_dpo = rng.integers(0, V_sim, T_dpo)
    logits_w_policy  = np.eye(V_sim)[targets_w_dpo] * 5        # policy prefers winner
    logits_l_policy  = rng.normal(0, 1, (T_dpo, V_sim))        # policy uncertain on loser
    logits_w_ref     = rng.normal(0, 1, (T_dpo, V_sim))        # reference uncertain
    logits_l_ref     = rng.normal(0, 1, (T_dpo, V_sim))
    loss_dpo = dpo_loss(logits_w_policy, logits_l_policy,
                        logits_w_ref,    logits_l_ref,
                        targets_w_dpo,   targets_l_dpo)
    print(f"DPO loss (policy prefers winner): {loss_dpo:.3f}  (lower is better)")
    # Reverse: policy prefers loser.
    loss_dpo_bad = dpo_loss(logits_l_policy, logits_w_policy,
                            logits_w_ref,    logits_l_ref,
                            targets_w_dpo,   targets_l_dpo)
    print(f"DPO loss (policy prefers loser):  {loss_dpo_bad:.3f}  (should be high)")
    """),

    md(r"""
    ### 5e LoRA forward pass from scratch
    """),

    code(r"""
    # 5e. LoRA: low-rank adaptation of a weight matrix.
    class LoRALinear:
        def __init__(self, d_in, d_out, rank=4, alpha=8):
            self.W0 = rng.normal(0, 0.02, (d_in, d_out))   # pretrained, frozen
            # LoRA adapters: A initialized with small random; B initialized to 0.
            self.A  = rng.normal(0, 0.01, (d_in, rank))
            self.B  = np.zeros((rank, d_out))
            self.scale = alpha / rank                       # LoRA scaling factor

        def forward(self, x):
            base   = x @ self.W0                           # pretrained path (no grad)
            lora   = (x @ self.A) @ self.B * self.scale   # adapter path (trained)
            return base + lora                             # additive merge

        def param_count(self):
            total   = self.W0.size
            adapter = self.A.size + self.B.size
            return total, adapter, adapter / total

    layer = LoRALinear(d_in=512, d_out=512, rank=8)
    total, adapter, frac = layer.param_count()
    print(f"Weight matrix params:   {total:,}")
    print(f"LoRA adapter params:    {adapter:,}")
    print(f"Fraction trained:       {frac*100:.2f}%")
    print()
    x_test = rng.normal(0, 1, (4, 512))          # batch of 4 tokens
    out = layer.forward(x_test)
    print(f"LoRA output shape: {out.shape}")
    print("At init B=0, so out == x @ W0 (no change to base model behaviour)")
    print(f"Max |delta| vs base: {np.abs(out - x_test @ layer.W0).max():.6f}")
    """),

    code(r"""
    # LoRA parameter savings across common model sizes.
    configs = [
        ("7B (d=4096, r=8)",  4096, 4096, 8),
        ("13B (d=5120, r=8)", 5120, 5120, 8),
        ("70B (d=8192, r=16)",8192, 8192, 16),
    ]
    print(f"{'Config':<25} {'W0 params':>12} {'LoRA params':>12} {'Ratio':>8}")
    for name, d_in, d_out, r in configs:
        w0 = d_in * d_out
        lo = d_in * r + r * d_out
        print(f"{name:<25} {w0:>12,} {lo:>12,} {lo/w0:>7.3%}")
    print("\nWith ~32 attention projection layers per 7B model,")
    print("LoRA r=8 trains ~4M params out of 7B = 0.06% of all weights.")
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — BPE merge frequency: how many pairs are merged at each step.
    bpe_freq_all = []
    vocab_temp = Counter()
    for sent in CORPUS_SAMPLE:
        for word in sent.split():
            vocab_temp[" ".join(list(word)) + " </w>"] += 1
    for i in range(20):
        pairs = get_pairs(vocab_temp)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        bpe_freq_all.append(pairs[best])
        vocab_temp = merge_vocab(best, vocab_temp)

    fig, ax = plt.subplots()
    ax.bar(range(len(bpe_freq_all)), bpe_freq_all, color="steelblue")
    ax.set_xlabel("merge step"); ax.set_ylabel("frequency of merged pair")
    ax.set_title("Figure 1 — BPE: merge frequency decreases as common pairs are exhausted")
    plt.show()
    """),

    md(r"""
    **Figure 1.** The frequency of the most-common pair merged at each BPE step.
    Early merges are high-frequency pairs ("t h" → "th", "a t" → "at") that appear
    across many words. Later merges combine rarer subwords. This diminishing-frequency
    curve reflects the Zipfian distribution of natural language: a few patterns are
    very common, and the long tail is rare. In production (GPT-4 uses cl100k with
    ~100K merges), the curve is much smoother because the training corpus is orders
    of magnitude larger.
    """),

    code(r"""
    # Figure 2 — Loss landscape: pre-training → SFT → alignment (schematic).
    stages = ["Random\ninit", "Mid\npre-train", "Pre-train\ndone", "Post\nSFT", "Post\nRLHF/DPO"]
    clm_loss   = [np.log(50257), 3.5, 2.0, None, None]          # GPT-2 vocab size
    sft_loss_v = [None, None, 2.0, 1.2, None]
    align_r    = [None, None, None, 5.5, 8.2]                   # reward (higher=better)

    x = np.arange(len(stages))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    valid_clm = [(i, v) for i, v in enumerate(clm_loss) if v is not None]
    valid_sft = [(i, v) for i, v in enumerate(sft_loss_v) if v is not None]
    valid_al  = [(i, v) for i, v in enumerate(align_r) if v is not None]

    ax1.plot([i for i, v in valid_clm], [v for i, v in valid_clm], "o-b", label="CLM loss")
    ax1.plot([i for i, v in valid_sft], [v for i, v in valid_sft], "s--g", label="SFT loss")
    ax2.plot([i for i, v in valid_al],  [v for i, v in valid_al],  "^-r", label="Reward")
    ax1.set_xticks(x); ax1.set_xticklabels(stages, fontsize=8)
    ax1.set_ylabel("loss (lower is better)", color="blue")
    ax2.set_ylabel("reward (higher is better)", color="red")
    ax1.set_title("Figure 2 — LLM training stages: loss and reward across the pipeline")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** A schematic of the LLM training pipeline. **CLM loss** (blue)
    starts near $\log(V) \approx 10.8$ for GPT-2's 50K vocabulary (random init) and
    falls to ~2.0 after pre-training — the model has learned English grammar and world
    knowledge. **SFT** (green) fine-tunes on instruction-response pairs; the loss
    drops further because the task is narrower than all of language. **Alignment**
    (RLHF/DPO, red) doesn't necessarily reduce loss — it optimises a **reward** score
    (human preference), which can sometimes trade off perplexity for helpfulness.
    This is why alignment is a separate, complementary stage, not a continuation of
    language modelling.
    """),

    code(r"""
    # Figure 3 — LoRA parameter efficiency: trainable vs frozen params by rank.
    ranks = [1, 2, 4, 8, 16, 32, 64]
    d = 4096
    total_w = d * d
    trainable = [d * r + r * d for r in ranks]
    fractions = [t / total_w * 100 for t in trainable]
    fig, ax = plt.subplots()
    ax.semilogx(ranks, fractions, "o-", base=2)
    ax.set_xlabel("LoRA rank r (log2 scale)"); ax.set_ylabel("% trainable params (of one W matrix)")
    ax.set_title(f"Figure 3 — LoRA rank vs parameter budget (d={d}x{d} matrix)")
    for r, f in zip(ranks, fractions):
        ax.annotate(f"{f:.2f}%", (r, f), textcoords="offset points", xytext=(5, 0), fontsize=8)
    plt.show()
    """),

    md(r"""
    **Figure 3.** LoRA rank vs the percentage of trainable parameters for a single
    $4096 \times 4096$ weight matrix. At rank $r=8$ (the standard default), LoRA uses
    only 0.39% of the parameters of that matrix — yet retains >95% of the fine-tuning
    quality on most tasks. Increasing rank trades efficiency for expressiveness: $r=64$
    approaches the quality of full fine-tuning. The typical production choice is
    $r \in \{8, 16\}$ with $\alpha = 2r$ (so the effective scale is ~2). Applied to
    all 32+ attention projections in a 7B model, this remains well under 1% of total
    parameters.
    """),

    code(r"""
    # Figure 4 — DPO loss as a function of the log-ratio margin.
    margins = np.linspace(-3, 3, 200)               # beta*(ratio_w - ratio_l)
    dpo_loss_curve = -np.log(sigmoid(margins))
    grad_magnitude  = sigmoid(-margins)              # gradient wrt margin
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(margins, dpo_loss_curve, "b-", label="DPO loss")
    ax2.plot(margins, grad_magnitude, "r--", label="gradient magnitude")
    ax1.set_xlabel("beta * (log-ratio winner - log-ratio loser)")
    ax1.set_ylabel("DPO loss", color="blue")
    ax2.set_ylabel("gradient magnitude", color="red")
    ax1.set_title("Figure 4 — DPO loss and gradient: easy vs hard preferences")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2)
    plt.show()
    """),

    md(r"""
    **Figure 4.** The DPO loss (blue) as a function of the margin
    $\beta(\text{ratio}_w - \text{ratio}_l)$. When the policy strongly prefers the
    winner (positive margin), the loss is near zero and the gradient (red, dashed) is
    tiny — these "easy" preferences are already satisfied and contribute little to
    training. When the policy prefers the loser (negative margin), the loss is high
    and the gradient is large — the model must be corrected. This **adaptive focusing
    on hard preferences** is why DPO converges efficiently without the instability of
    PPO's reward maximisation. Note: if the margin is very negative (model strongly
    prefers loser), the gradient *still* saturates — DPO doesn't give special
    treatment to very wrong answers, which can be a failure mode (§7).
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Catastrophic forgetting** | Model loses pre-trained capability after SFT | Full fine-tuning overwrites base weights | LoRA/PEFT; small LR; replay |
    | **Reward hacking (RLHF)** | Policy learns to fool reward model | Reward model not robust; KL penalty too weak | Stronger KL; better RM; constitutional AI |
    | **DPO collapse** | Policy ignores loser, doesn't learn winner | Margin saturates both sides | IPO/cDPO; add KL term; higher beta |
    | **Data contamination** | Inflated eval scores | Test prompts in SFT data | Dedup train vs eval; time-split evals |
    | **Mode collapse (SFT)** | Model gives identical outputs | Over-fitting small SFT set; LR too high | Lower LR; early stopping; diverse data |
    | **BPE vocabulary mismatch** | OOV subwords at inference | Tokeniser trained on different domain | Fine-tune tokeniser or use byte-level BPE |
    | **Length bias (RM)** | Reward model prefers longer answers | Length correlated with perceived quality in training | Length-penalised reward; length-normalised reward |
    | **SFT format sensitivity** | Model breaks if prompt format changes | Format memorised, not generalised | Data augmentation with format variants |
    """),

    code(r"""
    # Demonstrate: catastrophic forgetting via full update vs LoRA.
    # We measure how much the output changes after full vs LoRA update.
    d_demo = 64
    W_pretrained = rng.normal(0, 0.02, (d_demo, d_demo))
    x_demo = rng.normal(0, 1, (8, d_demo))
    out_pretrained = x_demo @ W_pretrained

    # Full fine-tune: update all weights (large step simulating over-fitting).
    W_full = W_pretrained + rng.normal(0, 0.5, W_pretrained.shape)   # large perturbation
    out_full = x_demo @ W_full
    delta_full = np.abs(out_full - out_pretrained).mean()

    # LoRA: only adapters change (B still 0 at init, small A step).
    A = rng.normal(0, 0.01, (d_demo, 4))
    B = rng.normal(0, 0.5, (4, d_demo))    # even large B perturbation
    out_lora = x_demo @ W_pretrained + (x_demo @ A) @ B * (8/4)
    delta_lora = np.abs(out_lora - out_pretrained).mean()

    print(f"Output change: full fine-tune = {delta_full:.4f}")
    print(f"Output change: LoRA (r=4)     = {delta_lora:.4f}")
    print(f"LoRA protects base model output much better than full fine-tuning.")
    print(f"At B=0 (init), delta_lora = 0 exactly; here B is randomly perturbed to show even")
    print(f"a large LoRA perturbation stays controlled due to the low-rank bottleneck.")
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 HuggingFace tokenizer (BPE, production-grade).
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        test = "Tokenization splits words into subword pieces efficiently."
        ids = tok.encode(test)
        decoded = [tok.decode([i]) for i in ids]
        print("GPT-2 BPE tokenizer:")
        print(f"  Input:  '{test}'")
        print(f"  Tokens: {decoded}")
        print(f"  IDs:    {ids}")
        print(f"  Vocab size: {tok.vocab_size}")
    except Exception as e:
        print(f"[transformers not available: {type(e).__name__}]")
        print("In production: AutoTokenizer.from_pretrained('gpt2') for GPT-2 BPE,")
        print("or 'meta-llama/Llama-2-7b' for SentencePiece BPE with 32K vocab.")
    """),

    code(r"""
    # 8.2 LoRA / PEFT with HuggingFace (guarded).
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM
        import torch

        base = AutoModelForCausalLM.from_pretrained("gpt2")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],     # GPT-2 attention projection
            lora_dropout=0.05,
        )
        peft_model = get_peft_model(base, lora_cfg)
        trainable, total = peft_model.get_nb_trainable_parameters()
        print(f"PEFT LoRA model: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.3f}%)")
    except Exception as e:
        print(f"[peft/transformers/torch not available: {type(e).__name__}]")
        print("Production pattern:")
        print("  from peft import LoraConfig, get_peft_model")
        print("  cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj','v_proj'])")
        print("  model = get_peft_model(base_model, cfg)")
        print("  # Then train with standard Trainer or SFTTrainer from trl")
    """),

    code(r"""
    # 8.3 DPO training pattern (trl library, guarded).
    lines = [
        "DPO training with trl -- production pattern (guarded, requires GPU + model):",
        "  from trl import DPOTrainer, DPOConfig",
        "  from transformers import AutoModelForCausalLM, AutoTokenizer",
        "  ",
        "  model     = AutoModelForCausalLM.from_pretrained('llama3-8b')",
        "  ref_model = AutoModelForCausalLM.from_pretrained('llama3-8b')  # frozen ref",
        "  tokenizer = AutoTokenizer.from_pretrained('llama3-8b')",
        "  ",
        "  # dataset: columns 'prompt', 'chosen', 'rejected'",
        "  trainer = DPOTrainer(",
        "      model=model, ref_model=ref_model,",
        "      args=DPOConfig(beta=0.1, max_length=512),",
        "      train_dataset=pref_dataset, tokenizer=tokenizer,",
        "  )",
        "  trainer.train()",
        "",
        "Key: beta=0.1 controls KL-divergence strength from reference policy.",
    ]
    print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Fine-Tuning a Domain LLM for Medical Q&A

    **Scenario.** A healthtech company wants an LLM that answers clinician questions
    about drug interactions and dosing. The base model (Llama-3-8B) is general; they
    need domain accuracy without hallucination.

    **Pipeline chosen:**
    1. **Continual pre-training** on 2B tokens of de-identified clinical notes + medical
       textbooks (domain vocabulary adaptation, $\sim$200 GPU-hours on A100).
    2. **SFT** on 50K clinician-verified Q&A pairs in the ChatML format.
       Loss masked on prompt tokens.
    3. **DPO alignment** on 10K preference pairs where clinicians rated two responses
       and marked one as more accurate/safe.

    **Why not RLHF?** PPO requires a reward model (additional training cost) and is
    unstable — hyperparameter-sensitive, reward-hacking prone. DPO achieves comparable
    quality with simpler training and is the current production default.

    **Why LoRA?** Full fine-tuning of 8B parameters requires 8× A100s and risks
    catastrophic forgetting of medical knowledge already in the base model. LoRA on
    $q$/$v$ projections ($r=16$) trains on 2 A100s in 6 hours.

    **Cost of mistakes:** hallucinated drug dosage → patient harm → liability. This
    mandates: (1) constitutional AI rules in the system prompt; (2) factual
    grounding via RAG (Phase 5); (3) LLM-as-judge eval on a medical golden set
    (Notebook 40); (4) human expert review of random outputs weekly.

    **KPIs:** factual accuracy on medical QA benchmark (MedQA), hallucination rate
    (Notebook 24), p95 latency, cost-per-query.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Data quality > quantity for SFT.** 1K high-quality instruction-response pairs
      often outperform 100K noisy ones. Data curation (filtering, dedup, quality
      scoring) is the highest-leverage step.
    - **Deduplication for pre-training.** Near-duplicate removal (MinHash, exact
      substring) is essential — duplicates inflate effective data size, encourage
      memorisation, and leak test data.
    - **Gradient checkpointing + bf16.** Standard for fine-tuning on limited GPU:
      recompute activations during backward pass to save memory (at cost of ~30% more
      compute); use bfloat16 for numerical stability.
    - **Learning rate schedule.** SFT: cosine decay from $2 \times 10^{-5}$ to $0$.
      LoRA: $1 \times 10^{-4}$ (adapters start from scratch). DPO: $1 \times 10^{-6}$
      (very conservative — small updates relative to SFT model).
    - **Evaluation during training.** Monitor: SFT loss on held-out set, perplexity,
      and task-specific metrics (ROUGE, accuracy). Early-stop on task metric.
    - **Merging LoRA for inference.** After training, merge $W = W_0 + BA\cdot\alpha/r$
      back into the base model weights — zero inference overhead vs the base model.
    - **System prompt as alignment.** For many production use cases, a well-crafted
      system prompt (Notebook 23) combined with SFT is sufficient — RLHF/DPO adds
      complexity and cost that may not be justified.
    - **Versioning.** Track: base model version, LoRA adapter checkpoint, tokeniser
      version, training data hash. A mismatch between any of these breaks behaviour
      silently.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Fine-tuning strategy:**

    | Strategy | Cost | Risk | Quality | When to use |
    |---|---|---|---|---|
    | Prompt engineering | Zero | None | Baseline | Quick experiments (Ntbk 23) |
    | LoRA/PEFT | Low (2–4 GPU) | Low (forgetting) | ≈full FT | **Default for domain adaptation** |
    | Full fine-tuning | High (8-32 GPU) | Forgetting | Best | Sufficient data, compute budget |
    | RAG (no FT) | Near-zero | Staleness | Good for factual | Knowledge-heavy tasks (Ntbk 29) |

    **Alignment strategy:**

    | Strategy | RM needed | Stability | Cost | Quality |
    |---|---|---|---|---|
    | SFT only | No | High | Low | Good baseline |
    | RLHF (PPO) | Yes | Low | High | Best (if tuned) |
    | **DPO** | **No** | **High** | **Low** | **≈RLHF** |
    | ORPO | No | High | Low | Combines SFT+DPO in one pass |

    **LoRA rank selection:**

    | Rank | Trainable % | Quality | Memory | When |
    |---|---|---|---|---|
    | r=1 | ~0.05% | Low | Minimal | Prompt style only |
    | r=8 | ~0.4% | Good | ~1GB | **Default** |
    | r=16 | ~0.8% | Better | ~2GB | Domain-specific tasks |
    | r=64 | ~3% | ≈Full FT | ~8GB | Complex domain shifts |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"Explain BPE tokenisation."* → Start with characters. Iteratively merge the
      most frequent adjacent pair into a new token. Stop at target vocab size. Encoding
      new text: apply merges greedily in training order. Why BPE: handles OOV via
      subwords, no character-level slowness, vocabulary scales with corpus richness.
    - *"SFT vs RLHF vs DPO?"* → SFT: teach format and domain via supervised examples.
      RLHF: learn human preferences via reward model + PPO. DPO: same objective as RLHF
      but reformulated to eliminate the RM — train directly on preference pairs.

    **Deep-dive questions**
    - *"Why does DPO work without a reward model?"* → The RLHF optimal policy
      $\pi^*(y|x) \propto \pi_{\text{ref}} \exp(r/\beta)$ can be inverted to express
      $r$ as a function of $\pi^*$ and $\pi_{\text{ref}}$. Plugging into the Bradley-
      Terry preference model gives a loss purely in terms of policy log-ratios — no RM.
    - *"How does LoRA avoid catastrophic forgetting?"* → $W_0$ is frozen; the full
      pre-trained weight is preserved. $\Delta W = BA$ is initialised to zero so the
      model starts at the same behaviour as the base. Gradients flow only through $A$
      and $B$ — a tiny subspace. Even large adapter perturbations don't corrupt $W_0$.
    - *"What is loss masking in SFT?"* → We compute the cross-entropy only on response
      tokens, not prompt tokens. If we trained on prompt tokens too, the model would
      learn to "imitate" the instruction format, which wastes capacity and can cause
      the model to predict the instruction rather than the response in edge cases.

    **Whiteboard questions**
    - "Write the DPO loss and explain each term." (§4.5)
    - "Draw the BPE merge algorithm step-by-step for the word 'cat'." (§4.1, §5a)

    **Strong vs weak answers**
    - *"When should we use RLHF vs DPO?"*
      - **Weak:** "RLHF is always better."
      - **Strong:** "DPO is the default today: cheaper (no RM training), more stable
        (no PPO), and achieves comparable quality on most tasks. RLHF/PPO may still
        win if you have a very accurate RM (e.g., InstructGPT's scale of human labels),
        or for online RL where the model needs to explore (DPO is offline). In practice,
        start with DPO; only invest in RLHF if DPO plateaus."

    **Common mistakes:** thinking BPE is character-level (it's subword); confusing SFT
    loss (response-only) with pre-training loss (all tokens); not knowing LoRA merging
    at inference; thinking RLHF requires human labels at inference time (no — RM is
    trained offline).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **BPE in one minute.** Walk through the merge algorithm from character tokens to
       subwords. What does the vocabulary contain at the end?
    2. **CLM loss.** Write the formula. What does a loss of 2.0 mean in terms of bits
       per token?
    3. **SFT masking.** Why do we mask the prompt tokens from the loss? What goes wrong
       if we don't?
    4. **RLHF stages.** Name the four steps. What is the reward model trained on?
    5. **DPO.** What is $\pi_{\text{ref}}$ and why is it needed? What does $\beta$
       control?
    6. **LoRA.** Write $h = xW_0 + x(BA)\alpha/r$. Why is $B$ initialised to zero?
    7. **Forgetting.** How does LoRA prevent catastrophic forgetting?
    8. **When to use what.** Your company has 10K instruction-response pairs and 2 A100
       GPUs. What pipeline do you run?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Trace BPE on the word "running": start from ['r','u','n','n','i','n','g','</w>']
       and show 3 merge steps, choosing the pair "n n" as the first merge.
    2. What is the expected cross-entropy loss for a 50K-vocabulary model at random
       initialisation? Compute it.

    **Beginner → Intermediate (coding)**
    3. Extend the BPE implementation to encode unseen words by applying the learned
       merges in order. Encode "fine-tuning" and compare its tokens to "tuning" and
       "fine".
    4. Add loss masking to the causal LM loss: accept a `mask` array where 0 = ignored,
       1 = included. Verify that masking all prompt tokens produces the SFT loss.

    **Intermediate (analysis)**
    5. Implement the RLHF reward model training loss (Bradley-Terry, §4.4) from scratch.
       Generate synthetic (winner, loser) log-probability pairs and train a simple linear
       reward head to predict preference. Measure accuracy on a held-out set.
    6. Show that LoRA with rank $r=d$ is equivalent to full fine-tuning (theoretically
       and by checking that the output matches a direct $\Delta W$ update when $A$ and
       $B$ are set appropriately).

    **Senior (interview + production design)**
    7. *Design:* the full training pipeline for the medical Q&A LLM in §9. Include:
       data curation (dedup, quality filter), continual pre-training compute budget
       (Chinchilla), SFT data size and format, DPO preference collection strategy,
       evaluation (Notebooks 38–40), and A/B rollout plan.
    8. *Scaling:* you have 64 A100 GPUs and a target validation loss of 2.5 nat.
       Using Chinchilla laws, estimate the model size and token count for optimal
       compute use. Then estimate how long training takes assuming 312 TFLOP/s per A100
       at 40% MFU.
    9. *Debugging:* after DPO fine-tuning, your model's outputs become shorter and
       more generic. Identify two likely causes from §7 and propose concrete mitigations.
    """),

    md(r"""
    ---
    ### Summary
    The LLM training pipeline has four stages: **pre-training** (learn language from
    internet-scale text via next-token prediction, budget by Chinchilla), **continual
    pre-training** (domain adaptation), **SFT** (teach instruction-following via
    response-only cross-entropy), and **alignment** (RLHF via PPO+RM, or DPO via
    direct preference pairs — simpler and the current default). **LoRA** makes SFT
    and DPO affordable by training only low-rank adapter matrices ($r \times (d+k)$
    parameters) while keeping the base model frozen.

    **Next:** `23 · Prompt Engineering` — the other side of the LLM interface: how
    to construct system prompts, few-shot examples, chain-of-thought, and structured
    output to maximise model capability without fine-tuning.
    """),
]

build("phase4_nlp_llms/22_llm_training_pipeline.ipynb", cells)
