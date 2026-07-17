"""Builder for Lesson DL-08 — Transformers.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # DL-08 · Transformers
    ### Section 04 — Deep Learning Foundations · *ML/AI Senior Mastery Curriculum*

    > Lesson DL-07 derived the **attention mechanism**. The Transformer (Vaswani et al.,
    > "Attention Is All You Need", 2017) stacks it into a complete architecture: **multi-
    > head self-attention** + **position-wise feed-forward** sub-layers, each wrapped in
    > a **residual connection** and **layer normalisation**, repeated $N$ times, topped
    > by token embeddings and **positional encoding** so the otherwise order-agnostic
    > attention knows where each token sits. Variations of this block underpin GPT,
    > BERT, T5, and many modern language models. We first expose a single-head forward
    > pass in NumPy, then train a real multi-head **GPT-style** decoder with PyTorch
    > autograd on a tiny character-level corpus. The two levels separate transparent
    > arithmetic from a complete learning experiment.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The **Transformer block** anatomy: multi-head self-attention + FFN sub-layers,
      each with a **pre-norm** residual connection.
    - **Positional encoding**: why attention needs it (Lesson DL-07's permutation-
      invariance) and the sinusoidal formula.
    - **Token embeddings** and the **language-model head** (tied-weight linear layer).
    - **GPT (decoder-only, causal)** vs **BERT (encoder-only, bidirectional)** vs
      **encoder–decoder** — when each is used.
    - A transparent single-head forward pass in NumPy followed by a real miniature
      multi-head GPT trained with backpropagation and AdamW.
    - Greedy, temperature, top-k, and top-p generation from learned weights.
    - **Scaling laws**, **pre-training vs fine-tuning**, and the production landscape
      of LLMs that Section 05 builds on.

    **Why it matters**
    - Every topic in Sections 05–07 (LLMs, RAG, Agents) assumes fluent understanding of
      the Transformer architecture. Section 05 starts with how Transformers are trained
      on text (Lesson NLP-03) and extended with prompting and safeguards (NLP-04 and
      NLP-05).

    **Typical interview questions**
    - "Walk me through a Transformer block from scratch."
    - "Why are residual connections and layer norm essential?"
    - "GPT vs BERT — when would you use each?"
    - "What are scaling laws and what do they predict?"
    - "How does positional encoding work and why does the Transformer need it?"
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Before the Transformer.** NLP in 2017 combined LSTMs (Lesson DL-06) with attention
    (Lesson DL-07), using recurrence to process sequences and attention to let the
    decoder look at the encoder. Training was slow (sequential), and very long-range
    dependencies still degraded.

    **"Attention Is All You Need" (Vaswani et al., 2017).** The key step: remove the
    recurrence entirely. Build a model *only* from self-attention and feed-forward
    layers. Every layer is fully parallel, every position attends to every other, and
    depth replaces sequential memory. The Transformer outperformed the best RNN models
    on translation in both quality and training speed — and it scaled.

    **BERT (Devlin et al., 2018)** showed that a *bidirectional* Transformer encoder
    pre-trained on masked-language modelling could be fine-tuned to dominate nearly
    every NLP benchmark. **GPT (Radford et al., 2018–2020)** showed that a *decoder-
    only* Transformer pre-trained on next-token prediction, scaled to billions of
    parameters, developed remarkable few-shot abilities. Both branches led to today's
    LLMs.

    **The scaling-law era.** Kaplan et al. (2020) found that model performance follows
    smooth **power laws** in model size, dataset size, and compute — giving practitioners
    principled budget allocation. Chinchilla (Hoffmann et al., 2022) showed models had
    been over-sized and under-trained; compute-optimal models train on ~20 tokens per
    parameter.

    **Why the Transformer dominates.** Three properties: (1) **fully parallel** training
    (no sequential dependency); (2) **direct long-range access** (attention in $O(1)$
    paths); (3) **scales smoothly** with data and compute following power laws. These
    three together are why it displaced CNNs for vision and LSTMs for text.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The Transformer block as a two-stage filter.** Each block does two things:
    (1) **mix information across positions** (multi-head self-attention — "what context
    from other tokens is relevant here?"); (2) **transform each position independently**
    (FFN — "given the gathered context, how should I update this token's representation?").
    The residual connection ensures each block only needs to learn the *correction* to
    the input — like ResNets (Lesson DL-03's vanishing-gradient fix) applied to NLP.

    **Layer norm stabilises the residual stream.** Without normalisation the residual
    stream's scale grows with depth; layer norm keeps each sub-layer's input well-
    conditioned (analogous to standardisation in FND-04 and MLE-03).

    **Positional encoding is the "address label."** Attention is permutation-invariant
    (Lesson DL-07, §7); to give positions meaning, we add a positional signal to each
    token embedding — sinusoidal patterns at different frequencies so each position has
    a unique, smooth fingerprint that generalises to unseen lengths.

    **Stacking blocks = hierarchical representation.** Early layers learn local,
    syntactic patterns; later layers learn global, semantic ones — the same hierarchy
    as CNNs (Lesson DL-05), but in semantic space.

    ```mermaid
    flowchart LR
        TokEmb["token embedding"] --> PosEnc["+ positional encoding"]
        PosEnc --> B1["Transformer Block 1\n(Attn → Add&Norm → FFN → Add&Norm)"]
        B1 --> B2["Block 2 ... Block N"]
        B2 --> Head["LM head (linear + softmax)"]
        Head --> NextTok["next-token distribution"]
    ```
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    def softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def layer_norm(x, eps=1e-5):
        # Simplified LayerNorm without learned scale or shift.
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + eps)

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    print("NumPy", np.__version__)
    print("helpers defined: softmax, layer_norm, gelu")
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Sinusoidal positional encoding
    For position $p$ and dimension $i$ (out of $d_{model}$):
    $$PE(p,2i)=\sin\!\left(\frac{p}{10000^{2i/d}}\right),\quad
    PE(p,2i{+}1)=\cos\!\left(\frac{p}{10000^{2i/d}}\right).$$
    Different dimensions oscillate at different frequencies; each position gets a unique
    vector. $PE(p+k)$ has a structured relationship to $PE(p)$, which gives the model
    a useful relative-position signal. This structure permits evaluation beyond the
    trained length, but does not guarantee that the model will perform well there.
    Modern models often use
    **RoPE** (Rotary Position Embedding) which bakes relative position into the Q/K
    dot product itself.

    ### 4.2 Transformer block (pre-norm formulation)
    $$\tilde x = x + \text{MHA}(\text{LayerNorm}(x)),\quad
    x' = \tilde x + \text{FFN}(\text{LayerNorm}(\tilde x)).$$
    The **FFN** is a two-layer MLP applied *position-wise*:
    $\text{FFN}(x)=W_2\,\text{GELU}(W_1 x+b_1)+b_2$, with $d_{ff}=4d_{model}$ typically.

    ### 4.3 Language model objective (next-token prediction)
    For a sequence $x_1,\dots,x_T$, the cross-entropy loss (Lesson FND-02) is
    $$J = -\frac{1}{T-1}\sum_{t=1}^{T-1}\log p(x_{t+1}\mid x_1,\dots,x_t).$$
    With causal masking (Lesson DL-07), all $T-1$ next-token predictions are computed
    in parallel — the teacher-forcing trick that makes GPT training fully parallel.

    ### 4.4 GPT vs BERT vs encoder–decoder
    | Architecture | Masking | Training objective | Use case |
    |---|---|---|---|
    | **GPT** (decoder-only) | Causal | Next-token prediction | Generation, in-context learning, general-purpose |
    | **BERT** (encoder-only) | Bidirectional | Masked LM + NSP | Classification, extraction, embeddings |
    | **Encoder-decoder** | Enc: bi-dir; Dec: causal | Seq2seq (source→target) | Translation, summarisation |

    ### 4.5 Scaling laws
    Kaplan et al. observed empirical power-law relationships between validation loss
    and model size, data, and compute over the regimes they measured. Hoffmann et al.
    later found that their studied compute-optimal models used roughly 20 training
    tokens per parameter. Treat this as historical empirical guidance, not a universal
    formula: architecture, data quality, objective, and the measured scale all matter.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    First inspect a minimal single-head causal forward pass in NumPy. This level is not
    trainable because NumPy does not supply the autograd path used here. We then use a
    directly implemented PyTorch decoder—without `nn.Transformer` or a hosted model—to
    perform real multi-head training, validation, checkpoint preparation, and decoding.
    """),

    code(r"""
    # 5.1 Sinusoidal positional encoding.
    def positional_encoding(T, d_model):
        PE = np.zeros((T, d_model))
        positions = np.arange(T)[:, None]
        div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)
        PE[:, 0::2] = np.sin(positions / div_term)
        PE[:, 1::2] = np.cos(positions / div_term[:d_model // 2])
        return PE

    # 5.2 Self-attention (causal).
    def causal_self_attention(x, Wq, Wk, Wv, Wo):
        T, d = x.shape
        Q, K, V = x @ Wq, x @ Wk, x @ Wv
        scale = Q.shape[-1] ** 0.5
        scores = Q @ K.T / scale
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        weights = softmax(scores + mask, axis=-1)
        attn_out = weights @ V
        return attn_out @ Wo

    # 5.3 Feed-forward network (position-wise).
    def ffn(x, W1, b1, W2, b2):
        return gelu(x @ W1 + b1) @ W2 + b2

    # 5.4 A single Transformer block (pre-norm).
    def transformer_block(x, params):
        Wq, Wk, Wv, Wo, W1, b1, W2, b2 = params
        x = x + causal_self_attention(layer_norm(x), Wq, Wk, Wv, Wo)
        x = x + ffn(layer_norm(x), W1, b1, W2, b2)
        return x

    # 5.5 Educational single-head decoder forward pass (1 block, no training yet).
    def init_params(vocab, d_model, d_ff, seed=0):
        r = np.random.default_rng(seed)
        s = 0.02
        tok_emb = r.normal(0, s, (vocab, d_model))
        Wq = r.normal(0, s, (d_model, d_model))
        Wk = r.normal(0, s, (d_model, d_model))
        Wv = r.normal(0, s, (d_model, d_model))
        Wo = r.normal(0, s, (d_model, d_model))
        W1 = r.normal(0, s, (d_model, d_ff))
        b1 = np.zeros(d_ff)
        W2 = r.normal(0, s, (d_ff, d_model))
        b2 = np.zeros(d_model)
        lm_head = tok_emb                               # weight tying: share with embedding
        return tok_emb, (Wq, Wk, Wv, Wo, W1, b1, W2, b2), lm_head

    def forward(token_ids, params_all):
        tok_emb, block_params, lm_head = params_all
        T = len(token_ids)
        x = tok_emb[token_ids] + positional_encoding(T, tok_emb.shape[1])
        x = transformer_block(x, block_params)
        x = layer_norm(x)
        logits = x @ lm_head.T                          # (T, vocab_size)
        return logits

    vocab_size = 27                                      # a-z + space
    d_model, d_ff = 32, 64
    params_all = init_params(vocab_size, d_model, d_ff)
    # Test forward pass shape
    dummy = np.array([0, 1, 2, 3, 4])
    logits = forward(dummy, params_all)
    print(f"Forward pass shape: {logits.shape}  (T x vocab_size) -- OK")
    """),

    code(r"""
    # 5.6 Real training: import the inspectable project implementation.
    import sys
    from pathlib import Path
    import torch

    candidates = [Path.cwd(), *Path.cwd().parents]
    repo_root = next(path for path in candidates if (path / "projects/tiny_language_model").exists())
    project_root = repo_root / "projects" / "tiny_language_model"
    sys.path.insert(0, str(project_root / "src"))

    from tiny_language_model.training import train

    corpus = (project_root / "data" / "learning_corpus.txt").read_text(encoding="utf-8")
    trained_model, tokenizer, training_report = train(
        corpus,
        seed=42,
        max_epochs=8,
        batch_size=16,
        config_overrides={"block_size": 32, "d_model": 32, "n_heads": 4, "n_layers": 1},
    )
    first_epoch = training_report["history"][0]
    last_epoch = training_report["history"][-1]
    print("initial validation loss:", round(training_report["initial_validation_loss"], 3))
    print("first epoch:", {key: round(value, 3) if isinstance(value, float) else value for key, value in first_epoch.items()})
    print("last epoch:", {key: round(value, 3) if isinstance(value, float) else value for key, value in last_epoch.items()})
    print("best validation loss:", round(training_report["best_validation_loss"], 3))
    print("bigram validation loss:", round(training_report["bigram_validation_loss"], 3))
    assert training_report["best_validation_loss"] < training_report["initial_validation_loss"]
    """),

    code(r"""
    # 5.7 Generation from learned weights. Greedy is the deterministic baseline.
    prompt = "the model"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    greedy_ids = trained_model.generate(prompt_ids.clone(), 50, temperature=0)
    top_k_ids = trained_model.generate(
        prompt_ids.clone(), 50, temperature=0.8, top_k=8,
        generator=torch.Generator().manual_seed(42),
    )
    top_p_ids = trained_model.generate(
        prompt_ids.clone(), 50, temperature=0.8, top_p=0.9,
        generator=torch.Generator().manual_seed(42),
    )
    print("greedy:", tokenizer.decode(greedy_ids[0].tolist()))
    print("top-k :", tokenizer.decode(top_k_ids[0].tolist()))
    print("top-p :", tokenizer.decode(top_p_ids[0].tolist()))
    print("These samples demonstrate learned continuation mechanics, not factual reliability.")
    """),

    md(r"""
    ## 6 · Visualization

    Three figures: the sinusoidal positional encoding pattern, scaling laws, and a
    comparison of the attention-pattern diversity across two blocks in a deeper model.
    """),

    code(r"""
    # Figure 1 — Sinusoidal positional encoding: each position has a unique pattern.
    T, d = 50, 64
    PE = positional_encoding(T, d)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].imshow(PE.T, aspect="auto", cmap="RdBu")
    axes[0].set_xlabel("position"); axes[0].set_ylabel("dimension")
    axes[0].set_title("Positional encoding matrix (sinusoidal, d=64)")
    # frequency structure along a few dimensions
    for dim in [0, 2, 6, 14]:
        axes[1].plot(PE[:, dim], label=f"dim {dim}")
    axes[1].set_xlabel("position"); axes[1].set_ylabel("encoding value")
    axes[1].set_title("Individual dimensions: high-freq to low-freq")
    axes[1].legend(fontsize=8)
    plt.suptitle("Figure 1 — Positional encoding: unique, smooth fingerprint per position")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** The sinusoidal positional encoding assigns each position a unique
    vector of sine/cosine values at varying frequencies (left: the full matrix —
    dark/light stripes oscillate faster at low dimensions, slower at high). Individual
    dimensions (right) show the progression from fast to slow oscillation. The model
    adds this to the token embedding, so the attention scores $QK^\top$ implicitly
    depend on position. The smooth, deterministic pattern generalises to lengths unseen
    in training; modern models replace this with **RoPE** (Rotary Position Embedding),
    which encodes relative position directly into the Q/K dot product for better length
    extrapolation.
    """),

    code(r"""
    # Figure 2 — Scaling law schematic: loss ~ N^(-alpha) (conceptual).
    Ns = np.logspace(6, 12, 100)                        # 1M to 1T params
    alpha = 0.076                                       # Kaplan et al. exponent (approx)
    loss = 2.5 * (Ns / 1e6) ** (-alpha)                # illustrative power law
    data_balanced = 2.5 * (Ns / 1e6) ** (-alpha * 1.15)  # illustrative, not fitted data
    fig, ax = plt.subplots()
    ax.loglog(Ns / 1e9, loss, label="fixed dataset (original GPT3 regime)")
    ax.loglog(Ns / 1e9, data_balanced, "--", label="illustrative data-balanced curve")
    ax.set_xlabel("model size (B parameters)"); ax.set_ylabel("validation loss (schematic)")
    ax.set_title("Figure 2 — Scaling laws: loss is a smooth power law in model size")
    ax.legend()
    plt.show()
    print("Conceptual plot only: these curves are not measurements from this notebook.")
    print("Scaling exponents and compute-optimal allocations depend on the measured regime.")
    """),

    md(r"""
    **Figure 2.** This is a labelled schematic, not benchmark evidence. Published work
    has observed approximate power-law relationships over specific ranges of models,
    data, and compute. The useful engineering lesson is to measure a small scaling
    frontier for the actual architecture and data before extrapolating a large run;
    a single tokens-per-parameter ratio is not a universal law.
    """),

    code(r"""
    # Figure 3 — show that positional encoding makes attention position-sensitive.
    d_small = 16
    PE_small = positional_encoding(8, d_small)
    no_pe = np.zeros_like(PE_small)

    # same tokens but once with PE, once without -> different attention patterns
    tok_emb_test = rng.normal(0, 0.1, (vocab_size, d_small))
    ids_test = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    Wq_t = rng.normal(0, 0.1, (d_small, d_small))
    Wk_t = rng.normal(0, 0.1, (d_small, d_small))

    def attn_weights(ids, pe, tok_emb, Wq, Wk):
        x = tok_emb[ids] + pe
        Q, K = x @ Wq, x @ Wk
        scores = Q @ K.T / Q.shape[-1] ** 0.5
        mask = np.triu(np.ones((len(ids), len(ids))), k=1) * -1e9
        return softmax(scores + mask, axis=-1)

    W_with_pe = attn_weights(ids_test, PE_small, tok_emb_test, Wq_t, Wk_t)
    W_no_pe   = attn_weights(ids_test, no_pe, tok_emb_test, Wq_t, Wk_t)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (w, title) in zip(axes, [(W_no_pe, "Without PE (only token identity)"),
                                      (W_with_pe, "With sinusoidal PE (token + position)")]):
        im = ax.imshow(w, cmap="viridis")
        ax.set_title(title); plt.colorbar(im, ax=ax)
    plt.suptitle("Figure 3 — Positional encoding changes attention patterns (position now matters)")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** Without positional encoding the attention pattern depends only on
    token identity — any permutation of the same tokens would produce an identical
    heatmap (permuted). With sinusoidal PE the patterns change, because the model can
    now distinguish "token A at position 2" from "token A at position 6." This is why
    every Transformer requires positional encoding: the self-attention computation is
    otherwise *completely* position-agnostic (Lesson DL-07 §7).
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Training instability** | Loss spikes / NaN early | Pre-norm not applied; LR too high | Use pre-norm; warm-up LR schedule |
    | **Attention sink** | All attention to one token ("sink token") | Residual stream bias | Add explicit sink token; SoftMax-off strategies |
    | **Length generalisation** | Degrades on sequences longer than training | Sinusoidal PE doesn't extrapolate | RoPE/ALiBi encodings; longer training |
    | **O(T²) OOM** | GPU OOM on long context | Full attention matrix | Flash Attention; sliding-window; chunked attention |
    | **Catastrophic forgetting** | Fine-tuned model forgets pre-training | Full-parameter fine-tuning | LoRA/PEFT; small learning rate; replay |
    | **Hallucination** | Plausible but false generation | LM objective only maximises token prob | RLHF/DPO alignment; RAG grounding (NLP-05) |
    | **Over-smoothing** | Deep models lose token distinction | Layer-norm + residual collapse | Residual dropout; skip connections |
    | **Data contamination** | Inflated benchmark scores | Test data in pre-training | Decontamination; time-split evaluation |
    """),

    code(r"""
    # Demonstrate: without pre-norm, residual stream variance grows with depth.
    d = 32; T = 10; n_layers = 8
    x = rng.normal(0, 1, (T, d))
    # Without layer norm: variance grows
    x_no_norm = x.copy()
    vars_no_norm = [x_no_norm.var()]
    for _ in range(n_layers):
        W = rng.normal(0, 0.1, (d, d))
        x_no_norm = x_no_norm + x_no_norm @ W               # residual without norm
        vars_no_norm.append(x_no_norm.var())
    # With pre-norm: variance stays controlled
    x_prenorm = x.copy()
    vars_prenorm = [x_prenorm.var()]
    for _ in range(n_layers):
        W = rng.normal(0, 0.1, (d, d))
        x_prenorm = x_prenorm + layer_norm(x_prenorm) @ W   # pre-norm residual
        vars_prenorm.append(x_prenorm.var())

    fig, ax = plt.subplots()
    ax.plot(vars_no_norm, "o-", label="No LayerNorm (explodes)")
    ax.plot(vars_prenorm, "s-", label="Pre-norm (stable)")
    ax.set_xlabel("depth (layer)"); ax.set_ylabel("variance of residual stream")
    ax.set_title("Figure 4 — LayerNorm prevents residual stream explosion with depth")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 4.** Without layer normalisation the residual stream's variance grows
    exponentially with depth (the same vanishing/exploding problem as Lesson DL-03,
    now in the *scale* of activations). **Pre-norm** (applying LayerNorm *before*
    the sub-layer, then adding the un-normed residual) keeps variance controlled
    across all depths, enabling stable training of 96-layer models. This is one of
    the key architectural differences between original (post-norm) and modern (pre-
    norm) Transformers.
    """),

    md(r"""
    ## 8 · Production Library Implementation

    In production you never build a Transformer from scratch — you use HuggingFace
    `transformers` (model loading, fine-tuning, generation) or PyTorch directly.
    The key production workflow: **load a pretrained checkpoint** → **tokenize**
    with the model's vocabulary → **generate** or **fine-tune** (LoRA/PEFT for
    efficiency). The guard ensures the notebook runs without transformers/torch.
    """),

    code(r"""
    # Production: load a tiny HuggingFace model and generate text. Guarded.
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        # Use a very small model for demo (gpt2 is the smallest widely available)
        # Note: this requires internet access to download; skip gracefully if not available.
        model_name = "gpt2"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        prompt = "The Transformer architecture"
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        print(tok.decode(out[0]))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\\nGPT-2 parameters: {n_params:,}")
    except Exception as e:
        print(f"[HuggingFace/torch not available or no network: {type(e).__name__}]")
        print("In production: AutoModelForCausalLM.from_pretrained('gpt2') + tokenizer")
        print("is the standard 3-line inference pattern.")
        print("The scratch mini-GPT above demonstrates the same forward-pass mechanics.")
    """),

    md(r"""
    **Teaching model vs production model.** The NumPy forward pass and trained PyTorch
    project expose the common decoder pattern—embedding, positional information,
    attention, FFN, normalization, residual paths, and an LM head. Production model
    families vary in normalization, positional method, attention layout, activation,
    tokenization, parallelism, and many other details. The project is an architectural
    microscope, not a small replica of any proprietary model.
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Fine-Tuning a Transformer for Customer Support

    **Scenario.** A company wants to automate tier-1 customer support using an LLM that
    answers product-specific questions accurately, stays on-brand, and avoids
    hallucinating policy details.

    **Architecture choice:**
    - **Decoder-only (GPT-style)** for open-ended response generation.
    - **Start from a suitable pre-trained base** when its license, data policy,
      evaluation, and deployment constraints fit. Training cost depends strongly on
      scale and infrastructure, so estimate it for the actual workload.
    - **LoRA/PEFT** fine-tuning: freeze base weights and train small adapter matrices.
      This reduces trainable parameters, but does not guarantee retention or quality.

    **Business objectives:** high factual accuracy on product/policy questions; latency
    under 2 s; avoid harmful or off-brand outputs.

    **Cost of mistakes (asymmetric):** hallucinated policy → customer churn, legal;
    slow latency → abandonment; off-brand tone → brand damage.

    **Constraints:** context window (must fit conversation + retrieved docs — Notebook
    29); fine-tuning compute budget; inference cost per query; monitoring for
    hallucination and drift (NLP-05 and PROD-05).

    **KPIs:** answer accuracy on a golden QA set, hallucination rate (Lesson NLP-05),
    P90 latency, customer satisfaction score, and escalation rate to human agents.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Flash Attention** (IO-aware exact attention, Lesson DL-07) is the default for
      training efficiency — ~2–4× speedup and $O(T)$ memory.
    - **KV caching** for inference: cache key/value projections of all past tokens so
      each new token only computes $O(T)$ work rather than $O(T^2)$.
    - **Quantization** (INT8/INT4): reduce model size and inference cost with minimal
      quality loss — essential for edge/cost-sensitive deployments.
    - **LoRA/PEFT fine-tuning**: update low-rank adapters rather than every base-model
      weight. Measure retention and task quality; limited updates do not prevent every
      form of regression.
    - **Alignment** (RLHF/DPO): a language model trained on next-token prediction is
      not intrinsically helpful or safe — alignment is a separate production step
      (Lesson NLP-05).
    - **Context window management** (Lesson RAG-02): documents that don't fit the context
      must be chunked and retrieved (RAG, Section 06), not truncated naively.
    - **Monitoring**: track output distribution, hallucination rate, and latency; set
      up RLHF/human-feedback loops for continuous improvement (EVAL-04 and EVAL-05).
    - **Scaling budget (§4.5)**: use scaling laws to predict quality before committing
      to a large training run.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Architecture variants:**

    | Architecture | Masking | Parallelism | Use for |
    |---|---|---|---|
    | Decoder-only (GPT) | Causal | Full | Generation, ICL, general-purpose LLMs |
    | Encoder-only (BERT) | Bidirectional | Full | Classification, embeddings, extraction |
    | Encoder-decoder (T5, BART) | Enc: bi-dir; Dec: causal | Full | Translation, summarisation |

    **Positional encoding:**

    | PE | Pros | Cons |
    |---|---|---|
    | Sinusoidal (original) | No learned params; relative-position expressible | Limited length extrapolation |
    | Learned absolute | Flexible | Doesn't extrapolate beyond trained length |
    | **RoPE** | Encodes relative offsets in Q/K rotations; widely used | Long-length behavior still needs evaluation |
    | ALiBi | Simple distance-dependent score bias | Quality and extrapolation depend on task and training |

    **Fine-tuning strategy:**

    | Strategy | Cost | Risk | Use |
    |---|---|---|---|
    | Full fine-tuning | High | Catastrophic forgetting | Enough data, enough budget |
    | **LoRA/PEFT** | Lower trainable state | Still needs regression testing | Resource-constrained adaptation |
    | Prompt engineering | Zero | None | Quick experiments (NLP-04) |
    | RAG | Low (no gradient) | Freshness | Factual/knowledge-heavy tasks (Section 06) |

    **Senior lesson:** the Transformer's power comes from parallel, direct-access
    attention and smooth scaling — but its $O(T^2)$ attention cost and the alignment
    gap between next-token prediction and helpful behaviour are the two enduring
    production challenges.
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Walk me through a Transformer block.* → MHA sub-layer + residual + layer norm →
      FFN sub-layer + residual + layer norm; pre-norm modern standard (§4.2).
    - *Why layer norm and residual connections?* → Residual: each block learns a
      correction, not a full transformation (Lesson DL-03); layer norm: stabilises scale
      (Fig 4, FND-04 and MLE-03).

    **Deep-dive questions**
    - *GPT vs BERT — when each?* → GPT: generation/ICL; BERT: classification/embedding
      (§4.4, §11).
    - *What are scaling laws?* → Power-law relationship between loss and (params, data,
      compute); Chinchilla says ~20 tokens per param is compute-optimal (§4.5).
    - *Flash Attention?* → IO-aware tiling of the $T^2$ attention computation to SRAM;
      exact, $O(T)$ memory, ~2–4× faster (Lesson DL-07 §10).

    **Whiteboard questions**
    - "Write positional encoding formulas and explain why we need them." (§4.1, §5.1.)
    - "Describe the full GPT forward pass from token IDs to logits." (§5.5.)

    **Strong vs weak answers**
    - *"Should we pre-train or fine-tune for our domain?"*
      - **Weak:** "Pre-train from scratch for best results."
      - **Strong:** "First compare prompting, retrieval, and adaptation against a
        measured baseline. If weight updates are justified, evaluate LoRA against full
        tuning under the same data and holdout; lower trainable state does not guarantee
        equal quality or prevent every regression."
    - *"Our Transformer loses coherence on sequences >4K tokens."*
      - **Weak:** "Use a larger model."
      - **Strong:** "The model's PE and attention patterns may not extrapolate beyond
        its training length. Switch to RoPE or ALiBi for better length generalisation,
        and/or use sliding-window/sparse attention; if factual grounding is the goal,
        RAG (Section 06) is cleaner than extending context blindly."

    **Follow-ups:** "LoRA — what does it do?" (low-rank adapter matrices on attention
    projections). "How does KV caching work?" (cache past K/V, only compute new token's
    Q). "What is temperature sampling?" (divide logits by T before softmax).

    **Common mistakes:** not knowing pre-norm vs post-norm; thinking attention is
    causal by default (it's not — mask is explicit); confusing scaling law dimensions;
    forgetting positional encoding; dismissing fine-tuning for pre-training.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Describe the Transformer block: attention, FFN, residual, norm.
    2. **Why was it invented?** What RNN/LSTM limits does it fix (Lesson DL-06)?
    3. **How does it work?** Walk the GPT forward pass: tokens → embeddings → PE →
       blocks → LM head → next-token probabilities.
    4. **Why does it work?** Why do residual connections and layer norm matter?
    5. **When to use it?** GPT vs BERT vs encoder-decoder — one-sentence each.
    6. **When NOT to use it?** When would you pick a simpler model?
    7. **Tradeoffs?** LoRA vs full fine-tuning; sinusoidal vs RoPE PE; scaling law
       implications.
    8. **How would you productionize it?** Flash Attention, KV cache, LoRA, alignment,
       context management, monitoring.
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Why does a Transformer need positional encoding even though an LSTM doesn't?
    2. What does the language-model training objective (next-token prediction) actually
       optimise, and how does that differ from "understanding"?

    **Beginner → Intermediate (coding)**
    3. Run the project one-batch overfit diagnostic. If loss does not fall sharply,
       inspect target shifting, the causal mask, gradients, and optimizer step in order.
    4. Compare greedy, temperature, top-k, and top-p under one fixed prompt and seed.
       Describe diversity without treating one sample as a quality evaluation.

    **Intermediate (analysis)**
    5. Replace sinusoidal PE with **random absolute PE** (learned during training) in
       the mini-GPT and compare the attention patterns in Figure 3.
    6. Implement a simple **LoRA adapter** on top of $W_q$: add two low-rank matrices
       $AB$ ($r=4$) instead of updating $W_q$ directly and show the parameter count
       reduction.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive the full forward pass of the GPT block from first principles,
       counting FLOPs for the attention and FFN components.
    8. *Design:* the fine-tuning pipeline for the customer support LLM of §9 — model
       selection, LoRA vs full fine-tuning, data curation, evaluation (Lesson EVAL-02),
       alignment (Lesson NLP-05), and drift monitoring (Lesson PROD-05).
    9. *Scaling:* you have a compute budget of $10^{23}$ FLOP. Using Chinchilla scaling
       laws, estimate the optimal model size and token count; compute the expected
       validation loss reduction vs a 1B-parameter over-trained baseline.
    """),

    md(r"""
    ---
    ### Summary
    The Transformer stacks **multi-head self-attention + position-wise FFN** sub-layers,
    each wrapped in a **residual connection and layer normalisation** (Fig 4), with a
    **positional encoding** to give attention its missing sense of order (Fig 1). This
    fully-parallel architecture scales smoothly with data and compute (**scaling laws**,
    Fig 2) and is the foundation of many modern LLM families. We exposed the arithmetic
    in NumPy, then trained a real multi-head decoder with PyTorch (§5), verified that
    validation loss falls, and generated from learned weights. We traced each component
    to its motivation: residuals from Lesson DL-03, attention from Lesson DL-07, LN from
    Lesson FND-04, loss from Lesson FND-02.

    **Section 04 (Deep Learning) is now complete.** You can build the full deep-learning
    stack from first principles: MLP → backprop → CNN → RNN/LSTM → attention →
    Transformer. Complete `projects/tiny_language_model/MASTERY_CHECKPOINT.md` before
    moving into sentence embeddings, alignment, prompting, or RAG.

    **Related lesson:** `Section 05 — NLP and LLMs` begins with `NLP-01 · TF-IDF and Word Embeddings` —
    how language was represented before Transformers and how those ideas live on inside
    them (word2vec/GloVe as the precursors to contextual embeddings).
    """),
]

build("04_deep_learning/08_transformers.ipynb", cells)
