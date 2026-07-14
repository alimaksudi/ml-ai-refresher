"""Builder for Notebook 18 — Attention Mechanism.

Run:  python3 tools/builders/phase3_18_attention.py
Emits: notebooks/phase3_deep_learning/18_attention_mechanism.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # 18 · The Attention Mechanism
    ### Phase 3 — Deep Learning Foundations · *ML/AI Senior Mastery Curriculum*

    > Notebook 17 ended with a hard limit: RNNs must process sequences **one step at
    > a time**, and all past context is squeezed through a fixed-size hidden state
    > bottleneck. Long-range dependencies still bleed out, even with an LSTM. The
    > **attention mechanism** removes *both* constraints: every position can look
    > directly at every other position in a single parallel operation, with no fixed
    > memory bottleneck. Bahdanau et al. introduced the idea for machine translation
    > in 2015; the Transformer (Notebook 19) replaced the recurrence entirely with
    > **self-attention stacks** and became the backbone of all modern LLMs. This
    > notebook derives attention from scratch — query, key, value, scaled dot-product,
    > softmax, causal masking, multi-head — and shows why each choice matters.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The two RNN limits attention was invented to fix: **sequential computation** and
      the **fixed-size memory bottleneck** (Notebook 17 link).
    - The **query / key / value** abstraction and what each represents.
    - **Scaled dot-product attention**: $\text{Attention}(Q,K,V)=\text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right)V$ — derived from cosine-similarity intuition (Notebook 01).
    - *Why* divide by $\sqrt{d_k}$: variance of the dot-product and softmax saturation.
    - **Causal (decoder) masking** for autoregressive generation.
    - **Multi-head attention**: run $H$ independent heads, concatenate, project.
    - **Self-attention vs cross-attention** — and where each appears.
    - The **O(T²) cost** of attention and why it matters at scale.

    **Why it matters**
    - Attention is the core primitive of every modern LLM (GPT, Claude, Gemini).
      Phase 4 (NLP/LLMs) and Phases 5–6 (RAG, Agents) all assume fluent understanding
      of what attention computes and how it scales.
    - Retrieval-augmented generation (Phase 5) is conceptually an external attention
      over a document store; cross-attention in encoder–decoder models is the original
      application.

    **Typical interview questions**
    - "Walk me through scaled dot-product attention from scratch."
    - "Why do we divide by √d_k?"
    - "What is multi-head attention and why use multiple heads?"
    - "Self-attention vs cross-attention — when is each used?"
    - "What is the computational complexity of self-attention and why does it matter?"
    """),

    md(r"""
    ## 2 · Historical Motivation

    **The RNN bottleneck (recap of Notebook 17).** An LSTM processes tokens one by one
    and compresses all past context into a fixed-size vector $h_t$. Two problems remain:
    (1) token $t$ can only reach token $1$ by traversing $t{-}1$ sequential steps —
    an $O(T)$ information path, and gradients along it still attenuate; (2) the
    fixed-size $h_t$ is a *lossy* summary of an arbitrarily long sequence.

    **Bahdanau attention (2015).** In neural machine translation, the decoder needs to
    focus on *different parts of the source sentence* at each output step. Bahdanau
    et al. learned a soft alignment — a **weighted sum of all encoder hidden states**,
    with weights computed by a small network. Suddenly the decoder could skip the
    bottleneck and look directly at any source token. Accuracy jumped sharply on long
    sentences.

    **Luong attention, self-attention, "Attention Is All You Need" (2017).** After
    Bahdanau, attention was applied not just from decoder to encoder (cross-attention)
    but *within* the same sequence (**self-attention**). Vaswani et al. removed
    recurrence entirely: a **Transformer** built exclusively from self-attention and
    feed-forward layers, trainable fully in parallel. It outperformed LSTM models on
    translation and sparked the entire modern LLM era — GPT, BERT, Claude, etc.

    **The key insight: direct access, parallel computation.** Every position can
    attend to every other position in *one matrix multiply* — $O(1)$ information
    path, $O(T^2)$ total computation (the new cost). Training is fully parallel across
    positions, which is why Transformers scale to billions of parameters on modern
    hardware in ways RNNs fundamentally cannot.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The library analogy.** You walk into a library with a **query** ("books about
    gradient descent"). Each book has a **key** on its spine (the title/topic). You
    compare your query to every key and decide how *relevant* each book is (a score).
    You then take a **weighted blend of the books' contents** (values), concentrating
    on the most relevant ones. Attention does exactly this — for every token in a
    sequence, it computes a soft match against all other tokens and retrieves a
    weighted sum of their representations.

    **Three roles in the computation:**
    - **Q (query)**: what this position is *looking for*.
    - **K (key)**: what each position *advertises as its content*.
    - **V (value)**: what each position *contributes when attended to*.

    **Self-attention**: Q, K, V all come from the *same* sequence (each token queries
    the others in the same sentence). **Cross-attention**: Q comes from one sequence
    (decoder), K and V from another (encoder).

    **Why dot-product similarity?** Notebook 01 showed that $\cos\theta=\mathbf u^\top\mathbf v/(\|\mathbf u\|\|\mathbf v\|)$ measures how aligned two vectors are. The dot product $q^\top k$ is the unnormalized version — directly measuring how well a query matches a key. Softmax turns these scores into a probability distribution over which positions to attend.

    **Multi-head: parallel perspectives.** A single head learns one pattern of
    relationships. Multiple heads learn different patterns simultaneously (syntax,
    coreference, position, …) and the outputs are concatenated — more expressive and
    more robust.

    ```mermaid
    flowchart LR
        X["input X"] --> Wq["× W_Q"] --> Q["Q"]
        X --> Wk["× W_K"] --> K["K"]
        X --> Wv["× W_V"] --> V["V"]
        Q --> Scores["QK^T / sqrt(d_k)"]
        K --> Scores
        Scores --> Mask["+ mask (opt)"]
        Mask --> SM["softmax"]
        SM --> AttnW["attention weights"]
        AttnW --> Out["× V = output"]
    ```
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    print("NumPy", np.__version__)

    def softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)   # numerical stability
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Scaled dot-product attention
    Given matrices $Q\in\mathbb R^{T_q\times d_k}$, $K\in\mathbb R^{T_k\times d_k}$,
    $V\in\mathbb R^{T_k\times d_v}$:
    $$\boxed{\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}$$
    The $T_q\times T_k$ score matrix contains the raw dot-product similarity of each
    query to each key; softmax normalizes across keys (rows) to get attention weights;
    the result is a weighted sum of values.

    ### 4.2 Why divide by √d_k?
    If $q,k$ are independent with zero mean and unit variance, then
    $q^\top k=\sum_{i=1}^{d_k}q_i k_i$ has variance $d_k$ (sum of $d_k$ unit-variance
    terms). For large $d_k$ the pre-softmax logits grow large in magnitude, pushing
    softmax into the **saturation region** (gradient ≈ 0 — the sigmoid problem again,
    Notebook 15). Dividing by $\sqrt{d_k}$ restores variance ≈ 1 regardless of
    embedding dimension.

    ### 4.3 Causal (auto-regressive) masking
    For a *decoder* generating one token at a time, position $t$ must not attend to
    future positions $t{+}1,\dots,T$. We add $-\infty$ (implemented as a large
    negative constant) to those entries *before* the softmax; they become exactly 0
    in the attention weights. This is the upper-triangular mask.

    ### 4.4 Multi-head attention
    Run $H$ attention heads in parallel, each with its own projections:
    $$\text{head}_i=\text{Attention}(QW_i^Q,\,KW_i^K,\,VW_i^V),\qquad
    \text{MultiHead}=\text{Concat}(\text{head}_1,\dots,\text{head}_H)W^O.$$
    Each head sees a lower-dimensional space ($d_k = d_{\text{model}}/H$) and can
    specialise on different relationship types. The output projection $W^O$ mixes
    the heads back.

    ### 4.5 Self-attention vs cross-attention
    - **Self-attention**: $Q=K=V=XW$ (same sequence; every token attends to all
      others). Used inside Transformer encoder/decoder layers.
    - **Cross-attention**: $Q$ from one sequence (decoder), $K,V$ from another
      (encoder outputs or retrieved documents). Original Bahdanau use case; reappears
      in RAG (Phase 5) as external attention over retrieved chunks.

    ### 4.6 Complexity
    The score matrix $QK^\top$ is $O(T^2 d_k)$ in time and $O(T^2)$ in memory. For
    short sequences this is fine; for long contexts ($T>4096$) it's the binding
    constraint driving research into **sparse attention**, **linear attention**,
    **Flash Attention** (IO-aware), and **Mamba** (state-space models).
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    We implement **scaled dot-product attention**, **causal masking**, and
    **multi-head attention** entirely in NumPy — then verify multi-head against
    a reference projection and confirm shape + masking correctness.
    """),

    code(r"""
    # 5.1 Scaled dot-product attention from scratch (the core formula).
    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)    # (batch, T_q, T_k)
        if mask is not None:
            scores = scores + mask                          # -1e9 where masked
        weights = softmax(scores, axis=-1)
        return weights @ V, weights                         # output and weights for viz

    # toy sequence: T=6 tokens, d_model=8
    T, d_model = 6, 8
    X = rng.normal(size=(T, d_model))                       # input sequence (no batch dim)
    # random projection weights (no bias for simplicity)
    Wq = rng.normal(size=(d_model, d_model))
    Wk = rng.normal(size=(d_model, d_model))
    Wv = rng.normal(size=(d_model, d_model))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                      # project: shape (T, d_model)

    out, attn_w = scaled_dot_product_attention(Q, K, V)
    print(f"input  shape: {X.shape}")
    print(f"Q/K/V  shape: {Q.shape}")
    print(f"output shape: {out.shape}")
    print(f"attn weights sum per row (should be 1.0): {attn_w.sum(axis=-1)}")
    """),

    code(r"""
    # 5.2 Why sqrt(d_k) matters: variance of dot-products grows with d_k.
    d_ks = [8, 32, 128, 512]
    for d in d_ks:
        q_test = rng.normal(size=(100, d))
        k_test = rng.normal(size=(100, d))
        dots = (q_test * k_test).sum(axis=1)               # dot product per pair
        scaled = dots / np.sqrt(d)
        print(f"d_k={d:4d}:  var(q.k)={dots.var():.1f}   "
              f"var(q.k/sqrt(d))={scaled.var():.2f}  "
              f"max_softmax_logit={abs(dots).max():.1f}")
    print("\\nWithout scaling: logits explode -> softmax saturates (like sigmoid, Ntbk 15).")
    print("With 1/sqrt(d_k): variance stays ~1 regardless of embedding size.")
    """),

    code(r"""
    # 5.3 Causal mask: position t cannot see t+1, t+2, ...
    def causal_mask(T):
        mask = np.triu(np.ones((T, T)), k=1) * -1e9        # upper-tri = -inf
        return mask

    mask = causal_mask(T)
    out_causal, attn_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("Causal attention weight matrix (row i = where token i attends):")
    print(np.round(attn_causal, 3))
    print("\\nUpper triangle is ~0 (future positions masked out) -> autoregressive safe.")
    """),

    code(r"""
    # 5.4 Multi-head attention from scratch.
    def multi_head_attention(X, n_heads, d_model, mask=None, seed=0):
        r = np.random.default_rng(seed)
        assert d_model % n_heads == 0
        d_k = d_model // n_heads
        T = X.shape[0]
        # independent projection weights per head
        Wqs = [r.normal(0, 0.1, (d_model, d_k)) for _ in range(n_heads)]
        Wks = [r.normal(0, 0.1, (d_model, d_k)) for _ in range(n_heads)]
        Wvs = [r.normal(0, 0.1, (d_model, d_k)) for _ in range(n_heads)]
        Wo  = r.normal(0, 0.1, (d_model, d_model))
        head_outputs = []
        all_weights = []
        for i in range(n_heads):
            Qi, Ki, Vi = X @ Wqs[i], X @ Wks[i], X @ Wvs[i]
            head_i, w_i = scaled_dot_product_attention(Qi, Ki, Vi, mask=mask)
            head_outputs.append(head_i)             # (T, d_k)
            all_weights.append(w_i)                 # (T, T) — one per head
        concat = np.concatenate(head_outputs, axis=-1)  # (T, d_model)
        return concat @ Wo, all_weights

    n_heads = 4
    out_mha, head_weights = multi_head_attention(X, n_heads=n_heads, d_model=d_model)
    print(f"Multi-head attention output shape: {out_mha.shape}  (T x d_model)")
    print(f"Number of attention maps: {len(head_weights)} heads, each {head_weights[0].shape}")
    """),

    md(r"""
    ## 6 · Visualization

    Four figures: the **$\sqrt{d_k}$ softmax-saturation** effect, the **attention
    weight heatmap** (self-attention over a toy sentence), the **causal mask** zeroing
    upper-triangle entries, and **four heads** specialising on different patterns.
    """),

    code(r"""
    # Figure 1 — softmax saturation: large logits -> peaked distribution, tiny gradients.
    z_scales = [("no scaling (d=128)", np.linspace(-20, 20, 200)),
                ("with /sqrt(d) scaling", np.linspace(-3, 3, 200))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (label, z) in zip(axes, z_scales):
        sm = softmax(z)
        ax.plot(z, sm)
        ax.set_xlabel("logit"); ax.set_ylabel("softmax output")
        ax.set_title(label)
    plt.suptitle("Figure 1 — Why sqrt(d_k): without scaling, softmax saturates -> near-zero gradients")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** Without scaling (left), large logits push softmax into its flat
    extremes — the output is almost one-hot and the gradient is nearly zero, exactly
    the vanishing-gradient saturation we saw with sigmoid in Notebook 15. Dividing by
    $\sqrt{d_k}$ (right) keeps logits near zero and the softmax in its curved, high-
    gradient region — so attention can be trained effectively regardless of $d_k$. This
    one design choice is the difference between a trainable and an untrainable system.
    """),

    code(r"""
    # Figure 2 — self-attention heatmap on a toy sentence (token similarities).
    words = ["the", "cat", "sat", "on", "the", "mat"]
    # give each word a simple embedding (random but reproducible)
    embed = rng.normal(size=(len(words), 16))
    # make "the" embeddings similar to each other
    embed[4] = embed[0] + 0.1 * rng.normal(size=16)
    Wq2 = rng.normal(size=(16, 16)); Wk2 = rng.normal(size=(16, 16)); Wv2 = rng.normal(size=(16, 16))
    Q2, K2, V2 = embed @ Wq2, embed @ Wk2, embed @ Wv2
    _, weights2 = scaled_dot_product_attention(Q2, K2, V2)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(weights2, cmap="viridis", vmin=0, vmax=weights2.max())
    ax.set_xticks(range(len(words))); ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words); ax.set_yticklabels(words)
    ax.set_xlabel("keys (attended to)"); ax.set_ylabel("queries (attending from)")
    plt.colorbar(im, ax=ax, label="attention weight")
    ax.set_title("Figure 2 — Self-attention heatmap: each row sums to 1")
    plt.show()
    print("Rows sum to 1 (probability distributions):", np.round(weights2.sum(axis=1), 3))
    """),

    md(r"""
    **Figure 2.** Each **row** is a query token's probability distribution over all
    key tokens — where it "looks." Each **column** is how much each token is attended
    to. The pattern encodes relationships: tokens that share semantics or syntactic
    role attract higher weights. In a trained Transformer these patterns are
    task-driven (e.g., a pronoun attends strongly to its referent, a verb attends to
    its subject). Visualizing attention weights is a standard interpretability tool —
    though note they explain the model's *routing*, not the world's meaning (see §7).
    """),

    code(r"""
    # Figure 3 — causal mask: show the upper-triangle zeroed out.
    _, causal_w = scaled_dot_product_attention(Q2, K2, V2, mask=causal_mask(len(words)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (w, title) in zip(axes, [(weights2, "Bi-directional (encoder)"),
                                      (causal_w, "Causal / masked (decoder)")]):
        im = ax.imshow(w, cmap="viridis", vmin=0, vmax=w.max())
        ax.set_xticks(range(len(words))); ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=30); ax.set_yticklabels(words)
        ax.set_title(title); plt.colorbar(im, ax=ax)
    plt.suptitle("Figure 3 — Causal masking zeros upper-triangle (future tokens) for generation")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** The **encoder** (left) allows every token to attend to every other —
    full bidirectional context, as in BERT. The **decoder** (right) applies a causal
    mask: token $t$ can only attend to positions $\le t$ (the lower triangle), so
    auto-regressive generation never "cheats" by peeking at future tokens. The mask is
    added *before* softmax so masked entries become exactly 0 weight. This is the
    structural difference between GPT-style (decoder-only, causal) and BERT-style
    (encoder-only, bidirectional) models.
    """),

    code(r"""
    # Figure 4 — multi-head: each head specialises on a different part of the sequence.
    _, hw = multi_head_attention(embed, n_heads=4, d_model=16, seed=7)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(hw[i], cmap="viridis")
        ax.set_xticks(range(len(words))); ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, fontsize=7, rotation=30)
        ax.set_yticklabels(words, fontsize=7)
        ax.set_title(f"head {i+1}")
    plt.suptitle("Figure 4 — Multi-head: each head attends to different relationship patterns")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** With random (untrained) weights the four heads are already different;
    after training each head specialises on a distinct type of relationship — positional
    proximity, syntactic agreement, coreference, semantic similarity, etc. This is why
    multi-head attention is more expressive than a single head: it captures multiple
    aspects of context in parallel and then mixes them via the output projection $W^O$.
    The number of heads $H$ and the head dimension $d_k = d_{model}/H$ are
    hyperparameters; practical models use $H\in\{8,16,32\}$.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Softmax saturation** | Gradients vanish, attention collapses to one-hot | Large dot-products without $\sqrt{d_k}$ | Always divide by $\sqrt{d_k}$ |
    | **O(T²) memory OOM** | Out-of-memory on long contexts | Full T×T attention matrix | Flash Attention, sparse/local attention, sliding window |
    | **Attention is not causal** | Model peeks at future; inflated eval | Missing upper-triangular mask in decoder | Verify causal mask applied before softmax |
    | **Attention ≠ explanation** | Wrong reasoning from weight maps | High attention weight ≠ causal importance | Use gradient-based attribution (Ntbk 13) |
    | **Head collapse** | All heads learn the same pattern | Insufficient diversity signal | Layer-wise LR, entropy regularisation |
    | **Positional unawareness** | Permutation-invariant model on order-sensitive task | Attention has no notion of position | Add positional encoding (Ntbk 19) |
    | **Cross-attention mismatch** | Decoder attends to garbage | Wrong K/V source (e.g., wrong layer) | Verify encoder-decoder wiring |

    The cell shows **positional blindness** — self-attention treats permuted sequences
    identically without positional encodings.
    """),

    code(r"""
    # Self-attention is permutation-invariant WITHOUT positional encoding.
    X_perm = X[[3, 0, 5, 1, 4, 2], :]                     # shuffle rows
    Q_p = X_perm @ Wq; K_p = X_perm @ Wk; V_p = X_perm @ Wv
    out_orig, _ = scaled_dot_product_attention(Q, K, V)
    out_perm, _ = scaled_dot_product_attention(Q_p, K_p, V_p)
    perm_idx = [3, 0, 5, 1, 4, 2]
    equal = np.allclose(out_orig[perm_idx], out_perm)
    print(f"attention(permuted input) == permuted attention(original): {equal}")
    print("Without positional encoding, self-attention is ORDER-AGNOSTIC.")
    print("'the cat sat' and 'sat cat the' produce the same token embeddings (just shuffled).")
    print("This is why Transformers (Ntbk 19) add sinusoidal or RoPE positional encodings.")
    """),

    md(r"""
    ## 8 · Production Library Implementation

    In production, attention lives inside `torch.nn.MultiheadAttention` (PyTorch) or
    the HuggingFace Transformers library. The critical production-scale addition is
    **Flash Attention** — an IO-aware exact attention algorithm that tiles the $T\times T$
    computation to fit in SRAM, reducing memory from $O(T^2)$ to $O(T)$ without
    approximation. The import is guarded; if PyTorch is absent the scratch
    implementation above already demonstrates the full mechanism.
    """),

    code(r"""
    # Multi-head attention in PyTorch (guarded import).
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        d = 32; T = 8; H = 4
        mha = nn.MultiheadAttention(embed_dim=d, num_heads=H, batch_first=True)
        Xt = torch.randn(1, T, d)                          # (batch, T, d)
        with torch.no_grad():
            out_t, w_t = mha(Xt, Xt, Xt)                  # self-attention
        print(f"torch MHA output: {out_t.shape}")
        print(f"torch MHA weight: {w_t.shape}  (averaged across heads by default)")
        print("Same scaled-dot-product-attention formula, but fused CUDA kernel + Flash Attention option.")
    except Exception as e:
        print(f"[torch not available: {type(e).__name__}] "
              f"the scratch NumPy implementation above is the full mechanism.")
    """),

    md(r"""
    **Scratch vs production.** Our 20-line NumPy implementation computes the exact same
    formula as `nn.MultiheadAttention`; what the framework adds is a fused CUDA kernel,
    Flash Attention (IO-aware tiling for long contexts), `key_padding_mask` for
    variable-length batches, and `attn_mask` for causal generation — all wrapping the
    same mathematical object. The senior skill is understanding *what it computes* so
    you can debug attention patterns, tune head counts, and reason about when O(T²)
    cost becomes the binding constraint.
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Document Q&A (Cross-Attention over Retrieved Docs)

    **Scenario.** A legal-document assistant answers user questions by retrieving
    relevant clauses and generating a grounded answer. The generation step uses
    **cross-attention** from the decoder (query = user question embedding) to the
    encoder outputs (key/value = retrieved clause embeddings), letting the model focus
    on the most relevant legal language.

    **Why attention is the right tool:**
    - The relevant clause can be *anywhere* in the retrieved set — the model must
      attend selectively, not just read a fixed window.
    - Cross-attention gives a **soft, differentiable "search"** over retrieved content —
      the intuition behind RAG (Phase 5).
    - Attention weights provide a *partial* audit trail: which clauses the model focused
      on (caveat §7: weight ≠ causal importance).

    **Business objectives:** accurate, grounded answers with traceable sources; low
    hallucination (Phase 4, Notebook 24).

    **Cost of mistakes:** wrong legal interpretation → liability; slow response →
    user trust; source mis-attribution → compliance risk.

    **Constraints:** context length limits (O(T²) — must chunk documents, Notebook 29);
    latency (Flash Attention, caching); and the need to expose source citations.

    **KPIs:** answer faithfulness (grounded in retrieved text), retrieval precision/
    recall (Notebook 37), latency, hallucination rate (Notebook 24), and user-
    satisfaction surveys.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Flash Attention** is the standard for long contexts: $O(T)$ memory (vs $O(T^2)$),
      ~2–4× faster on GPU, exact (no approximation). Use `scaled_dot_product_attention`
      in PyTorch ≥2.0 which dispatches to Flash Attention automatically when possible.
    - **KV caching** for inference: in autoregressive generation, K and V of all past
      tokens are cached; only the new token's Q is computed per step. Without this,
      inference is $O(T^2)$ per token instead of $O(T)$.
    - **Context length and chunking** (Notebook 29): even with Flash Attention, very
      long documents must be chunked because the model was trained on a finite context
      length; splitting strategy is a key production decision.
    - **Positional encoding** is required (attention is permutation-invariant); modern
      models use RoPE or ALiBi (Notebook 19) for length-extrapolation.
    - **Batch padding masks**: in batched inference, sequences are padded to the same
      length; mask out padding positions to prevent them from influencing attention.
    - **Attention monitoring**: track attention entropy (all-flat = underspecified,
      all-peaked = over-attended), and use gradient-based attribution (Notebook 13) for
      explanation — not raw attention weights.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Attention complexity:**

    | Method | Time | Memory | Quality | Use case |
    |---|---|---|---|---|
    | Full (dense) attention | $O(T^2 d)$ | $O(T^2)$ | Exact | Short/medium contexts |
    | Flash Attention | $O(T^2 d)$ compute, $O(T)$ HBM | $O(T)$ | Exact | **Default for training** |
    | Sparse / sliding window | $O(T \cdot w \cdot d)$ | $O(Tw)$ | Approx | Very long documents |
    | Linear attention / Mamba | $O(Td)$ | $O(d)$ | Approx | Streaming, ultra-long |

    **Number of heads vs head dimension:**

    | Config | Pros | Cons |
    |---|---|---|
    | Many heads, small $d_k$ | More diverse relationship types | Each head has less capacity |
    | Fewer heads, large $d_k$ | Rich per-head representations | Less diversity |
    | GQA (grouped-query) | Fewer KV heads → smaller cache | Slightly less expressive |

    **Self-attention vs cross-attention:**

    | | Self-attention | Cross-attention |
    |---|---|---|
    | Q/K/V source | Same sequence | Q from one, K/V from another |
    | Use in encoder | **Yes** (full bidirectional) | No |
    | Use in decoder | **Yes** (causal, autoregressive) | **Yes** (attend to encoder) |
    | Use in RAG | — | **Yes** (query over retrieved docs) |

    **Senior lesson:** attention is a parallel, direct-access operation — it solves the
    RNN bottleneck but introduces $O(T^2)$ cost that Flash Attention tames in practice.
    Understanding the Q/K/V mechanics, causal masking, and multi-head design is
    mandatory for working with any modern LLM.
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Walk me through scaled dot-product attention.* → $\text{softmax}(QK^\top/\sqrt{d_k})V$;
      Q queries, K advertises, V contributes; softmax gives attention distribution
      (§4.1, §5.1).
    - *Why divide by $\sqrt{d_k}$?* → dot-product variance grows with $d_k$; without
      scaling, softmax saturates and gradients vanish (§4.2, Fig 1, §5.2).

    **Deep-dive questions**
    - *Self-attention vs cross-attention?* → Same formula; source of Q vs K/V differs
      (§4.5).
    - *Multi-head attention — why and how?* → $H$ parallel heads with independent
      projections capture diverse relationship types; concatenate, project (§4.4).
    - *O(T²) cost and how to address it?* → Flash Attention (exact, IO-aware); sparse/
      linear attention (approx); KV caching for inference (§10, §11).

    **Whiteboard questions**
    - "Implement scaled dot-product attention in NumPy." (Section 5.1.)
    - "Add a causal mask to the attention." (Section 5.3.)

    **Strong vs weak answers**
    - *"Why did Transformers replace RNNs?"*
      - **Weak:** "Transformers are more powerful."
      - **Strong:** "Self-attention gives $O(1)$ information paths between any two
        positions (vs $O(T)$ for RNNs) and is fully parallel — both removing the
        sequential bottleneck and fixed-size memory limits of Notebook 17. The cost is
        $O(T^2)$, addressed by Flash Attention for training and KV caching for
        inference."
    - *"Can we use attention weights to explain a model decision?"*
      - **Weak:** "Yes, high weights mean the model focused there."
      - **Strong:** "Not reliably — high attention weight correlates with but doesn't
        equal causal importance; use gradient-based attribution (Notebook 13) for
        explainability. Attention weights are a useful *debugging* signal, not a
        trustworthy explanation."

    **Follow-ups:** "GQA/MQA?" (fewer KV heads → smaller cache). "RoPE vs sinusoidal?"
    (rotation-based position encoding, length extrapolation). "Flash Attention — what
    makes it fast?" (IO-aware tiling to SRAM, avoids writing full T² matrix to HBM).

    **Common mistakes:** forgetting $\sqrt{d_k}$; missing causal mask in decoder;
    treating attention as explanation; ignoring positional encoding's necessity; not
    knowing O(T²) cost; confusing self- and cross-attention.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define attention in one sentence using query/key/value.
    2. **Why was it invented?** What two RNN limits (Notebook 17) does it fix?
    3. **How does it work?** Walk the formula: scores → scale → mask → softmax → mix.
    4. **Why does it work?** Why does $\sqrt{d_k}$ scaling matter?
    5. **When to use it?** Self vs cross-attention and when each appears.
    6. **When NOT to use it (or be careful)?** O(T²) scaling and the explanation caveat.
    7. **Tradeoffs?** Multi-head (diversity vs capacity); Flash Attention; KV caching.
    8. **How would you productionize it?** Flash Attention, KV cache, context chunking,
       positional encoding, padding masks.
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Verify by hand (3 tokens, $d_k=2$) that the attention weights sum to 1 and the
       output is a convex combination of the value rows.
    2. Explain in two sentences why removing the $\sqrt{d_k}$ factor hurts training.

    **Beginner → Intermediate (coding)**
    3. Extend `scaled_dot_product_attention` to handle a **batch dimension** (inputs of
       shape `(batch, T, d)`) and verify the output shapes.
    4. Implement **additive (Bahdanau) attention** — $e_{ij}=v^\top\tanh(W_q q_i+W_k k_j)$ —
       and compare its attention patterns to scaled dot-product.

    **Intermediate (analysis)**
    5. Show empirically that multi-head attention with $H=4$ heads learns qualitatively
       different patterns than a single full-$d_{model}$ head on the same toy sequence.
    6. Profile the wall-clock cost of the scratch `scaled_dot_product_attention` as
       $T$ grows from 64 to 2048 and fit an $O(T^2)$ curve to the measurements.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive the gradient of the attention output w.r.t. $Q$ using the
       chain rule through softmax — show why $\sqrt{d_k}$ is needed for stable gradients.
    8. *Design:* the document Q&A system of §9 — chunking strategy (Notebook 29),
       cross-attention wiring, KV caching for multi-turn, context-length budget, and
       how to expose clause-level citations.
    9. *Debug:* a generation model produces the same token every step after position 5.
       Diagnose (missing causal mask? collapsed attention to one token?) and propose
       the diagnostic steps.
    """),

    md(r"""
    ---
    ### Summary
    Attention gives every position **direct, parallel access** to every other position
    via a soft weighted sum — $\text{softmax}(QK^\top/\sqrt{d_k})V$ — fixing both the
    sequential bottleneck and the fixed-size memory limit of RNNs (Notebook 17).
    Dividing by $\sqrt{d_k}$ prevents softmax saturation (the Notebook-15 vanishing-
    gradient pattern, reappearing in a new form). **Causal masking** zeros future
    positions for autoregressive generation. **Multi-head attention** learns diverse
    relationship types in parallel. The cost is $O(T^2)$, addressed in production by
    Flash Attention (training) and KV caching (inference). Self-attention is
    **permutation-invariant**, so positional encoding is required (Notebook 19).

    **Next:** `19 · Transformers` — attention + positional encoding + layer norm +
    feed-forward residual blocks, stacked into the architecture that runs every modern
    LLM. We'll build a miniature GPT-style Transformer from scratch.
    """),
]

build("phase3_deep_learning/18_attention_mechanism.ipynb", cells)
