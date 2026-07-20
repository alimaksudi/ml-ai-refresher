"""Build DL-08: assemble and train an inspectable causal Transformer."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # DL-08 · Transformers

    **Prerequisites:** FND-01, FND-02, FND-04, DL-01 through DL-04, DL-06, DL-07, and NLP-01  
    **Estimated mastery time:** 12–16 hours, including the checkpoint  
    **Required next checkpoint:** `projects/tiny_language_model/MASTERY_CHECKPOINT.md`

    Attention retrieves information across positions. It is not yet a complete model.
    A Transformer adds position information, residual paths, normalization,
    position-wise feed-forward networks, repeated blocks, and a task-specific output
    head.

    We will assemble those pieces twice: first as transparent NumPy arithmetic, then as
    a small GPT-style decoder written directly in this notebook. We will expose the
    tokenizer, shifted targets, mask, loss, optimization, checkpoint selection,
    generation loop, and KV cache. No hosted model or API key is needed.
    """),

    md(r"""
    ## 1 · Mastery outcomes

    You will be able to:

    - trace token IDs through embeddings, positions, blocks, logits, and loss;
    - explain what attention mixing and the feed-forward network each contribute;
    - calculate sinusoidal positions and LayerNorm manually;
    - distinguish learned absolute, sinusoidal, relative-bias, and rotary positions;
    - write a pre-normalized residual Transformer block;
    - align input tokens with next-token targets without an off-by-one error;
    - prove that a causal decoder cannot use future tokens;
    - overfit one batch as a pipeline diagnostic;
    - train with a development split and restore the best validation checkpoint;
    - implement greedy, temperature, and top-k decoding;
    - explain why teacher-forced training is parallel across positions while generation is sequential;
    - implement a per-layer KV cache and verify cached logits against full recomputation;
    - compare encoder-only, decoder-only, and encoder–decoder families;
    - state when a simpler model is the better engineering choice.

    ```mermaid
    flowchart LR
        A[Text] --> B[Token IDs]
        B --> C[Token + position vectors]
        C --> D[Pre-norm attention]
        D --> E[Residual add]
        E --> F[Pre-norm FFN]
        F --> G[Residual add]
        G --> H[Repeat blocks]
        H --> I[Final norm]
        I --> J[LM logits]
        J --> K[Shifted cross-entropy]
    ```
    """),

    md(r"""
    ## 2 · Why every part exists

    | Part | Problem it solves | What it does not solve alone |
    |---|---|---|
    | token embedding | turns discrete IDs into vectors | order or context |
    | position representation | distinguishes locations and offsets | content retrieval |
    | masked self-attention | mixes legal earlier context into each position | per-position nonlinear transformation |
    | feed-forward network | transforms each position independently | communication across positions |
    | residual connection | preserves a direct identity and gradient path | activation-scale conditioning |
    | LayerNorm | normalizes features within each token | data leakage or exploding optimizer steps |
    | LM head | maps hidden vectors to vocabulary logits | probability calibration or factuality |

    Attention is permutation-**equivariant** without positions: permuting input rows
    permutes output rows. Adding positions breaks that symmetry in a controlled way.

    A pre-norm block calculates:

    $$
    r=x+\operatorname{MHA}(\operatorname{LN}(x))
    $$

    $$
    y=r+\operatorname{FFN}(\operatorname{LN}(r))
    $$

    The residual stream stays at width $d_{model}$ so additions are shape-compatible.
    """),

    code(r"""
    import copy
    import math
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch import nn

    np.set_printoptions(precision=5, suppress=True)


    def set_reproducible(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


    set_reproducible(31)
    DEVICE = torch.device("cpu")
    print("device:", DEVICE)
    """),

    md(r"""
    ## 3 · Position and normalization by hand

    ### Sinusoidal position

    For position $p$, model width $d$, and pair index $i$:

    $$
    PE(p,2i)=\sin\left(p/10000^{2i/d}\right)
    $$

    $$
    PE(p,2i+1)=\cos\left(p/10000^{2i/d}\right)
    $$

    At $p=0$, every sine coordinate is 0 and every cosine coordinate is 1. At $d=4$,
    position 1 is approximately:

    $$
    [\sin(1),\cos(1),\sin(0.01),\cos(0.01)]
    \approx[0.8415,0.5403,0.0100,1.0000]
    $$

    These vectors can be generated beyond training length, but model quality beyond
    the trained lengths is not guaranteed.

    ### Layer normalization

    For one token vector $x=[1,2,3]$:

    $$
    \mu=2,\qquad \sigma^2=\frac{(1-2)^2+(2-2)^2+(3-2)^2}{3}=\frac23
    $$

    $$
    \widehat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
    \approx[-1.225,0,1.225]
    $$

    A learned LayerNorm then applies feature-wise scale $\gamma$ and shift $\beta$.
    Unlike dataset standardization, these statistics are calculated inside each token
    vector, not fitted from training rows.
    """),

    code(r"""
    def sinusoidal_positions(length, model_width):
        if model_width % 2 != 0:
            raise ValueError("This teaching implementation uses an even model width.")
        positions = np.arange(length)[:, None]
        pair_indices = np.arange(0, model_width, 2)[None, :]
        angles = positions / np.power(10000.0, pair_indices / model_width)
        encoding = np.empty((length, model_width))
        encoding[:, 0::2] = np.sin(angles)
        encoding[:, 1::2] = np.cos(angles)
        return encoding


    def layer_norm_numpy(values, epsilon=1e-5):
        mean = values.mean(axis=-1, keepdims=True)
        variance = values.var(axis=-1, keepdims=True)
        return (values - mean) / np.sqrt(variance + epsilon)


    positions = sinusoidal_positions(8, 4)
    normalized = layer_norm_numpy(np.array([[1.0, 2.0, 3.0]]))
    print("position 0:", positions[0])
    print("position 1:", positions[1])
    print("normalized [1,2,3]:", normalized)
    print("normalized mean / variance:", normalized.mean(), normalized.var())
    """),

    code(r"""
    position_picture = sinusoidal_positions(50, 32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(position_picture.T, aspect="auto", cmap="RdBu")
    axes[0].set_xlabel("position")
    axes[0].set_ylabel("dimension")
    axes[0].set_title("Sinusoidal position matrix")
    for dimension in (0, 2, 8, 16):
        axes[1].plot(position_picture[:, dimension], label=f"dimension {dimension}")
    axes[1].set_xlabel("position")
    axes[1].set_title("Different frequencies")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 4 · One transparent NumPy block

    The feed-forward network is the same MLP at every position:

    $$
    \operatorname{FFN}(x)=\operatorname{GELU}(xW_1+b_1)W_2+b_2
    $$

    Attention communicates **between** positions. The FFN changes features **within**
    each position. The following block uses one head for readable shapes; DL-07 already
    verified the multi-head operation against PyTorch.
    """),

    code(r"""
    def softmax_numpy(values, axis=-1):
        shifted = values - values.max(axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=axis, keepdims=True)


    def gelu_numpy(values):
        return 0.5 * values * (
            1 + np.tanh(np.sqrt(2 / np.pi) * (values + 0.044715 * values**3))
        )


    def numpy_causal_block(inputs, parameters):
        W_q, W_k, W_v, W_o, W_1, b_1, W_2, b_2 = parameters
        normalized = layer_norm_numpy(inputs)
        queries = normalized @ W_q
        keys = normalized @ W_k
        values = normalized @ W_v
        scores = queries @ keys.T / np.sqrt(queries.shape[-1])
        future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        weights = softmax_numpy(np.where(future, -np.inf, scores), axis=-1)
        after_attention = inputs + (weights @ values) @ W_o
        normalized_again = layer_norm_numpy(after_attention)
        feed_forward = gelu_numpy(normalized_again @ W_1 + b_1) @ W_2 + b_2
        return after_attention + feed_forward, weights


    length, model_width, feed_forward_width = 4, 6, 18
    rng = np.random.default_rng(9)
    scale = 0.1
    numpy_parameters = (
        rng.normal(0, scale, (model_width, model_width)),
        rng.normal(0, scale, (model_width, model_width)),
        rng.normal(0, scale, (model_width, model_width)),
        rng.normal(0, scale, (model_width, model_width)),
        rng.normal(0, scale, (model_width, feed_forward_width)),
        np.zeros(feed_forward_width),
        rng.normal(0, scale, (feed_forward_width, model_width)),
        np.zeros(model_width),
    )
    numpy_inputs = rng.normal(size=(length, model_width))
    numpy_outputs, numpy_attention = numpy_causal_block(numpy_inputs, numpy_parameters)

    print("input / output shapes:", numpy_inputs.shape, numpy_outputs.shape)
    print("attention shape:", numpy_attention.shape)
    print("largest future attention weight:", numpy_attention[np.triu_indices(length, 1)].max())
    assert numpy_outputs.shape == numpy_inputs.shape
    assert np.all(numpy_attention[np.triu_indices(length, 1)] == 0.0)
    """),

    md(r"""
    ## 5 · Tokenization and shifted targets

    A language model predicts the next token at every position:

    ```text
    text:     m  o  d  e  l
    input:    m  o  d  e
    target:   o  d  e  l
    ```

    For logits $Z\in\mathbb{R}^{B\times T\times V}$ and target IDs
    $Y\in\{0,\ldots,V-1\}^{B\times T}$:

    $$
    L=-\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
    \log p(Y_{b,t}\mid X_{b,1:t})
    $$

    The causal mask prevents a position from reading later **inputs**. Target shifting
    tells the loss which later token to predict. They solve different problems.

    We use a character tokenizer so every rule is visible. Production tokenizers use
    subword units and require their own training, normalization, special-token, and
    versioning contracts.
    """),

    code(r"""
    corpus = (
        "a model learns patterns from examples. "
        "attention mixes earlier context. "
        "a transformer predicts the next token. "
        "validation chooses the checkpoint. "
    ) * 35

    vocabulary = sorted(set(corpus))
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    id_to_token = {index: character for character, index in token_to_id.items()}


    def encode(text):
        return [token_to_id[character] for character in text]


    def decode(token_ids):
        return "".join(id_to_token[int(token_id)] for token_id in token_ids)


    all_ids = torch.tensor(encode(corpus), dtype=torch.long)
    split_index = int(0.85 * len(all_ids))
    train_ids = all_ids[:split_index]
    validation_ids = all_ids[split_index:]


    def make_windows(token_stream, block_size):
        inputs, targets = [], []
        for start in range(0, len(token_stream) - block_size - 1, block_size):
            window = token_stream[start:start + block_size + 1]
            inputs.append(window[:-1])
            targets.append(window[1:])
        return torch.stack(inputs), torch.stack(targets)


    block_size = 32
    train_X, train_y = make_windows(train_ids, block_size)
    validation_X, validation_y = make_windows(validation_ids, block_size)
    print("vocabulary size:", len(vocabulary))
    print("training windows:", train_X.shape, train_y.shape)
    print("first input: ", repr(decode(train_X[0].tolist())))
    print("first target:", repr(decode(train_y[0].tolist())))
    assert torch.equal(train_X[0, 1:], train_y[0, :-1])
    """),

    md(r"""
    ## 6 · Build a tiny GPT directly

    Shape contract for $B=16$, $T=32$, $d_{model}=48$, $H=4$, and vocabulary $V$:

    ```text
    token IDs             (16, 32)
    token + position      (16, 32, 48)
    Q/K/V per head        (16, 4, 32, 12)
    block output          (16, 32, 48)
    vocabulary logits     (16, 32, V)
    ```

    Learned absolute positions keep this first trainable model compact. They cannot
    index beyond `block_size`; sinusoidal, relative-bias, RoPE, and ALiBi make different
    tradeoffs, but none guarantees length extrapolation.
    """),

    code(r"""
    class CausalSelfAttention(nn.Module):
        def __init__(self, model_width, number_of_heads):
            super().__init__()
            if model_width % number_of_heads != 0:
                raise ValueError("model width must divide evenly across heads")
            self.number_of_heads = number_of_heads
            self.head_width = model_width // number_of_heads
            self.qkv = nn.Linear(model_width, 3 * model_width)
            self.output = nn.Linear(model_width, model_width)

        def forward(self, hidden, cache=None):
            batch, query_length, model_width = hidden.shape
            q, k_new, v_new = self.qkv(hidden).chunk(3, dim=-1)

            def as_heads(tensor):
                return tensor.view(batch, query_length, self.number_of_heads, self.head_width).transpose(1, 2)

            q, k_new, v_new = as_heads(q), as_heads(k_new), as_heads(v_new)
            if cache is None:
                k, v = k_new, v_new
                past_length = 0
            else:
                past_k, past_v = cache
                past_length = past_k.shape[-2]
                k = torch.cat([past_k, k_new], dim=-2)
                v = torch.cat([past_v, v_new], dim=-2)

            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
            query_positions = past_length + torch.arange(query_length, device=hidden.device)
            key_positions = torch.arange(k.shape[-2], device=hidden.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            mixed = weights @ v
            mixed = mixed.transpose(1, 2).contiguous().view(batch, query_length, model_width)
            return self.output(mixed), (k, v), weights


    class TransformerBlock(nn.Module):
        def __init__(self, model_width, number_of_heads, feed_forward_width):
            super().__init__()
            self.norm_attention = nn.LayerNorm(model_width)
            self.attention = CausalSelfAttention(model_width, number_of_heads)
            self.norm_ffn = nn.LayerNorm(model_width)
            self.ffn = nn.Sequential(
                nn.Linear(model_width, feed_forward_width),
                nn.GELU(),
                nn.Linear(feed_forward_width, model_width),
            )

        def forward(self, hidden, cache=None):
            attention_output, new_cache, weights = self.attention(
                self.norm_attention(hidden), cache
            )
            hidden = hidden + attention_output
            hidden = hidden + self.ffn(self.norm_ffn(hidden))
            return hidden, new_cache, weights


    class TinyGPT(nn.Module):
        def __init__(self, vocabulary_size, block_size, model_width=48, heads=4, layers=2):
            super().__init__()
            self.block_size = block_size
            self.token_embedding = nn.Embedding(vocabulary_size, model_width)
            self.position_embedding = nn.Embedding(block_size, model_width)
            self.blocks = nn.ModuleList(
                [TransformerBlock(model_width, heads, 4 * model_width) for _ in range(layers)]
            )
            self.final_norm = nn.LayerNorm(model_width)
            self.lm_head = nn.Linear(model_width, vocabulary_size, bias=False)
            self.apply(self._initialize_weights)
            self.lm_head.weight = self.token_embedding.weight

        @staticmethod
        def _initialize_weights(module):
            # Small initial logits keep the first loss near the uniform reference
            # log(vocabulary_size), which makes optimization easier to diagnose.
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, token_ids, targets=None, cache=None):
            past_length = 0 if cache is None else cache[0][0].shape[-2]
            positions = torch.arange(
                past_length, past_length + token_ids.shape[1], device=token_ids.device
            )
            if positions[-1] >= self.block_size:
                raise ValueError("sequence exceeds learned position table")
            hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
            new_cache, attention_maps = [], []
            layer_caches = [None] * len(self.blocks) if cache is None else cache
            for block, layer_cache in zip(self.blocks, layer_caches):
                hidden, created_cache, weights = block(hidden, layer_cache)
                new_cache.append(created_cache)
                attention_maps.append(weights)
            logits = self.lm_head(self.final_norm(hidden))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            return logits, loss, new_cache, attention_maps
    """),

    code(r"""
    set_reproducible(31)
    shape_model = TinyGPT(len(vocabulary), block_size)
    shape_logits, shape_loss, _, shape_maps = shape_model(train_X[:3], train_y[:3])
    print("tokens:", train_X[:3].shape)
    print("logits:", shape_logits.shape)
    print("loss:", shape_loss.item())
    print("first attention map:", shape_maps[0].shape, "= (B,H,T,T)")

    future_indices = torch.triu_indices(block_size, block_size, offset=1)
    maximum_future_weight = shape_maps[0][:, :, future_indices[0], future_indices[1]].max().item()
    print("maximum future attention weight:", maximum_future_weight)
    assert maximum_future_weight == 0.0
    """),

    md(r"""
    ## 7 · Diagnose before long training: overfit one batch

    If a model cannot memorize one small batch, do not tune regularization or collect
    more data. Inspect, in order: target shift, causal direction, output shape, finite
    loss, nonzero gradients, optimizer step, and parameter change.
    """),

    code(r"""
    set_reproducible(4)
    diagnostic_model = TinyGPT(len(vocabulary), block_size, model_width=32, heads=4, layers=1)
    diagnostic_optimizer = torch.optim.AdamW(diagnostic_model.parameters(), lr=0.01)
    diagnostic_X, diagnostic_y = train_X[:4], train_y[:4]
    diagnostic_losses = []

    for step in range(80):
        diagnostic_optimizer.zero_grad(set_to_none=True)
        _, loss, _, _ = diagnostic_model(diagnostic_X, diagnostic_y)
        loss.backward()
        nn.utils.clip_grad_norm_(diagnostic_model.parameters(), 1.0)
        diagnostic_optimizer.step()
        diagnostic_losses.append(loss.item())

    print("first diagnostic loss:", diagnostic_losses[0])
    print("final diagnostic loss:", diagnostic_losses[-1])
    assert diagnostic_losses[-1] < 0.15 * diagnostic_losses[0]
    """),

    md(r"""
    ## 8 · Development training and checkpoint selection

    Training rows update parameters. Validation loss selects the epoch. This lesson
    does not claim a final generalization estimate; the tiny-LM checkpoint adds the
    stronger multi-seed and held-out evaluation contract.
    """),

    code(r"""
    def evaluate_loss(model, features, targets, batch_size=32):
        model.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                _, loss, _, _ = model(
                    features[start:start + batch_size], targets[start:start + batch_size]
                )
                losses.append(loss.item())
        return float(np.mean(losses))


    set_reproducible(8)
    model = TinyGPT(len(vocabulary), block_size, model_width=48, heads=4, layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_X, train_y),
        batch_size=24,
        shuffle=True,
        generator=torch.Generator().manual_seed(8),
    )
    best_validation_loss = math.inf
    best_state = None
    history = []

    for epoch in range(1, 13):
        model.train()
        training_losses = []
        for feature_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss, _, _ = model(feature_batch, target_batch)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite Transformer gradient")
            optimizer.step()
            training_losses.append(loss.item())
        validation_loss = evaluate_loss(model, validation_X, validation_y)
        row = {"epoch": epoch, "train_loss": float(np.mean(training_losses)), "validation_loss": validation_loss}
        history.append(row)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    display(pd.DataFrame(history))
    print("best validation loss:", best_validation_loss)
    assert best_validation_loss < history[0]["validation_loss"]
    """),

    code(r"""
    history_table = pd.DataFrame(history)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history_table["epoch"], history_table["train_loss"], marker="o", label="training")
    axis.plot(history_table["epoch"], history_table["validation_loss"], marker="s", label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("cross-entropy")
    axis.set_title("Tiny GPT learning curves")
    axis.legend()
    axis.grid(alpha=0.3)
    plt.show()
    """),

    md(r"""
    ## 9 · Generation is a loop over logits

    At each step, use the last position's logits. Greedy decoding chooses the largest.
    Temperature divides logits before softmax: below 1 sharpens; above 1 flattens.
    Top-k keeps only the $k$ largest candidates. These change sampling behavior, not
    factual knowledge.
    """),

    code(r"""
    def choose_next_token(logits, temperature=0.0, top_k=None, random_generator=None):
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled = logits / temperature
        if top_k is not None:
            threshold = torch.topk(scaled, min(top_k, scaled.shape[-1]), dim=-1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        probabilities = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probabilities, 1, generator=random_generator)


    def generate(model, prompt_ids, new_tokens, temperature=0.0, top_k=None, seed=0):
        generated = prompt_ids.clone()
        random_generator = torch.Generator().manual_seed(seed)
        for _ in range(new_tokens):
            context = generated[:, -model.block_size:]
            logits, _, _, _ = model(context)
            next_token = choose_next_token(
                logits[:, -1, :], temperature, top_k, random_generator
            )
            generated = torch.cat([generated, next_token], dim=1)
        return generated


    prompt = torch.tensor([encode("attention ")], dtype=torch.long)
    greedy = generate(model, prompt, 45)
    sampled = generate(model, prompt, 45, temperature=0.8, top_k=6, seed=22)
    print("greedy:", repr(decode(greedy[0].tolist())))
    print("top-k: ", repr(decode(sampled[0].tolist())))
    print("These are mechanics demonstrations from a tiny repetitive corpus, not useful prose generation.")
    """),

    md(r"""
    ## 10 · KV caching: reuse past projections

    Teacher-forced training knows the full target sequence and evaluates positions in
    parallel behind a causal mask. Generation cannot know the next token before it is
    chosen, so it remains sequential across new tokens.

    During generation, past keys and values do not change. Cache them per layer. For
    layers $L$, batch $B$, heads $H$, cached length $T$, head width $D$, and bytes $s$:

    $$
    M_{KV}=2LBHTDs
    $$

    Caching removes repeated K/V projection work. Each new query still attends over a
    growing history, and cache memory grows linearly with cached length.
    """),

    code(r"""
    # Full causal logits and token-by-token cached logits must agree in evaluation mode.
    model.eval()
    comparison = train_X[:2, :20]
    with torch.no_grad():
        full_logits, _, _, _ = model(comparison)
        cache = None
        incremental_logits = []
        for position in range(comparison.shape[1]):
            step_logits, _, cache, _ = model(
                comparison[:, position:position + 1], cache=cache
            )
            incremental_logits.append(step_logits)
        incremental_logits = torch.cat(incremental_logits, dim=1)

    maximum_cache_difference = (full_logits - incremental_logits).abs().max().item()
    first_layer_keys, first_layer_values = cache[0]
    print("cached K/V shapes:", first_layer_keys.shape, first_layer_values.shape)
    print("maximum full-versus-cached logit difference:", maximum_cache_difference)
    assert torch.allclose(full_logits, incremental_logits, atol=1e-5, rtol=1e-5)
    """),

    md(r"""
    ## 11 · Transformer families

    | Family | Information flow | Typical objective | Strong fit |
    |---|---|---|---|
    | encoder-only | bidirectional self-attention | masked/replaced-token representation learning | classification, extraction, embeddings |
    | decoder-only | causal self-attention | next-token prediction | generation and continuation |
    | encoder–decoder | bidirectional source; causal target; cross-attention | source-to-target prediction | translation, summarization, structured transformation |

    These are patterns, not guarantees. Later NLP lessons compare trained objectives
    behaviorally. “BERT understands” and “GPT generates” are shortcuts that hide model,
    data, objective, and evaluation differences.

    ### Position choices

    | Method | Core idea | Important limitation |
    |---|---|---|
    | sinusoidal | fixed multi-frequency vector added to tokens | extrapolation still unproven |
    | learned absolute | train one vector per position | fixed learned table |
    | relative bias | add learned distance bias to scores | chosen distance scheme matters |
    | RoPE | rotate Q/K features by position | long-context behavior needs training/evaluation |
    | ALiBi | add distance-dependent linear score bias | task and scale dependent |
    """),

    md(r"""
    ## 12 · When not to use a Transformer

    | Alternative | Prefer it when |
    |---|---|
    | linear/logistic model | sparse or structured baseline already meets the target |
    | boosted trees | tabular data and modest sample size dominate |
    | CNN | local spatial or temporal structure is central |
    | GRU/LSTM | compact streaming state and low per-step memory matter |
    | retrieval without generation | users need exact documents, not synthesized prose |

    Dense attention adds quadratic pair work. A Transformer also needs enough data,
    regularization, evaluation, and serving budget. Architecture popularity is not a
    substitute for a measured baseline.
    """),

    md(r"""
    ## 13 · Failure modes

    | Symptom | Likely cause | First check | Response |
    |---|---|---|---|
    | loss stays near $\log V$ | shift/mask/optimizer bug | one-batch overfit | trace contract in order |
    | validation suspiciously low | overlapping windows or corpus leakage | source boundaries | split before windows |
    | future changes alter earlier logits | causal mask wrong | perturb-future test | fix and assert behavior |
    | cached logits differ | wrong position offset or cache order | layerwise max difference | trace `(B,H,T,D)` and offsets |
    | generation repeats | tiny/biased data, greedy loop, overconfidence | probabilities and corpus | improve data/model; evaluate decoding |
    | learned positions crash | context exceeds table | requested position index | crop, reject, or retrain design |
    | long context exhausts memory | dense attention and KV cache | $BHT^2$, KV formula | reduce length/batch; efficient kernels/design |
    | plausible false answer | next-token objective is not truth verification | grounded evaluation | retrieval, citations, abstention, guardrails |

    LayerNorm and residuals support optimization; they do not guarantee stable training.
    Learning rate, initialization, precision, data, depth, and optimizer still matter.
    """),

    md(r"""
    ## 14 · Production bridge without hiding the foundation

    Production systems normally load a versioned pretrained model and its exact
    tokenizer rather than train from scratch. That workflow adds concerns not shown by
    a three-line download:

    - license and permitted use;
    - tokenizer/model revision compatibility;
    - context and chat-template contracts;
    - quantization quality regression;
    - batching, cache allocation, and latency;
    - prompt-injection and unsafe-output controls;
    - evaluation before and after adaptation;
    - monitoring and rollback.

    This lesson deliberately avoids an internet-dependent checkpoint. The local
    `projects/tiny_language_model` checkpoint is the next gate: it expands the
    experiment with artifact saving, behavioral tests, tokenizer comparisons,
    generation evaluation, and measured KV-cache timing.
    """),

    md(r"""
    ## 15 · Check your understanding

    1. Why do attention and the FFN perform different jobs?
    2. What statistics does LayerNorm calculate, and over which axis here?
    3. Why must residual input and sublayer output have the same width?
    4. What is the difference between a causal mask and shifted targets?
    5. Trace `(B,T)` token IDs to `(B,T,V)` logits.
    6. Why can training process target positions in parallel while generation cannot?
    7. What does a one-batch overfit test isolate?
    8. Why restore the best validation checkpoint rather than the last epoch?
    9. Which tensors enter a KV cache, and why not cache old queries?
    10. Why does a lower LM loss not establish truthfulness or usefulness?
    """),

    md(r"""
    ## 16 · Practice and mini-project

    **Beginner**

    1. Calculate the sinusoidal vector for position 2 at width 4.
    2. Shift the string `model` into input and target rows by hand.

    **Intermediate**

    3. Add dropout to attention weights, projection output, and FFN. Explain why it
       must be disabled for cache-equivalence testing.
    4. Add a perturb-future behavioral test at the final-logit level.
    5. Implement top-p sampling and verify that retained probability mass reaches the threshold.

    **Challenge**

    6. Compare one- and two-layer models across three seeds under the same development
       split. Report loss, variation, parameter count, training time, and generation
       diagnostics. Choose no configuration using final test data.

    **Mini-project · Tiny domain language model**

    Use `projects/tiny_language_model`. Complete its tokenizer, data-window, baseline,
    one-batch overfit, multi-seed training, artifact, cache, and generation checks.
    Your report must explain why this tiny model is an architectural microscope—not a
    useful general LLM—and name one falsifiable next experiment.
    """),

    md(r"""
    ## 17 · Summary and memory aid

    A Transformer block alternates content mixing and position-wise transformation:

    $$
    x\rightarrow x+\operatorname{Attention}(\operatorname{LN}(x))
    \rightarrow r+\operatorname{FFN}(\operatorname{LN}(r))
    $$

    Token and position representations enter repeated blocks; a final head produces
    vocabulary logits. Causal masking limits information, shifted targets define the
    prediction, cross-entropy trains probabilities, and validation selects a frozen
    checkpoint. Training is parallel across known sequence positions, while generation
    is an autoregressive loop. KV caching reuses past projections without removing the
    growing attention or memory cost.

    **Memory aid:** *Position the tokens, mix across time, transform each position,
    preserve the residual stream, then predict the next ID.*

    Section 04 is complete only after the tiny-language-model mastery checkpoint passes.
    """),
]


build("04_deep_learning/08_transformers.ipynb", cells)
