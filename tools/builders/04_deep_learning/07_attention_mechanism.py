"""Build DL-07: attention from weighted retrieval to verified multi-head code."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # DL-07 · The Attention Mechanism

    **Prerequisites:** FND-01, DL-01, DL-02, DL-03, and DL-06  
    **Estimated mastery time:** 9–12 hours, including practice  
    **Next lesson:** DL-08 · Transformers

    An RNN carries one compressed note through a sequence. Attention offers a different
    operation: ask a question, score every available item, and blend the useful
    information. A position can therefore access another position through one direct
    weighted retrieval instead of waiting for information to travel through many
    recurrent steps.

    We will not start by memorizing $Q$, $K$, and $V$. We will start with a weighted
    average, calculate one attention row by hand, introduce learned projections only
    when their purpose is clear, and verify the NumPy implementation against PyTorch.

    ### Scope

    This lesson teaches the attention operation. Positional representations, residual
    blocks, normalization, feed-forward sublayers, training objectives, and a complete
    causal Transformer belong to DL-08.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - explain attention as content-based weighted retrieval;
    - calculate scores, scaling, softmax weights, and an output manually;
    - distinguish queries, keys, values, attention weights, and outputs;
    - trace batch, head, query-length, key-length, and feature dimensions;
    - explain why dot products are scaled by $\sqrt{d_k}$;
    - implement stable batched scaled dot-product attention in NumPy;
    - distinguish causal masks from key-padding masks and combine them safely;
    - prove with a behavioral test that a causal output ignores changed future values;
    - implement batched multi-head attention and match PyTorch numerically;
    - distinguish self-attention, cross-attention, and retrieval augmentation;
    - explain permutation equivariance without calling it invariance;
    - calculate quadratic score-matrix growth;
    - explain what Flash Attention and KV caching improve—and what they do not;
    - use attention maps as diagnostics without treating them as causal explanations.

    ```mermaid
    flowchart LR
        A[Weighted average] --> B[Content scores]
        B --> C[Softmax weights]
        C --> D[Query, key, value roles]
        D --> E[Scaling]
        E --> F[Causal and padding masks]
        F --> G[Multiple heads]
        G --> H[Verified PyTorch match]
        H --> I[Position and cost limits]
        I --> J[Transformer block]
    ```
    """),

    md(r"""
    ## 2 · Begin with the problem, not the letters

    Imagine three sensor records. You need a summary for the current event, but not
    every record is equally relevant.

    | Record | Value vector | Relevance weight |
    |---|---:|---:|
    | A | $[10,0]$ | $0.6$ |
    | B | $[0,6]$ | $0.3$ |
    | C | $[2,2]$ | $0.1$ |

    The retrieved summary is a weighted average:

    $$
    0.6[10,0]+0.3[0,6]+0.1[2,2]=[6.2,2.0]
    $$

    The weights are non-negative and sum to 1. The missing piece is how to calculate
    them from the current need. Attention learns representations that make useful
    query–key matches score highly, then softmax converts scores into weights.

    ### The library analogy

    - **Query:** what the reader is looking for.
    - **Key:** what each book advertises on its label.
    - **Value:** the information retrieved from that book.

    A key is not necessarily the information returned. It is the address used for
    matching; the value is the payload.
    """),

    code(r"""
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    np.set_printoptions(precision=5, suppress=True)
    generator = np.random.default_rng(17)


    def stable_softmax(values, axis=-1):
        values = np.asarray(values, dtype=float)
        shifted = values - np.max(values, axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=axis, keepdims=True)


    sensor_values = np.array([[10.0, 0.0], [0.0, 6.0], [2.0, 2.0]])
    chosen_weights = np.array([0.6, 0.3, 0.1])
    weighted_summary = chosen_weights @ sensor_values

    print("weights sum:", chosen_weights.sum())
    print("weighted summary:", weighted_summary)
    assert np.allclose(weighted_summary, [6.2, 2.0])
    """),

    md(r"""
    ## 3 · One attention calculation by hand

    Use one query, three keys, and three values:

    $$
    q=[1,0]
    $$

    $$
    K=
    \begin{bmatrix}
    2&0\\
    1&0\\
    0&1
    \end{bmatrix},\qquad
    V=
    \begin{bmatrix}
    10&0\\
    0&6\\
    2&2
    \end{bmatrix}
    $$

    The key dimension is $d_k=2$. First calculate dot-product scores:

    $$
    qK^\top=[2,1,0]
    $$

    Scale them:

    $$
    \frac{qK^\top}{\sqrt{d_k}}
    =\frac{[2,1,0]}{\sqrt 2}
    \approx[1.414,0.707,0]
    $$

    Apply softmax:

    $$
    \alpha\approx[0.576,0.284,0.140]
    $$

    Retrieve the weighted value:

    $$
    \alpha V
    \approx0.576[10,0]+0.284[0,6]+0.140[2,2]
    \approx[6.04,1.98]
    $$

    Softmax does not “pick” one record. It produces a differentiable blend. It may
    become sharp, but the operation remains a weighted sum.
    """),

    code(r"""
    manual_query = np.array([[1.0, 0.0]])
    manual_keys = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    manual_values = sensor_values.copy()

    manual_scores = manual_query @ manual_keys.T
    manual_scaled_scores = manual_scores / np.sqrt(manual_query.shape[-1])
    manual_weights = stable_softmax(manual_scaled_scores, axis=-1)
    manual_output = manual_weights @ manual_values

    print("dot-product scores:", manual_scores)
    print("scaled scores:     ", manual_scaled_scores)
    print("attention weights: ", manual_weights)
    print("weights sum:       ", manual_weights.sum(axis=-1))
    print("retrieved output:  ", manual_output)
    """),

    md(r"""
    ## 4 · The complete scaled dot-product operation

    For batched attention:

    $$
    Q\in\mathbb{R}^{B\times T_q\times d_k},\quad
    K\in\mathbb{R}^{B\times T_k\times d_k},\quad
    V\in\mathbb{R}^{B\times T_k\times d_v}
    $$

    $$
    S=\frac{QK^\top}{\sqrt{d_k}}
    $$

    $$
    A=\operatorname{softmax}(S+M)
    $$

    $$
    O=AV
    $$

    | Symbol | Meaning | Shape |
    |---|---|---:|
    | $B$ | batch size | scalar |
    | $T_q$ | number of query positions | scalar |
    | $T_k$ | number of key/value positions | scalar |
    | $d_k$ | query and key feature width | scalar |
    | $d_v$ | value feature width | scalar |
    | $S$ | compatibility scores | $(B,T_q,T_k)$ |
    | $M$ | broadcastable mask bias | compatible with $(B,T_q,T_k)$ |
    | $A$ | attention weights | $(B,T_q,T_k)$ |
    | $O$ | retrieved outputs | $(B,T_q,d_v)$ |

    Softmax runs across the **key axis**, the last axis. Every query row distributes
    its weight over available keys.

    ### Where $Q$, $K$, and $V$ come from

    In self-attention, learned projections start from the same input $X$:

    $$
    Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
    $$

    “Same source” does not mean $Q=K=V$. The projection matrices usually differ, so
    matching and retrieved content can learn different representations.
    """),

    code(r"""
    def scaled_dot_product_attention(queries, keys, values, allowed=None):
        # Batched attention; True entries in allowed may receive probability.
        if queries.shape[-1] != keys.shape[-1]:
            raise ValueError("Queries and keys must have the same feature width.")
        if keys.shape[-2] != values.shape[-2]:
            raise ValueError("Keys and values must have the same sequence length.")

        scores = queries @ np.swapaxes(keys, -2, -1)
        scores = scores / np.sqrt(queries.shape[-1])

        if allowed is not None:
            allowed = np.asarray(allowed, dtype=bool)
            allowed = np.broadcast_to(allowed, scores.shape)
            if np.any(~allowed.any(axis=-1)):
                raise ValueError("Every query must be allowed to attend to at least one key.")
            scores = np.where(allowed, scores, -np.inf)

        weights = stable_softmax(scores, axis=-1)
        outputs = weights @ values
        return outputs, weights, scores


    batched_queries = manual_query[None, :, :]
    batched_keys = manual_keys[None, :, :]
    batched_values = manual_values[None, :, :]
    batched_output, batched_weights, _ = scaled_dot_product_attention(
        batched_queries, batched_keys, batched_values
    )

    print("Q shape:", batched_queries.shape)
    print("K shape:", batched_keys.shape)
    print("V shape:", batched_values.shape)
    print("A shape:", batched_weights.shape)
    print("O shape:", batched_output.shape)
    assert np.allclose(batched_output[0], manual_output)
    assert np.allclose(batched_weights.sum(axis=-1), 1.0)
    """),

    md(r"""
    ## 5 · Why divide by $\sqrt{d_k}$?

    Suppose query and key components are independent, with mean 0 and variance 1.
    Their dot product is a sum of $d_k$ products:

    $$
    q\cdot k=\sum_{j=1}^{d_k}q_jk_j
    $$

    Under those assumptions its variance grows approximately like $d_k$. Dividing by
    $\sqrt{d_k}$ keeps the variance near 1. This is a variance argument, not a claim
    that learned components remain perfectly independent or unit variance.

    Large score differences make softmax very sharp. For the softmax Jacobian:

    $$
    \frac{\partial a_i}{\partial s_j}=a_i(\mathbf{1}_{i=j}-a_j)
    $$

    probabilities near 0 or 1 often produce small local derivatives. Scaling makes
    severe early saturation less likely.
    """),

    code(r"""
    scaling_rows = []
    for key_width in (8, 32, 128, 512):
        queries = generator.normal(size=(2000, key_width))
        keys = generator.normal(size=(2000, key_width))
        raw_scores = np.sum(queries * keys, axis=1)
        scaled_scores = raw_scores / np.sqrt(key_width)
        scaling_rows.append(
            {
                "d_k": key_width,
                "raw variance": raw_scores.var(),
                "scaled variance": scaled_scores.var(),
            }
        )

    scaling_table = pd.DataFrame(scaling_rows)
    display(scaling_table)

    # Use many independent score rows, rather than plotting one softmax distribution
    # over an arbitrary linspace, to measure sharpness honestly.
    score_rows = generator.normal(size=(500, 64)) * np.sqrt(128)
    unscaled_probabilities = stable_softmax(score_rows, axis=1)
    scaled_probabilities = stable_softmax(score_rows / np.sqrt(128), axis=1)

    def mean_entropy(probabilities):
        return float(np.mean(-np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)))

    print("mean maximum probability without scaling:", unscaled_probabilities.max(axis=1).mean())
    print("mean maximum probability with scaling:   ", scaled_probabilities.max(axis=1).mean())
    print("mean entropy without scaling:", mean_entropy(unscaled_probabilities))
    print("mean entropy with scaling:   ", mean_entropy(scaled_probabilities))
    """),

    md(r"""
    ## 6 · Masks answer two different questions

    A mask says which key positions a query may use. Apply it **before softmax**.

    ### Causal mask

    In next-token generation, query position $t$ may use positions $0$ through $t$,
    but not later positions. This prevents future-target leakage during parallel
    training.

    ### Key-padding mask

    Batches pad short sequences. A real query must not retrieve a padded key. Padding
    rules depend on each example, while a causal pattern usually depends on query and
    key indices.

    These masks can be combined with logical AND. An entirely masked query is invalid:
    softmax over all $-\infty$ values is undefined. The scratch function raises an
    explicit error instead of silently creating `NaN`.
    """),

    code(r"""
    def causal_allowed(query_length, key_length=None):
        key_length = query_length if key_length is None else key_length
        query_positions = np.arange(query_length)[:, None]
        key_positions = np.arange(key_length)[None, :]
        return key_positions <= query_positions


    batch_size, sequence_length, feature_width = 2, 5, 4
    mask_inputs = generator.normal(size=(batch_size, sequence_length, feature_width))
    key_is_real = np.array(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    causal = causal_allowed(sequence_length)
    combined_allowed = causal[None, :, :] & key_is_real[:, None, :]

    masked_outputs, masked_weights, _ = scaled_dot_product_attention(
        mask_inputs, mask_inputs, mask_inputs, allowed=combined_allowed
    )

    print("combined mask shape:", combined_allowed.shape)
    print("attention shape:    ", masked_weights.shape)
    print("example 1 weights on padded keys:", masked_weights[0, :, 3:].max())
    print("maximum future weight:", masked_weights[:, np.triu_indices(sequence_length, 1)[0],
                                                    np.triu_indices(sequence_length, 1)[1]].max())
    assert np.all(masked_weights[~combined_allowed] == 0.0)
    """),

    code(r"""
    # Behavioral causal test: changing the future must not change an earlier output.
    original = generator.normal(size=(1, 6, 4))
    changed_future = original.copy()
    changed_future[:, 4:, :] += 1000.0
    causal_six = causal_allowed(6)[None, :, :]

    original_output, _, _ = scaled_dot_product_attention(
        original, original, original, allowed=causal_six
    )
    changed_output, _, _ = scaled_dot_product_attention(
        changed_future, changed_future, changed_future, allowed=causal_six
    )

    print("largest change through position 3:", np.max(np.abs(original_output[:, :4] - changed_output[:, :4])))
    assert np.allclose(original_output[:, :4], changed_output[:, :4])
    """),

    md(r"""
    ## 7 · Multi-head attention

    One head creates one attention-weight matrix. Multiple heads learn separate
    projection spaces:

    $$
    \operatorname{head}_r=
    \operatorname{Attention}(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)})
    $$

    $$
    \operatorname{MHA}(X)=
    \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W_O
    $$

    With $d_{model}=8$ and $H=2$, each head commonly uses $d_h=4$. Shapes:

    ```text
    X                 (B, T, 8)
      ↓ three projections
    Q, K, V           (B, T, 8)
      ↓ split heads
    Qh, Kh, Vh        (B, 2, T, 4)
      ↓ attention
    head outputs      (B, 2, T, 4)
      ↓ merge heads
    concatenated      (B, T, 8)
      ↓ output projection
    final output      (B, T, 8)
    ```

    Multiple heads provide capacity for different routing patterns. They do not
    guarantee human-readable specialization, and different-looking random maps are not
    evidence that a head has learned syntax or meaning.
    """),

    code(r"""
    def split_heads(projected, number_of_heads):
        batch, length, model_width = projected.shape
        head_width = model_width // number_of_heads
        return projected.reshape(batch, length, number_of_heads, head_width).transpose(0, 2, 1, 3)


    def merge_heads(head_outputs):
        batch, heads, length, head_width = head_outputs.shape
        return head_outputs.transpose(0, 2, 1, 3).reshape(batch, length, heads * head_width)


    def multi_head_attention_numpy(inputs, W_q, W_k, W_v, W_o, number_of_heads, allowed=None):
        queries = split_heads(inputs @ W_q, number_of_heads)
        keys = split_heads(inputs @ W_k, number_of_heads)
        values = split_heads(inputs @ W_v, number_of_heads)
        if allowed is not None:
            allowed = np.asarray(allowed, dtype=bool)
            if allowed.ndim == 3:
                allowed = allowed[:, None, :, :]
        head_outputs, head_weights, _ = scaled_dot_product_attention(
            queries, keys, values, allowed=allowed
        )
        merged = merge_heads(head_outputs)
        return merged @ W_o, head_weights


    B, T, D, H = 2, 4, 8, 2
    multi_inputs = generator.normal(size=(B, T, D))
    W_q = generator.normal(0, 0.2, size=(D, D))
    W_k = generator.normal(0, 0.2, size=(D, D))
    W_v = generator.normal(0, 0.2, size=(D, D))
    W_o = generator.normal(0, 0.2, size=(D, D))
    numpy_multi_output, numpy_head_weights = multi_head_attention_numpy(
        multi_inputs, W_q, W_k, W_v, W_o, H
    )

    print("multi-head output:", numpy_multi_output.shape)
    print("per-head weights: ", numpy_head_weights.shape)
    """),

    code(r"""
    # Copy the exact NumPy matrices into PyTorch. A close match verifies projection,
    # head splitting, scaling, softmax axis, merging, and output projection together.
    torch_mha = nn.MultiheadAttention(D, H, bias=False, batch_first=True, dropout=0.0)
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(
            torch.tensor(np.concatenate([W_q.T, W_k.T, W_v.T], axis=0), dtype=torch.float32)
        )
        torch_mha.out_proj.weight.copy_(torch.tensor(W_o.T, dtype=torch.float32))

    torch_output, torch_head_weights = torch_mha(
        torch.tensor(multi_inputs, dtype=torch.float32),
        torch.tensor(multi_inputs, dtype=torch.float32),
        torch.tensor(multi_inputs, dtype=torch.float32),
        need_weights=True,
        average_attn_weights=False,
    )

    maximum_output_difference = np.max(
        np.abs(numpy_multi_output - torch_output.detach().numpy())
    )
    maximum_weight_difference = np.max(
        np.abs(numpy_head_weights - torch_head_weights.detach().numpy())
    )
    print("maximum output difference:", maximum_output_difference)
    print("maximum weight difference:", maximum_weight_difference)
    assert maximum_output_difference < 1e-6
    assert maximum_weight_difference < 1e-6
    """),

    md(r"""
    ## 8 · Self-attention, cross-attention, and retrieval are related but different

    ### Self-attention

    Queries, keys, and values are projected from one sequence. Encoder self-attention
    may be bidirectional. A causal decoder uses a future mask.

    ### Cross-attention

    Queries come from one sequence; keys and values come from another. In an
    encoder–decoder translation model, decoder states query encoder states. Usually
    $T_q\ne T_k$, which the general formula already supports.

    ### Retrieval-augmented generation

    Retrieval also matches a query against keys and returns useful values, so the
    analogy is helpful. But RAG is not necessarily a neural cross-attention layer.
    Many decoder-only systems retrieve text, place it in the prompt, and then use
    ordinary causal self-attention over the combined tokens. Keep the conceptual
    analogy separate from the implemented architecture.
    """),

    code(r"""
    # Cross-attention has different query and source lengths.
    decoder_queries = generator.normal(size=(2, 3, 6))
    encoder_keys = generator.normal(size=(2, 5, 6))
    encoder_values = generator.normal(size=(2, 5, 9))
    cross_output, cross_weights, _ = scaled_dot_product_attention(
        decoder_queries, encoder_keys, encoder_values
    )
    print("cross-attention weights:", cross_weights.shape, "= (B, T_query, T_source)")
    print("cross-attention output: ", cross_output.shape, "= (B, T_query, d_value)")
    """),

    md(r"""
    ## 9 · Attention alone does not know order

    Without position information, self-attention is **permutation-equivariant**:

    > Permute the input rows and the output rows are permuted in the same way.

    It is not permutation-invariant because the ordered output tensor changes. If you
    later sum or average all rows, that pooled result becomes permutation-invariant.

    This is why a Transformer adds position information. DL-08 will compare sinusoidal
    positions and learned positions and will explain causal order separately from
    positional identity.
    """),

    code(r"""
    order_inputs = generator.normal(size=(1, 6, 8))
    identity = np.eye(8)
    original_order_output, _ = multi_head_attention_numpy(
        order_inputs, identity, identity, identity, identity, number_of_heads=2
    )
    permutation = np.array([3, 0, 5, 1, 4, 2])
    permuted_inputs = order_inputs[:, permutation, :]
    permuted_output, _ = multi_head_attention_numpy(
        permuted_inputs, identity, identity, identity, identity, number_of_heads=2
    )

    equivariant = np.allclose(permuted_output, original_order_output[:, permutation, :])
    pooled_invariant = np.allclose(permuted_output.mean(axis=1), original_order_output.mean(axis=1))
    print("permutation equivariant:", equivariant)
    print("mean-pooled result invariant:", pooled_invariant)
    assert equivariant and pooled_invariant
    """),

    md(r"""
    ## 10 · Cost: direct access is not free

    For one head, the score matrix has $T_qT_k$ entries. Self-attention with length $T$
    therefore stores or processes $T^2$ pair scores.

    | Length $T$ | Scores per head | Float32 size for scores only |
    |---:|---:|---:|
    | 512 | 262,144 | 1 MiB |
    | 4,096 | 16,777,216 | 64 MiB |
    | 32,768 | 1,073,741,824 | 4 GiB |

    Those numbers exclude batches, heads, gradients, Q/K/V tensors, and other layers.

    **Flash Attention** changes how exact attention is tiled and moved through the GPU
    memory hierarchy. It can avoid materializing the full score matrix in expensive
    high-bandwidth memory, reducing auxiliary memory traffic. It does **not** turn dense
    attention's pairwise arithmetic into linear-time computation.

    **KV caching** stores past keys and values during autoregressive inference. The new
    token does not recompute their projections. It still compares its query with the
    growing history, so per-token attention work and cache size grow with context.

    Sparse, sliding-window, linearized, and state-space approaches change other parts
    of the tradeoff. They are alternatives to evaluate, not free upgrades.
    """),

    code(r"""
    cost_rows = []
    for length in (512, 4096, 32768):
        score_count = length * length
        cost_rows.append(
            {
                "T": length,
                "scores per head": score_count,
                "float32 MiB": score_count * 4 / (1024**2),
            }
        )
    display(pd.DataFrame(cost_rows))
    """),

    md(r"""
    ## 11 · Reading attention maps responsibly

    An attention map answers:

    > For this head and query, how were value vectors mixed at this layer?

    It does not by itself prove:

    - which token caused the final prediction;
    - that the model used a human-like reason;
    - that one head has a stable semantic role;
    - that low-weight tokens were unimportant through other layers or residual paths.

    Use maps for shape checks, mask checks, routing diagnostics, and hypothesis
    generation. For decision explanation, combine interventions, gradients, controlled
    ablations, and task-specific evaluation. Attention weight is not causal importance.
    """),

    code(r"""
    fig, axes = plt.subplots(1, H, figsize=(10, 4))
    for head_index, axis in enumerate(axes):
        image = axis.imshow(numpy_head_weights[0, head_index], cmap="viridis")
        axis.set_title(f"random head {head_index + 1}")
        axis.set_xlabel("key position")
        axis.set_ylabel("query position")
        plt.colorbar(image, ax=axis)
    plt.suptitle("Different random maps are not evidence of learned specialization")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 12 · When to use attention—and when not to

    | Method | Main advantage | Main limitation | Good fit |
    |---|---|---|---|
    | recurrence | compact streaming state | serial training path | low-latency streams, modest data |
    | temporal CNN | parallel local pattern detection | designed receptive field | local motifs and signals |
    | dense attention | direct content-based access | quadratic pair matrix | moderate sequences, long relations |
    | local/sparse attention | cheaper structured access | may miss global links | very long locally structured inputs |
    | retrieval system | external searchable memory | retrieval errors and system complexity | changing or very large corpora |

    Use attention when relationships between positions should be selected dynamically
    from their content. Do not use dense attention automatically when order is absent,
    a simple aggregate is sufficient, sequences are too long for the budget, or a
    streaming state model better matches latency and memory constraints.
    """),

    md(r"""
    ## 13 · Failure modes

    | Symptom | Likely cause | Check | Response |
    |---|---|---|---|
    | validation leaks future targets | causal mask absent or reversed | perturb future-token test | fix Boolean direction; test behavior |
    | `NaN` attention row | every key masked | allowed count per query | reject or define a valid fallback |
    | padding receives weight | key-padding mask missing | inspect padded columns | combine padding and causal masks |
    | shapes broadcast silently | head or batch axis misplaced | print `(B,H,T,d_h)` stages | assert every contract |
    | outputs ignore word/order changes | no positional representation | permutation test | add and verify positions in DL-08 |
    | memory grows quadratically | dense score matrix | estimate $BHT^2$ | Flash, local/sparse design, shorter context |
    | all heads look alike | redundant learned projections | compare trained maps and ablations | validate fewer heads or regularization |
    | explanation claim is unstable | weights treated as causality | intervention test | use multiple explanation methods |

    Mask values also need dtype care. Boolean masks express intent clearly. Frameworks
    may accept additive masks, but large finite negatives can behave differently across
    low-precision dtypes, and mask conventions differ between APIs. Test the behavior,
    not just the mask's printed triangle.
    """),

    md(r"""
    ## 14 · Real scenario: matching a question to evidence

    A support system has a question representation and several candidate evidence
    representations. Content-based weighting can help a model mix the most relevant
    evidence.

    Before calling this a production RAG system, separate the stages:

    1. retrieval chooses candidate documents from an external index;
    2. optional reranking refines candidate order;
    3. the generator consumes evidence, often as prompt tokens in a decoder-only model;
    4. grounded evaluation checks whether claims are supported.

    The generator's internal attention does not repair missing evidence. High attention
    to an irrelevant chunk does not make the answer grounded. Measure retrieval recall,
    answer faithfulness, citation correctness, latency, context usage, and failure under
    adversarial or conflicting evidence.
    """),

    md(r"""
    ## 15 · Check your understanding

    1. Why are keys and values separate roles?
    2. Which axis receives softmax, and why?
    3. Calculate the shape of $QK^\top$ for $Q:(8,12,16)$ and $K:(8,20,16)$.
    4. Why does the scale use $\sqrt{d_k}$ rather than $d_k$?
    5. What differs between a causal mask and a key-padding mask?
    6. Why is an entirely masked row invalid?
    7. Trace `(B,T,d_model)` through four heads.
    8. Why is self-attention permutation-equivariant rather than invariant?
    9. What does Flash Attention improve without changing quadratic arithmetic?
    10. Why is an attention map not automatically an explanation?
    """),

    md(r"""
    ## 16 · Practice and mini-project

    ### Beginner

    1. Recalculate Section 3 after changing the query to $[0,1]$.
    2. Draw a `4×4` causal mask using `True` for allowed positions.

    ### Intermediate

    3. Add a finite-difference check for one query element. Explain which weights and
       output coordinates change.
    4. Extend the PyTorch comparison to a combined causal and padding mask. Verify both
       weights and outputs.

    ### Challenge

    5. Train one- and four-head models on a small key–value retrieval task across three
       seeds. Compare validation accuracy, variation, parameter count, and latency.
       Inspect maps only after measuring task performance.

    ### Mini-project · Evidence selector

    **Goal:** select supporting policy clauses for a customer question.

    **Columns:** `question_id`, `question_vector`, `clause_id`, `clause_vector`,
    `is_supporting`, `document_id`, and `policy_version`.

    **Workflow:** split by policy version or document before training; create negative
    candidates; compare cosine retrieval, single-head scoring, and multi-head scoring;
    mask padding; report Recall@k and MRR; test missing-evidence and conflicting-evidence
    cases; never claim answer faithfulness from attention weights alone.

    **Expected output:** shape assertions, mask behavior tests, multi-seed metrics,
    latency and memory estimates, attention diagnostics, and a written limitation.

    **Evaluation:** no document leakage, correct masks, fair candidate sets, validation-
    only selection, and claims supported by retrieval metrics.
    """),

    md(r"""
    ## 17 · Summary and memory aid

    Attention is a learned weighted retrieval operation:

    $$
    \operatorname{Attention}(Q,K,V)
    =\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
    $$

    Queries express what is needed, keys support matching, and values carry retrieved
    content. Scaling controls score variance. Masks define legal information flow.
    Multiple heads repeat the operation in different learned projection spaces. Dense
    access shortens information paths and parallelizes full-sequence training, but its
    score matrix grows quadratically and causal generation remains sequential across
    newly produced tokens.

    **Memory aid:** *Query the labels, soften the scores, and blend the values—only from
    positions the mask allows.*

    DL-08 comes next because attention alone has no position representation, residual
    pathway, normalization, feed-forward transformation, or language-model objective.
    """),
]


build("04_deep_learning/07_attention_mechanism.ipynb", cells)
