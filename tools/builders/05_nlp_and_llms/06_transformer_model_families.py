"""Build NLP-06: Transformer families as masks, objectives, and behavior."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # NLP-06 · Transformer Model Families

    **Prerequisites:** DL-07, DL-08, NLP-01, and the tiny-language-model checkpoint  
    **Estimated mastery time:** 7–10 hours, including the project assessment  
    **Next canonical lesson:** NLP-02 · Sentence Embeddings

    GPT, BERT, and T5 are not three unrelated inventions. They reuse Transformer
    components but enforce different information flow and train against different
    targets.

    This lesson asks four questions for every family:

    1. Which positions may this representation read?
    2. What is hidden or shifted in the input?
    3. Which target produces the loss?
    4. What output structure matches the task?

    Brand names are memory aids. Masks, objectives, tensor shapes, and measured
    behavior are the transferable knowledge.
    """),

    md(r"""
    ## 1 · Learning outcomes

    You will be able to:

    - derive causal, bidirectional-with-padding, and cross-attention visibility;
    - trace `(B,H,T_query,T_key)` for every attention operation;
    - construct next-token, masked-token, classification, and source-to-target targets;
    - explain why padding masks and masked-language-model tokens have different jobs;
    - prove that later input cannot change an earlier decoder logit;
    - prove that visible right context can change an earlier encoder representation;
    - prove that padded key IDs cannot change valid encoder outputs;
    - distinguish decoder self-attention from encoder–decoder cross-attention;
    - explain teacher forcing in both decoder-only and encoder–decoder training;
    - compare teacher-forced token accuracy with free-running exact match;
    - choose a family—or a simpler model—from information and output requirements;
    - avoid presenting synthetic memorization as general language ability.

    ```mermaid
    flowchart TD
        A[What information is legal?] --> B{One sequence or two?}
        B -->|one, predict future| C[Causal decoder]
        B -->|one, understand complete input| D[Bidirectional encoder]
        B -->|source and target separate| E[Encoder-decoder]
        C --> F[Next-token loss]
        D --> G[Masked-token or task head]
        E --> H[Shifted target plus cross-attention]
    ```
    """),

    md(r"""
    ## 2 · One comparison to orient yourself

    | Family | Self-attention visibility | Typical training signal | Natural output |
    |---|---|---|---|
    | decoder-only | current and earlier target positions | next token | continuation |
    | encoder-only | all real input positions | reconstructed token or task label | contextual states or label |
    | encoder–decoder | encoder: full source; decoder: earlier target; cross: full source | next target token | transformed sequence |

    Analogies:

    - A causal decoder writes while the unwritten page is covered.
    - A bidirectional encoder reviews a completed page.
    - An encoder–decoder writes a separate output while consulting a source page.

    The analogy stops there. The models calculate vectors and optimize losses; they do
    not read or understand in the human sense.
    """),

    code(r"""
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    candidates = [Path.cwd(), *Path.cwd().parents]
    repository_root = next(
        path for path in candidates if (path / "projects/transformer_families").exists()
    )
    project_root = repository_root / "projects" / "transformer_families"
    sys.path.insert(0, str(project_root / "src"))

    from transformer_families.models import (
        DecoderOnlyModel,
        EncoderDecoderModel,
        EncoderOnlyModel,
        FamilyConfig,
    )
    from transformer_families.training import (
        BOS,
        EOS,
        MASK,
        make_classification_data,
        make_cycle_sequences,
        make_reversal_data,
        run_family_lab,
    )

    torch.manual_seed(42)
    config = FamilyConfig()
    print(config)
    """),

    md(r"""
    ## 3 · Visibility matrices before model names

    Let `True` mean a query may read a key.

    For target length 4, causal visibility is:

    $$
    \begin{bmatrix}
    1&0&0&0\\
    1&1&0&0\\
    1&1&1&0\\
    1&1&1&1
    \end{bmatrix}
    $$

    A bidirectional encoder permits all **real** keys. If the last two entries are
    padding, their columns must be blocked for every query.

    Cross-attention is rectangular. With target length $T_t=3$ and source length
    $T_s=5$, scores have shape:

    $$
    (B,H,T_t,T_s)=(B,H,3,5)
    $$

    Decoder states provide queries; encoder states provide keys and values.
    """),

    code(r"""
    target_length, source_length = 4, 5
    causal_visibility = torch.tril(torch.ones(target_length, target_length, dtype=torch.bool))
    source_is_real = torch.tensor([True, True, True, False, False])
    encoder_visibility = source_is_real.repeat(source_length, 1)
    cross_visibility = source_is_real.repeat(3, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, matrix, title in zip(
        axes,
        [causal_visibility, encoder_visibility, cross_visibility],
        ["causal decoder", "encoder with padding", "cross-attention"],
    ):
        axis.imshow(matrix, vmin=0, vmax=1, cmap="gray")
        axis.set_xlabel("key position")
        axis.set_ylabel("query position")
        axis.set_title(title)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 4 · One shared attention calculation

    All three families can reuse:

    $$
    Q=Q_{states}W_Q,\quad K=C_{states}W_K,\quad V=C_{states}W_V
    $$

    $$
    A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)
    $$

    $$
    O=AV
    $$

    For self-attention, `Q_states` and `C_states` are the same sequence. For
    cross-attention they are different. $d_h$ is one head's width, and $M$ blocks
    illegal keys before softmax.

    The block around attention—not the dot-product formula—determines the family.
    """),

    code(r"""
    def visible_attention(query_states, context_states, allowed):
        scores = query_states @ context_states.transpose(-2, -1)
        scores = scores / np.sqrt(query_states.shape[-1])
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ context_states, weights


    query_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    context_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    allowed = torch.tensor([[[True, True, False], [True, True, True]]])
    retrieved, weights = visible_attention(query_states, context_states, allowed)
    print("score/weight shape:", weights.shape, "= (B,T_query,T_key)")
    print("weights:\n", weights)
    print("retrieved:\n", retrieved)
    assert torch.all(weights[~allowed] == 0)
    """),

    md(r"""
    ## 5 · Decoder-only: shifted targets and causal behavior

    For `[4,5,6,7]`:

    ```text
    decoder input:  [4,5,6]
    target:         [5,6,7]
    ```

    At position 1, the model sees input IDs 4 and 5 and predicts 6. Teacher forcing
    supplies the correct earlier input tokens for all positions during training. The
    causal mask prevents it from reading later inputs, so all position losses can be
    calculated in parallel.

    $$
    L_{causal}=-\frac1{BT}\sum_{b,t}\log p(y_{b,t}\mid x_{b,0:t})
    $$

    Generation remains sequential because the next input ID is the model's newly
    selected output.
    """),

    code(r"""
    cycle = make_cycle_sequences(count=3, length=6)
    decoder_inputs, decoder_targets = cycle[:, :-1], cycle[:, 1:]
    print("full row: ", cycle[0].tolist())
    print("input:    ", decoder_inputs[0].tolist())
    print("target:   ", decoder_targets[0].tolist())
    assert torch.equal(decoder_inputs[:, 1:], decoder_targets[:, :-1])

    decoder = DecoderOnlyModel(config).eval()
    original = torch.tensor([[4, 5, 6, 7, 8]])
    changed_future = torch.tensor([[4, 5, 6, 12, 13]])
    real = torch.ones_like(original, dtype=torch.bool)
    with torch.no_grad():
        original_logits = decoder(original, real)
        changed_logits = decoder(changed_future, real)
    earlier_difference = (original_logits[:, :3] - changed_logits[:, :3]).abs().max().item()
    print("largest earlier-logit change after future perturbation:", earlier_difference)
    assert earlier_difference == 0.0
    """),

    md(r"""
    ## 6 · Encoder-only: reconstruct or classify with full context

    ### Masked-token prediction

    A dedicated `[MASK]` ID replaces selected input tokens. The targets retain the
    originals, and loss is calculated only at selected positions $R$:

    $$
    L_{MLM}=-\frac1{|R|}\sum_{t\in R}\log p(x_t\mid \widetilde{x})
    $$

    `[MASK]` is content corruption. A **padding mask** is an attention rule that keeps
    nonexistent batch filler from becoming a key. They are not interchangeable.

    ### Classification

    A sequence representation can come from a special classification token or masked
    mean pooling. The project uses:

    $$
    h_{pool}=\frac{\sum_t m_th_t}{\max(1,\sum_t m_t)}
    $$

    where $m_t=1$ for real tokens and 0 for padding. A linear head maps $h_{pool}$ to
    class logits.
    """),

    code(r"""
    original_tokens = make_cycle_sequences(count=2, length=7)
    masked_inputs = original_tokens.clone()
    mask_position = 3
    masked_targets = original_tokens[:, mask_position]
    masked_inputs[:, mask_position] = MASK
    print("original:", original_tokens[0].tolist())
    print("corrupted:", masked_inputs[0].tolist())
    print("target at hidden position:", int(masked_targets[0]))

    encoder = EncoderOnlyModel(config).eval()
    first = torch.tensor([[4, 5, 6, 7, 8]])
    changed_right_context = torch.tensor([[4, 5, 6, 7, 12]])
    all_real = torch.ones_like(first, dtype=torch.bool)
    with torch.no_grad():
        first_hidden = encoder.encode(first, all_real)
        changed_hidden = encoder.encode(changed_right_context, all_real)
    right_context_effect = (first_hidden[:, 0] - changed_hidden[:, 0]).abs().max().item()
    print("change at position 0 after changing visible position 4:", right_context_effect)
    assert right_context_effect > 1e-6
    """),

    code(r"""
    # Padding IDs themselves may vary; valid outputs must not care because those keys
    # are masked. Padded query rows may still exist, so compare only real positions.
    padded_a = torch.tensor([[4, 5, 6, 0, 0]])
    padded_b = torch.tensor([[4, 5, 6, 12, 13]])
    padding_mask = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        hidden_a = encoder.encode(padded_a, padding_mask)
        hidden_b = encoder.encode(padded_b, padding_mask)
    valid_difference = (hidden_a[:, :3] - hidden_b[:, :3]).abs().max().item()
    print("valid-output change after altering padded IDs:", valid_difference)
    assert valid_difference == 0.0

    class_inputs, class_targets = make_classification_data(count=4)
    class_mask = torch.ones_like(class_inputs, dtype=torch.bool)
    print("classification logits shape:", encoder.classify(class_inputs, class_mask).shape)
    print("class targets:", class_targets.tolist())
    """),

    md(r"""
    ## 7 · Encoder–decoder: source states plus shifted target

    Three attention operations coexist:

    | Operation | Queries | Keys/values | Mask |
    |---|---|---|---|
    | encoder self-attention | source | source | source padding |
    | decoder self-attention | target prefix | target prefix | causal + target padding |
    | decoder cross-attention | decoder states | encoder states | source padding |

    For source `[4,6,8]` and desired target `[8,6,4,EOS]`:

    ```text
    decoder input: [BOS,8,6,4]
    target:        [8,6,4,EOS]
    ```

    This is teacher forcing again. During free-running generation, the decoder receives
    its own earlier outputs instead.
    """),

    code(r"""
    source, target_inputs, target_outputs = make_reversal_data(count=3, source_length=5)
    print("source:       ", source[0].tolist())
    print("decoder input:", target_inputs[0].tolist(), "starts with BOS", BOS)
    print("target:       ", target_outputs[0].tolist(), "ends with EOS", EOS)
    assert target_inputs[0, 0] == BOS
    assert target_outputs[0, -1] == EOS
    assert torch.equal(target_inputs[:, 1:], target_outputs[:, :-1])

    encoder_decoder = EncoderDecoderModel(config).eval()
    source_mask = torch.ones_like(source, dtype=torch.bool)
    target_mask = torch.ones_like(target_inputs, dtype=torch.bool)
    with torch.no_grad():
        base_logits = encoder_decoder(source, target_inputs, source_mask, target_mask)
        changed_source = source.clone()
        changed_source[:, -1] = 13
        source_changed_logits = encoder_decoder(
            changed_source, target_inputs, source_mask, target_mask
        )
        changed_target_future = target_inputs.clone()
        changed_target_future[:, 3:] = 12
        target_changed_logits = encoder_decoder(
            source, changed_target_future, source_mask, target_mask
        )

    print("source change affects first decoder logit:",
          (base_logits[:, 0] - source_changed_logits[:, 0]).abs().max().item())
    print("future target change affects first three logits:",
          (base_logits[:, :3] - target_changed_logits[:, :3]).abs().max().item())
    print("cross-attention score shape:",
          encoder_decoder.decoder_blocks[0].cross_attention.last_score_shape)
    assert (base_logits[:, 0] - source_changed_logits[:, 0]).abs().max() > 1e-6
    assert torch.allclose(base_logits[:, :3], target_changed_logits[:, :3], atol=1e-6)
    """),

    md(r"""
    ## 8 · Controlled training lab

    The local project trains four objectives with one shared configuration:

    - decoder next-token prediction on digit cycles;
    - encoder masked-token reconstruction;
    - encoder sequence classification;
    - encoder–decoder sequence reversal.

    These are wiring diagnostics. They show that gradients can teach each information
    path. They do not estimate performance on natural language or unseen distributions.
    """),

    code(r"""
    report = run_family_lab(seed=42, steps=120)
    evidence_table = pd.DataFrame(
        [
            {
                "objective": "decoder next token",
                "initial loss": report["gpt_decoder_only"]["initial_loss"],
                "final loss": report["gpt_decoder_only"]["final_loss"],
                "accuracy": report["gpt_decoder_only"]["token_accuracy"],
            },
            {
                "objective": "encoder masked token",
                "initial loss": report["bert_encoder_only"]["mlm_initial_loss"],
                "final loss": report["bert_encoder_only"]["mlm_final_loss"],
                "accuracy": report["bert_encoder_only"]["mlm_accuracy"],
            },
            {
                "objective": "encoder classification",
                "initial loss": report["bert_encoder_only"]["classification_initial_loss"],
                "final loss": report["bert_encoder_only"]["classification_final_loss"],
                "accuracy": report["bert_encoder_only"]["classification_accuracy"],
            },
            {
                "objective": "source-to-target reversal",
                "initial loss": report["t5_encoder_decoder"]["initial_loss"],
                "final loss": report["t5_encoder_decoder"]["final_loss"],
                "accuracy": report["t5_encoder_decoder"]["teacher_forced_token_accuracy"],
            },
        ]
    )
    display(evidence_table)
    print("free-running reversal exact match:",
          report["t5_encoder_decoder"]["greedy_exact_match_first_eight"])
    assert np.all(evidence_table["final loss"] < evidence_table["initial loss"])
    """),

    md(r"""
    Teacher-forced token accuracy can look stronger than free-running generation because
    every decoder step receives the correct earlier target during training evaluation.
    Once one generated token is wrong, later inputs change. Report both metrics and do
    not label teacher-forced accuracy as generation quality.
    """),

    md(r"""
    ## 9 · Choosing a family from the task

    | Need | First candidate | Why | Baseline to try |
    |---|---|---|---|
    | open continuation | decoder-only | causal objective matches output | n-gram or template |
    | complete-text classification | encoder-only | both-side context and compact task head | TF-IDF + logistic regression |
    | sentence representation | encoder-only, then contrastive adaptation | one contextual state per input token | TF-IDF or averaged embeddings |
    | explicit source-to-output transformation | encoder–decoder | source and target flows stay separate | rules or retrieval/template |
    | exact document lookup | retrieval | no synthesis required | lexical search |

    A decoder-only model can be prompted to perform many tasks. That does not make it
    the cheapest, fastest, easiest to evaluate, or most natural architecture for each
    task.
    """),

    md(r"""
    ## 10 · Common mistakes and diagnostics

    | Symptom | Likely cause | Behavioral test | Fix |
    |---|---|---|---|
    | decoder loss impossibly easy | future leakage | perturb future input | restore causal mask |
    | encoder cannot use right context | causal mask copied from decoder | perturb later real token | use bidirectional visibility |
    | padding changes valid states | padded keys visible | alter only padded IDs | key-padding mask before softmax |
    | MLM loss trains every position | corruption mask confused with loss mask | inspect selected indices | gather loss only at hidden positions |
    | seq2seq ignores source | cross-attention absent/miswired | change source, freeze prefix | decoder Q; encoder K/V |
    | seq2seq target is off by one | BOS/target shift wrong | print first row | `[BOS,*y[:-1]] → y` |
    | teacher-forced score high, generation poor | exposure mismatch | free-running exact match | report both; improve training/data |
    | synthetic accuracy called mastery | memorization only | held-out rule variation | narrow claim; add generalization split |

    Also mask padded **loss targets**, not only attention keys, when variable-length
    outputs are batched. An attention mask controls information; an ignore index controls
    which target positions contribute to loss.
    """),

    md(r"""
    ## 11 · Production boundaries

    Real checkpoints add subword tokenization, special-token conventions, pretrained
    weights, architecture-specific normalization and positions, dropout, large-scale
    data, and versioned generation settings. Loading a class named “causal LM” or
    “masked LM” does not verify its data license, context contract, calibration,
    robustness, latency, or fitness for a business decision.

    Pin model and tokenizer revisions together. Test mask behavior on the exact library
    API. Evaluate on one shared task distribution and include a simpler baseline.
    No hosted API or external download is required for this lesson.
    """),

    md(r"""
    ## 12 · Student check

    1. Which axis differs between target and source in cross-attention scores?
    2. Why can BERT use a later token to reconstruct an earlier masked token?
    3. Why can a plain bidirectional encoder not generate left-to-right safely?
    4. What is the difference between `[MASK]`, a padding mask, and an ignored loss target?
    5. Write GPT input and target rows for five IDs.
    6. Write T5 decoder input and target rows using BOS and EOS.
    7. Where do Q, K, and V come from in decoder cross-attention?
    8. Why can teacher-forced and free-running metrics disagree?
    9. Why is encoder-only a natural starting point for sentence embeddings?
    10. Name a realistic task where no Transformer family is the best first choice.
    """),

    md(r"""
    ## 13 · Practice and mastery project

    **Beginner**

    1. Draw all masks for source length 4 and target length 3.
    2. Calculate one masked-token loss when the correct-token probability is `0.8`.

    **Intermediate**

    3. Mask two encoder positions and calculate loss only at those positions.
    4. Intentionally remove decoder causality, predict which behavioral test fails,
       observe it fail, and restore the mask.
    5. Add padded source and target rows. Mask attention keys and loss targets separately.

    **Challenge**

    6. Create held-out digit rules for all four objectives. Compare memorization and
       generalization across three seeds, then recommend the simplest adequate family.

    Complete `projects/transformer_families/MASTERY_CHECKPOINT.md`. Passing requires all
    automated invariants, real loss reduction for every objective, a repaired broken
    mask, at least **17/20**, and no zero on the causal, cross-attention, or target-shift
    explanations.
    """),

    md(r"""
    ## 14 · Summary and memory aid

    The families differ less in the attention equation than in legal information flow,
    corruption, targets, and output structure:

    - Decoder-only: hide the future and predict the next token.
    - Encoder-only: read the complete real input and learn representations or labels.
    - Encoder–decoder: encode a source, then causally produce a separate target while
      cross-attending to source states.

    **Memory aid:** *Cover the future to continue, uncover the input to represent, and
    keep source and target separate to transform.*

    NLP-02 comes next because sentence embeddings require a principled way to pool and
    train encoder token states—not merely select an encoder-shaped architecture.
    """),
]


build("05_nlp_and_llms/06_transformer_model_families.ipynb", cells)
