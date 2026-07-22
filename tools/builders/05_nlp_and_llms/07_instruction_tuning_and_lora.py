"""Build NLP-07: response-supervised instruction tuning and LoRA mechanics."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # NLP-07 · Instruction Tuning and LoRA

    **Prerequisites:** DL-03, DL-04, DL-08, NLP-03, and the pretraining checkpoint

    **Estimated mastery time:** 9–12 hours, including the checkpoint

    **Next canonical lesson:** NLP-08 · Preference Learning and Alignment

    Pretraining learns to continue text from a broad corpus. Instruction tuning changes
    the examples and the loss contract: a prompt provides context, while selected
    response tokens provide direct supervision.

    Full supervised fine-tuning updates the whole model. LoRA freezes selected base
    linear layers and learns low-rank corrections. That reduces trainable and optimizer
    state; it does not reduce the need for representative data, validation, retention
    tests, or safe deployment.
    """),

    md(r"""
    ## 1 · Learning outcomes

    - define an instruction example and a versioned chat template;
    - align input IDs, shifted target IDs, and response-only labels;
    - identify the first supervised target without an off-by-one error;
    - explain why masked prompt positions still affect response gradients;
    - calculate response-token cross-entropy manually;
    - compare full SFT, LoRA, prompting, and RAG by the problem they solve;
    - derive LoRA shapes, scaling, parameter counts, and zero-update initialization;
    - explain which adapter matrix receives gradient first when $B=0$;
    - insert adapters into selected modules and verify the frozen base;
    - merge an adapter and prove merged/unmerged outputs agree;
    - compare training, held-out task, format, safety, and retention evidence;
    - version base model, tokenizer, template, adapter, data, and evaluation together.

    ```mermaid
    flowchart LR
        A[Reviewed prompt-response data] --> B[Versioned template]
        B --> C[Tokenize and shift]
        C --> D[Mask prompt and padding labels]
        D --> E{Update strategy}
        E -->|all weights| F[Full SFT]
        E -->|low-rank deltas| G[LoRA]
        F --> H[Held-out and retention evaluation]
        G --> H
    ```
    """),

    md(r"""
    ## 2 · What problem does SFT solve?

    A base model may know relevant language but not reliably follow the desired
    interaction format. SFT demonstrates behavior through prompt–response examples.

    | Need | First intervention | Why |
    |---|---|---|
    | clearer instruction | prompt/template | no weight update |
    | current factual knowledge | RAG | update sources rather than model memory |
    | domain language distribution | continued pretraining | broad next-token exposure |
    | stable demonstrated response behavior | SFT | supervised response examples |
    | preference between plausible responses | preference optimization | paired/ranked signal after SFT |

    SFT is not a database update and does not guarantee factuality, safety, or obedience
    outside the demonstrated distribution.
    """),

    code(r"""
    import copy
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    torch.manual_seed(42)
    np.set_printoptions(precision=5, suppress=True)
    """),

    md(r"""
    ## 3 · The template is part of the model contract

    A conceptual example:

    ```text
    <BOS><USER>what guides an orbit?<END_USER><ASSISTANT>gravity guides an orbit.<EOS>
    ```

    The template defines roles, separators, turn endings, beginning/end tokens, and
    whether system instructions are present. Training and inference must render the
    same convention. A tokenizer must know the required special tokens and their IDs.

    Useful data fields include `example_id`, prompt/messages, response, source,
    reviewer status, license/consent, safety category, language, split, template
    version, and quality decision. Split by conversation, source, user, or task family
    before near-duplicate examples can cross boundaries.

    Data quality matters more than merely increasing rows: remove contradictory,
    truncated, unlicensed, private, unsafe, or template-corrupted examples through an
    auditable policy.
    """),

    md(r"""
    ## 4 · Shift first, then decide which targets receive loss

    Suppose tokenized IDs are:

    ```text
    sequence: [BOS, USER, Q, ASSISTANT, A1, A2, EOS]
    input:    [BOS, USER, Q, ASSISTANT, A1, A2]
    target:   [USER, Q, ASSISTANT, A1, A2, EOS]
    labels:   [-100,-100,-100,       A1, A2, EOS]
    ```

    The label aligned with input `ASSISTANT` is the first response token `A1`. This is
    the common off-by-one boundary.

    PyTorch cross-entropy conventionally ignores label `-100`. It is not a vocabulary
    token. Padding labels are also ignored. Prompt IDs remain in the input so response
    states can attend to them.
    """),

    code(r"""
    BOS, USER, QUESTION, ASSISTANT, ANSWER_1, ANSWER_2, EOS, PAD = range(8)
    sequence = [BOS, USER, QUESTION, ASSISTANT, ANSWER_1, ANSWER_2, EOS]
    response_start = 4  # index of ANSWER_1 in the unshifted sequence
    block_size = 9

    inputs = sequence[:-1]
    targets = sequence[1:]
    labels = [
        target if target_sequence_index >= response_start else -100
        for target_sequence_index, target in enumerate(targets, start=1)
    ]
    padding_needed = block_size - len(inputs)
    inputs += [PAD] * padding_needed
    labels += [-100] * padding_needed

    alignment = pd.DataFrame({"input_id": inputs, "label": labels})
    display(alignment)
    supervised_positions = [index for index, label in enumerate(labels) if label != -100]
    print("supervised positions:", supervised_positions)
    print("supervised labels:", [labels[index] for index in supervised_positions])
    assert [labels[index] for index in supervised_positions] == [ANSWER_1, ANSWER_2, EOS]
    """),

    md(r"""
    With token-level negative log-likelihood $\ell_t$ and response mask $m_t$:

    $$
    J_{SFT}=\frac{\sum_t m_t\ell_t}{\sum_t m_t}
    =-\frac{1}{\sum_t m_t}\sum_t m_t
    \log p_\theta(y_t\mid x_{0:t})
    $$

    | Symbol | Plain-language meaning |
    |---|---|
    | $J_{SFT}$ | one average loss for the batch or sequence |
    | $t$ | the token position currently being scored |
    | $m_t$ | `1` when position $t$ is supervised; otherwise `0` |
    | $\ell_t$ | negative log-probability at position $t$ |
    | $p_\theta$ | probabilities produced by the model with parameters $\theta$ |
    | $y_t$ | the correct next token at position $t$ |
    | $x_{0:t}$ | all input tokens available through position $t$ |
    | $\sum_t m_t$ | the number of supervised tokens used in the average |

    For losses `[2.0,1.0,0.4,0.2]` and mask `[0,0,1,1]`:

    $$
    J=(0.4+0.2)/2=0.3
    $$

    Some recipes train on prompt tokens too. That is a deliberate objective choice,
    not automatically an error, but response-only masking prevents long prompts from
    dominating a behavior-focused loss. Record the policy.
    """),

    code(r"""
    token_losses = torch.tensor([2.0, 1.0, 0.4, 0.2])
    response_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
    response_only_loss = (token_losses * response_mask).sum() / response_mask.sum()
    all_token_loss = token_losses.mean()
    print("response-only loss:", response_only_loss.item())
    print("all-token loss:    ", all_token_loss.item())
    assert torch.isclose(response_only_loss, torch.tensor(0.3))
    """),

    md(r"""
    ## 5 · Masked prompt labels do not block gradients through context

    `-100` removes a position's **direct target loss**. It does not detach its hidden
    state. A later response representation attends to prompt representations, so the
    response loss can backpropagate through prompt-token embeddings and attention paths.

    The next tiny graph demonstrates the principle without a Transformer: a response
    prediction depends on a prompt embedding plus a response-prefix embedding. Loss is
    attached only to the response prediction, yet both embeddings receive gradients.
    """),

    code(r"""
    prompt_embedding = torch.tensor([1.0, -0.5], requires_grad=True)
    response_prefix_embedding = torch.tensor([0.2, 0.4], requires_grad=True)
    output_weight = torch.tensor([0.7, -0.3], requires_grad=True)
    response_logit = (prompt_embedding + response_prefix_embedding) @ output_weight
    response_loss = (response_logit - 1.0) ** 2
    response_loss.backward()

    print("prompt gradient:", prompt_embedding.grad)
    print("response-prefix gradient:", response_prefix_embedding.grad)
    assert prompt_embedding.grad.abs().sum() > 0
    assert response_prefix_embedding.grad.abs().sum() > 0
    """),

    md(r"""
    ## 6 · Packing multiple examples safely

    Packing reduces padding, but one example must not accidentally supervise or attend
    through another unless the design explicitly allows it.

    At minimum:

    - place EOS or turn-boundary tokens between examples;
    - keep prompt and padding labels ignored independently for each example;
    - decide whether attention crosses example boundaries;
    - avoid splitting away the response or EOS through careless truncation;
    - report supervised tokens, not only total tokens.

    A block-diagonal attention mask isolates packed examples. Simple concatenation with
    causal attention allows later examples to read earlier unrelated examples.
    """),

    code(r"""
    segment_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    positions = torch.arange(len(segment_ids))
    causal = positions[None, :] <= positions[:, None]
    same_example = segment_ids[None, :] == segment_ids[:, None]
    packed_allowed = causal & same_example
    print(packed_allowed.int())
    assert not packed_allowed[3, 2]  # first token of example 1 cannot read example 0
    """),

    md(r"""
    ## 7 · Full SFT versus LoRA

    Full SFT updates every selected base parameter. With Adam-like optimization,
    training memory includes weights, gradients, and optimizer moments, often in
    different precisions.

    LoRA replaces a frozen linear transformation with:

    $$
    h=W_0x+\frac{\alpha}{r}BAx
    $$

    | Symbol | Shape | Meaning |
    |---|---:|---|
    | $x$ | $(d_{in},)$ | layer input |
    | $W_0$ | $(d_{out},d_{in})$ | frozen base weight |
    | $A$ | $(r,d_{in})$ | down projection |
    | $B$ | $(d_{out},r)$ | up projection |
    | $r$ | scalar | adapter rank |
    | $\alpha/r$ | scalar | update scale |

    One adapter has $r(d_{in}+d_{out})$ trainable weights instead of
    $d_{in}d_{out}$ base weights. LoRA does not necessarily reduce activation memory,
    sequence cost, inference latency, or total base-model storage.
    """),

    code(r"""
    d_in, d_out, rank, alpha = 3, 2, 1, 2.0
    W0 = torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.0, 0.0]])
    A = torch.tensor([[0.2, -0.1, 0.4]])
    B = torch.zeros((d_out, rank))
    x = torch.tensor([2.0, 1.0, -1.0])

    base_output = W0 @ x
    initial_lora_output = base_output + (alpha / rank) * B @ A @ x
    print("base output:", base_output)
    print("initial LoRA output:", initial_lora_output)
    print("base parameters:", W0.numel())
    print("adapter parameters:", A.numel() + B.numel())
    assert torch.equal(base_output, initial_lora_output)
    """),

    md(r"""
    Initializing $B=0$ makes $BA=0$, so the adapted layer exactly matches the base.
    On the first backward pass, $B$ can receive gradient because it multiplies the
    nonzero $Ax$. The gradient to $A$ contains $B^\top$, so it starts at zero. After
    $B$ moves, $A$ can learn. Initializing both matrices to zero would block both.
    """),

    code(r"""
    A_parameter = nn.Parameter(A.clone())
    B_parameter = nn.Parameter(B.clone())
    output = base_output + (alpha / rank) * B_parameter @ A_parameter @ x
    loss = output.square().sum()
    loss.backward()
    print("first-step |gradient A|:", A_parameter.grad.abs().sum().item())
    print("first-step |gradient B|:", B_parameter.grad.abs().sum().item())
    assert A_parameter.grad.abs().sum() == 0
    assert B_parameter.grad.abs().sum() > 0
    """),

    md(r"""
    ## 8 · Which modules receive adapters?

    Common targets include query/key/value and attention-output projections, and
    sometimes FFN projections. Names and storage conventions differ across models.
    A fused QKV matrix may need one adapter or slices; some libraries store linear
    weights transposed.

    Print matched module names, shapes, trainable flags, and counts. Fail if zero or an
    unexpected number of modules matched. Decide whether biases, embeddings, norms, and
    LM heads remain frozen. “LoRA rank 8” is incomplete without target modules and
    scaling/dropout policy.
    """),

    code(r"""
    class LoRALinear(nn.Module):
        def __init__(self, base_layer, rank=2, alpha=4.0):
            super().__init__()
            # Keep the original transformation but prevent optimizer updates to it.
            self.base_layer = base_layer
            for parameter in self.base_layer.parameters():
                parameter.requires_grad = False

            # A projects down to rank dimensions; B projects back to output size.
            self.adapter_a = nn.Linear(base_layer.in_features, rank, bias=False)
            self.adapter_b = nn.Linear(rank, base_layer.out_features, bias=False)

            # Random A and zero B preserve the base output while allowing B to learn.
            nn.init.normal_(self.adapter_a.weight, std=0.02)
            nn.init.zeros_(self.adapter_b.weight)
            self.scale = alpha / rank

        def forward(self, inputs):
            return self.base_layer(inputs) + self.scale * self.adapter_b(self.adapter_a(inputs))


    torch.manual_seed(7)
    base_linear = nn.Linear(5, 4)
    reference_linear = copy.deepcopy(base_linear).eval()
    adapted_linear = LoRALinear(copy.deepcopy(base_linear), rank=2, alpha=4.0).eval()
    sample_inputs = torch.randn(3, 5)
    initial_difference = (reference_linear(sample_inputs) - adapted_linear(sample_inputs)).abs().max().item()
    trainable_names = [name for name, parameter in adapted_linear.named_parameters() if parameter.requires_grad]
    print("initial maximum difference:", initial_difference)
    print("trainable parameters:", trainable_names)
    assert initial_difference == 0.0
    assert trainable_names == ["adapter_a.weight", "adapter_b.weight"]
    """),

    md(r"""
    ## 9 · Merging and unmerging

    For deployment, merge the adapter into a copy of the base weight:

    $$
    W_{merged}=W_0+\frac{\alpha}{r}BA
    $$

    Merging can remove adapter operations for that linear layer. Preserve the original
    base and adapter artifacts; repeated merge/unmerge in low precision can accumulate
    error. Verify logits on representative inputs before shipping.
    """),

    code(r"""
    with torch.no_grad():
        adapted_linear.adapter_b.weight.normal_(std=0.03)
        delta_weight = (
            adapted_linear.scale
            * adapted_linear.adapter_b.weight
            @ adapted_linear.adapter_a.weight
        )
        merged_linear = copy.deepcopy(adapted_linear.base_layer)
        merged_linear.weight.add_(delta_weight)

    unmerged_output = adapted_linear(sample_inputs)
    merged_output = merged_linear(sample_inputs)
    merge_difference = (unmerged_output - merged_output).abs().max().item()
    print("merged/unmerged maximum difference:", merge_difference)
    assert torch.allclose(unmerged_output, merged_output, atol=1e-6, rtol=1e-6)
    """),

    md(r"""
    ## 10 · Controlled full-SFT versus LoRA evidence

    The shared local lab starts both candidates from identical continued-pretraining
    weights. It masks prompt and padding labels, tunes four training examples, and
    evaluates separate examples. This is a wiring and overfitting diagnostic—not a
    production-quality instruction benchmark.
    """),

    code(r"""
    # Find the repository whether Jupyter started here or in a child directory.
    repository = next(
        path for path in [Path.cwd(), *Path.cwd().parents]
        if (path / "projects/language_model_adaptation").exists()
    )
    # Import the same tested implementation used by the project checkpoint.
    sys.path[:0] = [
        str(repository / "projects/language_model_adaptation/src"),
        str(repository / "projects/tiny_language_model/src"),
    ]
    from language_model_adaptation.lab import run_adaptation_lab

    experiment_report = run_adaptation_lab(seed=42)
    tuning = experiment_report["instruction_tuning"]
    evidence = pd.DataFrame(
        [
            {"method": method, **metrics}
            for method, metrics in (("full", tuning["full"]), ("lora", tuning["lora"]))
        ]
    )
    display(evidence)
    print("total base parameters:", tuning["total_base_parameters"])
    assert tuning["full"]["train_loss_after"] < tuning["full"]["train_loss_before"]
    assert tuning["lora"]["train_loss_after"] < tuning["lora"]["train_loss_before"]
    assert tuning["lora"]["zero_initial_logit_delta"] == 0.0
    """),

    code(r"""
    methods = ["full", "lora"]
    positions = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(positions - 0.18, [tuning[name]["train_loss_after"] for name in methods], 0.36, label="training")
    axes[0].bar(positions + 0.18, [tuning[name]["held_out_loss"] for name in methods], 0.36, label="held out")
    axes[0].set_xticks(positions, methods)
    axes[0].set_ylabel("response-token loss")
    axes[0].set_title("Loss is split-dependent")
    axes[0].legend()
    axes[1].bar(methods, [tuning[name]["trainable_parameters"] for name in methods])
    axes[1].set_ylabel("trainable parameters")
    axes[1].set_title("Optimization state differs")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    Read current values from the table rather than memorizing hard-coded numbers. The
    tiny run may show full SFT fitting training more aggressively while LoRA generalizes
    better on two held-out examples. That is evidence of overfitting in this setup, not
    proof that LoRA is universally superior.

    A fair comparison freezes the base checkpoint, tokenizer, template, train and
    validation rows, update/token budget, decoding, and evaluation. Learning rates may
    require method-specific validation; “same learning rate” is not automatically fair.
    """),

    md(r"""
    ## 11 · Evaluation that can justify tuning

    Training loss only confirms optimization on supplied responses. Evaluate:

    - held-out task correctness and format/schema validity;
    - prompt and task-family slices;
    - response length and truncation;
    - instruction conflicts and irrelevant instructions;
    - safety and refusal behavior appropriate to policy;
    - hallucination and groundedness where facts matter;
    - pretraining/domain and old-task retention;
    - calibration, latency, memory, and adapter loading;
    - several seeds when data is small.

    Exact-match loss may punish valid alternative wording. Use task-specific validators,
    calibrated human rubrics, and paired comparisons as appropriate. Keep the final test
    sealed until model and decoding choices are frozen.
    """),

    md(r"""
    ## 12 · Failure modes

    | Symptom | Likely cause | Check | Repair |
    |---|---|---|---|
    | prompt dominates supervised count | labels not masked | print alignment and counts | set prompt labels to ignore index |
    | first response token unsupervised | off-by-one boundary | inspect shifted row | align label at assistant-prefix input |
    | adapter changes initial logits | $B$ not zero | zero-delta test | zero-initialize one factor |
    | no adapter learns | both factors zero or no modules matched | gradients and names | randomize $A$, zero $B$; fail on zero matches |
    | tiny train loss, weak task score | memorization | held-out slices | improve/diversify data, regularize |
    | LoRA uses unexpected memory | activations/base/optimizer ignored | measured peak memory | profile full process |
    | merged model differs | transpose/scale/bias error | representative-logit test | verify $W_0+(\alpha/r)BA$ convention |
    | old behavior regresses | adapter changes shared representations | retention suite | lower capacity/update, replay, routing |
    | inference format fails | template mismatch | rendered prompt diff | version one renderer with tokenizer |

    Frozen base weights do not mean frozen behavior: the adapter changes activations
    throughout the model.
    """),

    md(r"""
    ## 13 · Practical extensions

    PEFT libraries automate module replacement, adapter state saving, and merging.
    Treat them as implementations of the same contract: inspect target modules,
    fan-in/fan-out convention, rank, alpha, dropout, bias policy, trainable counts, and
    merge equivalence.

    QLoRA stores a quantized frozen base while training adapters, reducing base-weight
    memory. It introduces quantization choices and kernels and does not quantize away
    activations or every optimizer cost. Evaluate quality and hardware behavior rather
    than assuming the advertised memory ratio applies unchanged.

    Multiple adapters can share one base, but routing, compatibility, cache management,
    authorization, and combined-adapter behavior become production concerns.
    """),

    md(r"""
    ## 14 · Student check and exercises

    1. Which input position predicts the first response token?
    2. What does label `-100` mean, and what does it not mean?
    3. Why can prompt embeddings receive gradients under response-only loss?
    4. Why should one LoRA factor start nonzero and the other at zero?
    5. Which factor receives gradient on the first step in this convention?
    6. What does rank control and not control?
    7. How do you calculate adapter parameter count?
    8. Why can frozen base weights still show retention regressions?
    9. What must be identical in a fair full-versus-LoRA comparison?
    10. When should prompting or RAG be tried before SFT?

    **Beginner:** build and print labels for two prompt–response rows with different
    lengths. Verify prompt and padding labels are ignored and EOS is supervised.

    **Intermediate:** compare ranks 1, 4, and 8 under three seeds. Report trainable
    state, held-out task score, retention, training time, and peak memory.

    **Challenge:** insert adapters into fused QKV and attention-output projections,
    save adapter-only state, reload it over the exact base, merge a copy, and verify
    merged/unmerged logits within `1e-6`.
    """),

    md(r"""
    ## 15 · Mastery checkpoint and summary

    Complete `projects/language_model_adaptation/SFT_LORA_CHECKPOINT.md`. Add evidence
    for rendered templates, supervised-token counts, zero initial delta, matched target
    modules, trainable counts, held-out and retention metrics, and merge equivalence.

    SFT changes the supervision distribution; LoRA changes the parameterization of the
    update. Neither changes what counts as trustworthy evaluation.

    **Memory aid:** *Render one contract, shift once, mask the prompt, adapt a measured
    set of weights, and judge behavior beyond training loss.*

    NLP-08 comes next because preference learning compares candidate responses after a
    model can already produce instruction-shaped behavior.
    """),
]


build("05_nlp_and_llms/07_instruction_tuning_and_lora.ipynb", cells)
