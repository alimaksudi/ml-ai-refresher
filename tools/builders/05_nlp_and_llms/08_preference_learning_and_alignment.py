"""Build NLP-08: preference data, reward models, DPO, and alignment evidence."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # NLP-08 · Preference Learning and Alignment

    **Prerequisites:** FND-02 (built on PRE-05), NLP-07, response-only loss, and sequence log probability

    **Estimated mastery time:** 10–13 hours, including the checkpoint

    **Next canonical lesson:** EVAL-02 · LLM Evaluation Foundations

    Supervised fine-tuning demonstrates one acceptable response. It cannot directly say
    that one plausible response is better than another. Preference learning adds that
    comparison signal.

    This lesson builds the comparison carefully: preference records, Bradley–Terry
    reward modeling, response sequence scores, and Direct Preference Optimization
    (DPO). It then places DPO beside reward-model-plus-RL workflows without pretending
    that a tiny offline experiment proves broad alignment, truthfulness, or safety.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you should be able to:

    - explain why preference data is introduced after instruction tuning;
    - design and audit `(prompt, chosen, rejected)` records;
    - distinguish a human choice, a reward-model score, and real-world quality;
    - calculate sigmoid, Bradley–Terry probability, and pairwise loss by hand;
    - calculate response-only sequence log probability;
    - identify length bias in summed sequence scores;
    - calculate every part of the DPO loss;
    - explain the policy, frozen reference, relative margin, and `beta`;
    - explain why DPO starts at `log(2)` when policy equals reference;
    - trace which parameters receive gradients;
    - distinguish DPO, a KL-regularized RL objective, and PPO;
    - evaluate preference gain together with task retention and independent safety tests.

    ```mermaid
    flowchart LR
        A[Instruction-tuned model] --> B[Candidates for one prompt]
        B --> C[Reviewed preference record]
        C --> D{Learning route}
        D --> E[Reward model]
        E --> F[RL policy optimization]
        D --> G[DPO with frozen reference]
        F --> H[Independent evaluation]
        G --> H
    ```
    """),

    md(r"""
    ## 2 · The practical problem

    Imagine two assistants answer the same uncertain question:

    - response A confidently invents a source;
    - response B says what is known and asks to verify the source.

    Both may be grammatical. A next-token objective does not directly express the
    reviewer’s tradeoff. SFT could provide response B as a target, but a comparison says
    something extra: **for this prompt and rubric, B is preferred over A**.

    Preference optimization is useful when several responses are plausible and the
    desired difference is easier to judge than to write from scratch. It is not the
    first fix for every problem.

    | Need | Try first | Why |
    |---|---|---|
    | clearer request | prompt or template | no training required |
    | current, sourced facts | RAG | preferences do not update a knowledge source |
    | known correct target response | SFT | direct supervision is simpler |
    | choose between plausible behaviors | preference learning | pairwise signal fits the question |
    | deterministic forbidden output | validation or policy control | model training is not a hard guarantee |
    """),

    code(r"""
    import copy
    import math
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    np.set_printoptions(precision=5, suppress=True)
    """),

    md(r"""
    ## 3 · A preference label is a scoped observation

    A useful record contains more than three text strings:

    - a stable example and prompt ID;
    - the exact prompt and chat-template version;
    - candidate A and candidate B in randomized display order;
    - the chosen candidate, or tie/abstain when allowed;
    - the rubric and policy version;
    - annotator identity or anonymous worker ID;
    - reason codes, confidence, language, domain, and timestamp;
    - candidate model and decoding metadata;
    - privacy, consent, license, safety, and review status;
    - split and duplicate-group identifiers.

    “Chosen” does not mean objectively true. It means a reviewer preferred that response
    under a particular rubric and context. Reviewers can disagree, misunderstand facts,
    follow a biased rubric, or react to answer length and style.

    Split by prompt, user, source, conversation, or task family before close variants
    leak across training and evaluation. Keep ties and abstentions visible rather than
    forcing uncertain judgments into false certainty.
    """),

    code(r"""
    preference_records = pd.DataFrame(
        [
            {
                "example_id": "pref-001",
                "prompt": "How should I use an uncertain claim?",
                "candidate_a": "State it as a proven fact.",
                "candidate_b": "Label the uncertainty and verify the source.",
                "chosen": "candidate_b",
                "rubric_version": "carefulness-v1",
                "annotator": "reviewer-07",
                "confidence": 0.9,
                "split": "train",
            },
            {
                "example_id": "pref-002",
                "prompt": "Which draft is clearer?",
                "candidate_a": "Use the short explanation with one example.",
                "candidate_b": "Use the short explanation with one example.",
                "chosen": "tie",
                "rubric_version": "clarity-v1",
                "annotator": "reviewer-12",
                "confidence": 0.6,
                "split": "review",
            },
        ]
    )

    required_columns = {
        "example_id", "prompt", "candidate_a", "candidate_b", "chosen",
        "rubric_version", "annotator", "confidence", "split",
    }
    assert required_columns.issubset(preference_records.columns)
    assert preference_records["example_id"].is_unique
    assert preference_records["chosen"].isin({"candidate_a", "candidate_b", "tie", "abstain"}).all()
    display(preference_records)
    """),

    md(r"""
    ## 4 · From two reward scores to a choice probability

    A reward model gives each prompt–response pair a scalar score. The
    Bradley–Terry model converts the score difference into a probability:

    $$
    P(y_w \succ y_l \mid x)=\sigma(r_w-r_l)
    $$

    $$
    \sigma(z)=\frac{1}{1+e^{-z}}
    $$

    $$
    J_{RM}=-\log \sigma(r_w-r_l)
    $$

    | Symbol | Plain-language meaning |
    |---|---|
    | $x$ | the prompt |
    | $y_w$ | the response labeled as the winner, or chosen response |
    | $y_l$ | the response labeled as the loser, or rejected response |
    | $r_w$ | reward-model score for the chosen response |
    | $r_l$ | reward-model score for the rejected response |
    | $\succ$ | “is preferred to” |
    | $\sigma$ | sigmoid, which maps any number into the interval `(0, 1)` |
    | $J_{RM}$ | pairwise reward-model loss; smaller is better on that label |

    If the chosen score is `2` and the rejected score is `1`, the margin is `1`.
    The predicted choice probability is about `0.731`, and the loss is about `0.313`.
    Reversing the margin produces a larger loss of about `1.313`.
    """),

    code(r"""
    def sigmoid(number):
        return 1.0 / (1.0 + math.exp(-number))


    def bradley_terry_loss(chosen_score, rejected_score):
        score_margin = chosen_score - rejected_score
        chosen_probability = sigmoid(score_margin)
        loss = -math.log(chosen_probability)
        return score_margin, chosen_probability, loss


    for chosen_score, rejected_score in [(2.0, 1.0), (1.0, 2.0), (1.0, 1.0)]:
        margin, probability, loss = bradley_terry_loss(chosen_score, rejected_score)
        print(
            f"chosen={chosen_score:.1f}, rejected={rejected_score:.1f}, "
            f"margin={margin:+.1f}, P(chosen)={probability:.3f}, loss={loss:.3f}"
        )

    assert math.isclose(bradley_terry_loss(1.0, 1.0)[2], math.log(2))
    """),

    md(r"""
    The model assumes that one scalar difference represents the comparison and that the
    sigmoid is a useful link from difference to probability. Real preferences may be
    intransitive or depend on dimensions hidden by one score. A reward model is a learned
    proxy for labels—not a truth meter.

    For repeated labels, report agreement by rubric slice. Raw percent agreement is easy
    to read; chance-corrected statistics can add context but do not repair a vague rubric.
    Inspect disagreement examples directly.
    """),

    md(r"""
    ## 5 · Sequence log probability: score only the response

    DPO does not use one probability for a whole sentence in a single step. An
    autoregressive model supplies one next-token probability at each response position.
    Their product is the sequence probability; their log probabilities add:

    $$
    \log \pi_\theta(y\mid x)
    =\sum_{t=1}^{T}\log \pi_\theta(y_t\mid x,y_{<t})
    $$

    | Symbol | Plain-language meaning |
    |---|---|
    | $\pi_\theta$ | the policy model with parameters $\theta$ |
    | $x$ | prompt tokens, used as context but not included in the score sum |
    | $y$ | the complete response |
    | $y_t$ | response token at position $t$ |
    | $y_{<t}$ | earlier response tokens |
    | $T$ | number of scored response tokens, usually including EOS |

    Suppose response-token probabilities are `0.50`, `0.25`, and `0.80`.

    $$
    P(y\mid x)=0.50\times0.25\times0.80=0.10
    $$

    $$
    \log P(y\mid x)=\log(0.50)+\log(0.25)+\log(0.80)\approx-2.303
    $$

    Prompt tokens still affect every conditional probability. They are simply excluded
    from the summed response score. This is the same boundary discipline learned in
    NLP-07.
    """),

    code(r"""
    response_token_probabilities = torch.tensor([0.50, 0.25, 0.80])
    response_token_log_probabilities = response_token_probabilities.log()
    sequence_log_probability = response_token_log_probabilities.sum()
    sequence_probability = response_token_probabilities.prod()

    print("token log probabilities:", response_token_log_probabilities)
    print("sequence probability:", sequence_probability.item())
    print("sequence log probability:", sequence_log_probability.item())
    print("log of the product:", sequence_probability.log().item())
    assert torch.allclose(sequence_log_probability, sequence_probability.log())
    """),

    md(r"""
    ### Length is a real comparison variable

    Because log probabilities are usually negative, a longer response has more terms to
    add. Summed scores can therefore prefer shorter text even when average token quality
    is similar. Dividing by length changes the objective and can create the opposite
    incentive. There is no automatic correction that is right for every task.

    Audit chosen/rejected length distributions, score margin by length difference, EOS
    handling, truncation, and results in matched-length slices. Keep the selected scoring
    convention identical for policy and reference.
    """),

    code(r"""
    short_log_probabilities = torch.tensor([-0.20, -0.20])
    long_log_probabilities = torch.tensor([-0.20, -0.20, -0.20, -0.20])

    length_comparison = pd.DataFrame(
        [
            {
                "response": "short",
                "tokens": len(short_log_probabilities),
                "sum_log_probability": float(short_log_probabilities.sum()),
                "mean_log_probability": float(short_log_probabilities.mean()),
            },
            {
                "response": "long",
                "tokens": len(long_log_probabilities),
                "sum_log_probability": float(long_log_probabilities.sum()),
                "mean_log_probability": float(long_log_probabilities.mean()),
            },
        ]
    )
    display(length_comparison)
    """),

    md(r"""
    ## 6 · DPO compares changes relative to a frozen reference

    Let the policy and reference each score the chosen and rejected responses.

    $$
    \Delta_\theta=
    \log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)
    $$

    $$
    \Delta_{ref}=
    \log\pi_{ref}(y_w\mid x)-\log\pi_{ref}(y_l\mid x)
    $$

    $$
    z=\beta(\Delta_\theta-\Delta_{ref})
    $$

    $$
    J_{DPO}=-\log\sigma(z)
    $$

    | Symbol | Plain-language meaning |
    |---|---|
    | $\pi_\theta$ | trainable policy, normally initialized from SFT |
    | $\pi_{ref}$ | frozen reference copied from the starting policy |
    | $\Delta_\theta$ | policy preference margin |
    | $\Delta_{ref}$ | reference preference margin |
    | $\Delta_\theta-\Delta_{ref}$ | how much the policy improved the chosen-vs-rejected margin relative to the reference |
    | $\beta$ | positive scale applied to that relative margin |
    | $z$ | scaled DPO logit passed to the sigmoid |
    | $J_{DPO}$ | loss for one labeled pair |

    The reference is an anchor, not a second trainable competitor. Freeze it, place it in
    evaluation mode, and verify that its tokenizer, template, checkpoint, and precision
    match the intended contract.
    """),

    code(r"""
    policy_chosen_logp = -2.0
    policy_rejected_logp = -3.0
    reference_chosen_logp = -2.4
    reference_rejected_logp = -2.1
    beta = 0.1

    policy_margin = policy_chosen_logp - policy_rejected_logp
    reference_margin = reference_chosen_logp - reference_rejected_logp
    relative_margin = policy_margin - reference_margin
    dpo_logit = beta * relative_margin
    dpo_probability = sigmoid(dpo_logit)
    manual_dpo_loss = -math.log(dpo_probability)

    print("policy margin:   ", policy_margin)
    print("reference margin:", reference_margin)
    print("relative margin: ", relative_margin)
    print("scaled logit:    ", dpo_logit)
    print("DPO probability: ", dpo_probability)
    print("DPO loss:        ", manual_dpo_loss)
    """),

    md(r"""
    ### Why the initial loss is `log(2)`

    When policy and reference start identical, both margins are equal. Their relative
    margin is zero, so `z = 0`, `sigmoid(0) = 0.5`, and the loss is `-log(0.5) = log(2)`.

    This does **not** mean the gradient is zero. The chosen and rejected policy scores
    depend differently on policy parameters. The loss pushes their difference upward
    while the reference values remain fixed.
    """),

    code(r"""
    policy_chosen = torch.tensor(-3.0, requires_grad=True)
    policy_rejected = torch.tensor(-2.0, requires_grad=True)
    reference_chosen = policy_chosen.detach().clone()
    reference_rejected = policy_rejected.detach().clone()
    beta_tensor = torch.tensor(0.2)

    relative_margin = (
        policy_chosen - policy_rejected
        - (reference_chosen - reference_rejected)
    )
    initial_dpo_loss = -F.logsigmoid(beta_tensor * relative_margin)
    initial_dpo_loss.backward()

    print("initial loss:", initial_dpo_loss.item())
    print("gradient for chosen log-probability:", policy_chosen.grad.item())
    print("gradient for rejected log-probability:", policy_rejected.grad.item())
    assert math.isclose(initial_dpo_loss.item(), math.log(2), rel_tol=1e-6)
    assert policy_chosen.grad < 0  # gradient descent will increase chosen log-probability
    assert policy_rejected.grad > 0  # gradient descent will decrease rejected log-probability
    """),

    md(r"""
    ## 7 · What does `beta` do?

    In the implemented loss, `beta` multiplies the relative margin before the sigmoid.
    A larger value makes the loss and gradient change more sharply around the same raw
    margin. Its interpretation is connected to the KL-regularized derivation of DPO,
    but its practical effect also depends on learning rate, data, sequence scoring, and
    implementation details.

    Do not memorize “larger beta always means closer” as a universal tuning rule. Sweep
    it and measure preference outcomes, divergence or KL diagnostics, response length,
    task retention, and independent quality metrics.

    If $m=\Delta_\theta-\Delta_{ref}$, the magnitude of the loss derivative is:

    $$
    \left|\frac{\partial J_{DPO}}{\partial m}\right|
    =\beta\,\sigma(-\beta m)
    $$

    A very negative relative margin receives a large correction, approaching `beta`.
    A very positive relative margin receives a small correction, approaching zero. This
    is useful: badly ordered pairs do not get a vanishing logistic gradient.
    """),

    code(r"""
    relative_margins = np.linspace(-5, 5, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for beta_value in [0.05, 0.2, 1.0]:
        scaled_margins = beta_value * relative_margins
        losses = np.logaddexp(0.0, -scaled_margins)
        gradient_magnitudes = beta_value / (1.0 + np.exp(scaled_margins))
        axes[0].plot(relative_margins, losses, label=f"beta={beta_value}")
        axes[1].plot(relative_margins, gradient_magnitudes, label=f"beta={beta_value}")

    for axis in axes:
        axis.axvline(0, color="black", linewidth=1, alpha=0.5)
        axis.set_xlabel("policy margin minus reference margin")
        axis.legend()
    axes[0].set_ylabel("DPO pair loss")
    axes[0].set_title("Loss")
    axes[1].set_ylabel("gradient magnitude")
    axes[1].set_title("Correction strength")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 8 · The executable DPO loop

    For every pair in a batch:

    1. render the prompt with the same versioned template;
    2. append and tokenize chosen and rejected responses separately;
    3. locate response boundaries after next-token shifting;
    4. sum response-token log probabilities under the policy;
    5. compute the same four scores under the frozen reference without gradients;
    6. form policy, reference, and relative margins;
    7. average `-logsigmoid(beta * relative_margin)`;
    8. backpropagate through the policy only;
    9. record loss, margins, token counts, gradients, and retention evidence.

    Efficient trainers concatenate or batch branches, but they must preserve the same
    scoring contract. Unit-test one tiny batch against the transparent calculation
    before trusting a distributed training run.
    """),

    code(r"""
    # Locate and import the tested tiny adaptation project from any notebook directory.
    repository = next(
        path for path in [Path.cwd(), *Path.cwd().parents]
        if (path / "projects/language_model_adaptation").exists()
    )
    sys.path[:0] = [
        str(repository / "projects/language_model_adaptation/src"),
        str(repository / "projects/tiny_language_model/src"),
    ]

    from language_model_adaptation.lab import (
        PREFERENCE_TRAIN,
        dpo_loss,
        response_log_probability,
        run_adaptation_lab,
    )

    # Run every real stage so DPO starts from an actual SFT model and frozen copy.
    experiment_report = run_adaptation_lab(seed=42)
    alignment = experiment_report["preference_alignment"]
    print("objective:", alignment["objective"])
    print("DPO loss:", alignment["dpo_loss_before"], "->", alignment["dpo_loss_after"])
    print(
        "held-out preference accuracy:",
        alignment["held_out_preference_accuracy_before"],
        "->",
        alignment["held_out_preference_accuracy_after"],
    )
    """),

    code(r"""
    held_out_evidence = pd.DataFrame(alignment["held_out_examples"])
    display(held_out_evidence)

    retention_evidence = pd.DataFrame(
        [
            {
                "measurement": "SFT validation loss",
                "before_DPO": alignment["sft_retention_loss_before"],
                "after_DPO": alignment["sft_retention_loss_after"],
                "desired_direction": "lower",
            }
        ]
    )
    display(retention_evidence)

    assert alignment["dpo_loss_after"] < alignment["dpo_loss_before"]
    assert all(row["margin_after"] > row["margin_before"] for row in alignment["held_out_examples"])
    """),

    md(r"""
    The tiny experiment changes the narrow preference successfully, while SFT validation
    loss becomes worse. That is an alignment tradeoff in this setup—not evidence that
    every DPO run has the same outcome.

    The validation pairs also reuse the same chosen and rejected response texts with new
    prompts. They test a small style transfer, not broad generalization. The correct
    conclusion is limited: **the implementation can optimize this preference signal and
    exposes a measurable retention cost**.
    """),

    md(r"""
    ## 9 · Reward-model RLHF and PPO are related, but not identical

    A common reward-model-plus-RL path is:

    ```mermaid
    flowchart LR
        A[SFT policy] --> B[Sample responses]
        B --> C[Human comparisons]
        C --> D[Train reward model]
        D --> E[Score new rollouts]
        E --> F[Estimate advantages]
        F --> G[Update policy]
        G --> B
        A --> H[Frozen reference]
        H --> G
    ```

    A high-level KL-regularized objective is:

    $$
    \max_\pi\;\mathbb{E}_{x,y\sim\pi}[r(x,y)]
    -\lambda\,D_{KL}(\pi(\cdot\mid x)\Vert\pi_{ref}(\cdot\mid x))
    $$

    This says “seek reward while penalizing movement from the reference.” It is **not
    itself PPO**. PPO is an optimization method that uses sampled trajectories,
    advantages, old-policy probability ratios, and a clipped surrogate such as:

    $$
    L^{clip}=\mathbb{E}_t\left[
    \min(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)
    \right]
    $$

    Here, $\rho_t$ is the new-to-old action probability ratio, $A_t$ is an estimated
    advantage, and $\epsilon$ is the clipping width. Practical RLHF systems add value
    estimation, KL control, rollout generation, reward normalization, and careful
    distributed training. We do not claim to execute PPO in this lesson.
    """),

    code(r"""
    probability_ratios = torch.tensor([0.6, 0.9, 1.1, 1.5])
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
    clipping_width = 0.2

    unclipped_terms = probability_ratios * advantages
    clipped_ratios = probability_ratios.clamp(1 - clipping_width, 1 + clipping_width)
    clipped_terms = clipped_ratios * advantages
    ppo_surrogate_terms = torch.minimum(unclipped_terms, clipped_terms)

    ppo_example = pd.DataFrame(
        {
            "ratio": probability_ratios.numpy(),
            "advantage": advantages.numpy(),
            "unclipped_term": unclipped_terms.numpy(),
            "clipped_term": clipped_terms.numpy(),
            "surrogate_term": ppo_surrogate_terms.numpy(),
        }
    )
    display(ppo_example)
    """),

    md(r"""
    ## 10 · Choosing among SFT, DPO, and reward-model RL

    | Method | Main data | What it directly learns | Main strength | Main risk or cost |
    |---|---|---|---|---|
    | SFT | prompt plus target response | imitate demonstrated tokens | simple, stable baseline | one target may not express tradeoffs |
    | reward model | chosen/rejected pairs | predict pairwise preferences | reusable scorer | proxy can be exploited |
    | DPO | offline pairs plus reference | increase relative chosen margin | no online rollout loop | data/reference/length sensitivity |
    | PPO-style RLHF | reward plus online rollouts | optimize sampled policy behavior | explores current policy outputs | complex and unstable system |
    | prompting | instructions/examples | no parameter update | fastest intervention | limited behavioral change |

    DPO avoids a separately deployed reward model and online policy rollouts during its
    training loop. It does not eliminate preference-model assumptions, reference choice,
    distribution shift, evaluation, or safety work.
    """),

    md(r"""
    ## 11 · Evaluation: optimized signal and independent outcomes

    Use separate training, validation, and sealed test prompts. Report:

    - pairwise accuracy, mean margin, and loss with confidence intervals;
    - per-example margins rather than only an average;
    - chosen/rejected length and margin-by-length slices;
    - agreement, ties, abstentions, rubric, and annotator slices;
    - task correctness, factuality, groundedness, format validity, and calibration;
    - safety categories evaluated independently of the training preference;
    - SFT tasks, domain language, and broad capability retention;
    - divergence or KL diagnostics, response diversity, latency, and memory;
    - several seeds and relevant baselines.

    Pairwise accuracy can be gamed by a policy that learns a shallow cue shared by the
    chosen answers. A safety-flavored dataset does not turn preference accuracy into a
    safety metric. High-stakes deployment still needs domain experts, adversarial tests,
    policy controls, monitoring, and a defined rollback path.
    """),

    md(r"""
    ## 12 · Common failure modes

    | Symptom | Likely cause | Evidence to inspect | Possible repair |
    |---|---|---|---|
    | longer answers usually lose | summed-score length effect | margin vs length difference | rebalance data; evaluate scoring policy |
    | preference gain, task loss | narrow optimization or excessive update | sealed task and retention suites | tune data, beta, LR, steps; early stop |
    | model copies one phrase everywhere | chosen responses share a shortcut | n-grams and prompt slices | diversify and counterbalance data |
    | initial loss is not near `log(2)` | policy/reference or template mismatch | initial logits and hashes | rebuild frozen reference contract |
    | reference receives gradients | missing freeze or detach | parameter gradients | freeze, no-grad, and assert |
    | chosen/rejected scores include prompt | wrong boundary after shifting | token-level score table | mask prompt positions correctly |
    | reward rises while quality falls | reward hacking | independent human/adversarial review | improve proxy and constrain optimization |
    | annotators disagree systematically | vague rubric or real value conflict | agreement by item and group | revise rubric; preserve uncertainty |
    | offline test looks perfect | duplicates or prompt-family leakage | provenance and similarity audit | group split before training |
    | unsafe output remains possible | preference objective is not a guarantee | targeted safety tests | layered policy and system controls |
    """),

    md(r"""
    ## 13 · Production and governance contract

    Version these artifacts together:

    - base and SFT checkpoint hashes;
    - policy and reference hashes;
    - tokenizer and chat template;
    - preference-data snapshot, provenance, consent, and removal process;
    - rubric, policy, annotator protocol, and disagreement rules;
    - sequence scoring, EOS, truncation, normalization, and beta settings;
    - optimizer, seed, precision, and code version;
    - held-out evaluation suite, thresholds, model card, and approval record.

    Before release, compare the candidate with the SFT baseline, run safety and retention
    gates, inspect failed and high-margin examples, test the exact inference template,
    and verify rollback. Monitor outcomes that the preference proxy could miss.
    """),

    md(r"""
    ## 14 · Student check

    1. What does one preference record claim—and what does it not claim?
    2. Why randomize candidate display order?
    3. Calculate Bradley–Terry loss when both rewards are equal.
    4. Which tokens contribute to response sequence log probability?
    5. Why can summed log probability be affected by response length?
    6. What are the policy margin and reference margin?
    7. Why is the reference frozen?
    8. Why does identical policy/reference give loss `log(2)` but a nonzero gradient?
    9. What does beta multiply in the implemented DPO loss?
    10. Why is a KL-regularized reward objective not automatically PPO?
    11. Which result in the local experiment shows a retention cost?
    12. Why does preference accuracy not prove truthfulness or safety?
    """),

    md(r"""
    ## 15 · Practice and mastery checkpoint

    **Beginner 1:** calculate sigmoid probability and loss for reward margins `2`, `0`,
    and `-2`.

    **Beginner 2:** given prompt and response token probabilities, mark which terms enter
    the sequence score and calculate their sum manually.

    **Intermediate 1:** reverse one chosen/rejected pair. Compare its initial margin,
    loss, gradient direction, and effect after a short update.

    **Intermediate 2:** sweep three beta values under three seeds. Report preference
    accuracy, margin, SFT retention, response length, and divergence diagnostics.

    **Challenge:** create a small preference dataset with randomized order, ties,
    abstentions, two reviewers per item, provenance, group splits, and a disagreement
    report. Train a candidate only after the data audit passes.

    Complete `projects/language_model_adaptation/ALIGNMENT_CHECKPOINT.md`. Pass at least
    `17/20`, then complete the practical repair and explain the result without claiming
    general safety or alignment.
    """),

    md(r"""
    ## 16 · Summary

    Preference learning adds comparisons after a model can already produce useful
    instruction-shaped responses. Bradley–Terry turns a score difference into a choice
    probability. DPO compares the policy’s chosen-vs-rejected margin with the same margin
    from a frozen reference, then optimizes their relative difference.

    The optimized preference is only one proxy. Mastery means understanding the exact
    token scores and gradients **and** knowing when the resulting evidence is too narrow
    to justify a real-world claim.

    **Memory aid:** *Compare the pair, anchor the change, and test what the label could
    not see.*
    """),
]


build("05_nlp_and_llms/08_preference_learning_and_alignment.ipynb", cells)
