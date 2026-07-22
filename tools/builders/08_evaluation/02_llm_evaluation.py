"""Builder for EVAL-02 — LLM Evaluation Foundations."""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # EVAL-02 · LLM Evaluation Foundations
    ### Decide whether a language-model change really helped

    A model can have lower training loss and still become worse for users. This lesson
    builds a small, reproducible evaluation system around the local models trained in
    NLP-03, NLP-07, and NLP-08. It moves from one evaluation question to an appropriate
    metric, examines individual failures, and adds paired uncertainty before creating a
    regression gate. No hosted API or model download is required.

    **Prerequisites:** DL-08, NLP-03, NLP-07, NLP-08, train/validation/test separation,
    cross-entropy, probability, and cosine similarity.

    **Estimated mastery time:** 8–11 hours, including the checkpoint

    **Next canonical lesson:** NLP-04 · Controlled Prompt Engineering
    """),
    md(r"""
    ## 1 · Learning Objectives

    After this lesson, you can:

    - write an evaluation contract before selecting a metric;
    - keep evaluation examples separate from training and detect obvious contamination;
    - convert held-out token loss to perplexity and explain what it does not measure;
    - calculate exact match, token F1, and ROUGE-L by hand and in Python;
    - distinguish lexical overlap, sentence-embedding similarity, and real BERTScore;
    - compare the real base, continued-pretraining, SFT/LoRA, and DPO evidence;
    - inspect paired examples, slices, and confidence intervals instead of trusting one mean;
    - separate sampling uncertainty, coverage limits, and practical significance;
    - evaluate stochastic generation with repeated, versioned decoding;
    - build a deterministic regression gate with an explicit acceptance policy.
    """),
    md(r"""
    ## 2 · Historical Motivation and Practical Problem

    **Decision:** should we replace model A with model B after changing its data,
    weights, alignment objective, or prompt?

    This is difficult because “better” depends on the task. A fluent answer can be
    wrong. A correct answer can use different wording from one reference. A lower
    perplexity can coexist with worse instruction following. An average can hide one
    dangerous slice.

    The earlier baseline was training loss. It answers whether the optimizer fitted its
    objective, not whether the system improved on untouched examples. Later, BLEU and
    ROUGE automated reference comparison; embedding metrics added semantic matching.
    None is a universal quality score. We therefore begin with the decision and data,
    then choose a small metric suite whose limitations are visible.
    """),
    md(r"""
    ## 3 · Intuition and Evaluation Contract

    Think of evaluation as a driving test. Fuel efficiency, parking accuracy, braking
    distance, and passenger comfort measure different things. Averaging them without a
    rule cannot decide whether a driver is safe. The analogy stops where model outputs
    become subjective: human rubrics and disagreement need their own design in EVAL-04.

    Write this contract before running either model:

    | Contract field | Tiny example |
    |---|---|
    | Decision | Replace the SFT policy with the DPO policy? |
    | Unit | One untouched prompt and response comparison |
    | Required behavior | Chosen response receives higher response-token probability |
    | Primary metric | Held-out preference accuracy |
    | Guardrail metric | Held-out SFT loss must not regress beyond the agreed budget |
    | Slices | Prompt type and response length |
    | Generation settings | Not applicable to this likelihood comparison |
    | Acceptance rule | Improve primary metric and stay within the retention budget |

    Use automatic metrics for repeatable narrow claims. Avoid deploying from one score,
    evaluating on training examples, or inventing a threshold after seeing the results.
    Prefer exact executable checks for structured tasks; use EVAL-04 for human judgment
    and EVAL-05 for carefully validated model-based judging.
    """),
    md(r"""
    ### Keep unlike questions in separate metric lanes

    Before calculating anything, decide which question each measurement answers.

    | Lane | Question | Examples | It does not prove |
    |---|---|---|---|
    | model fit | Does the model predict held-out tokens? | loss, perplexity | task correctness |
    | task outcome | Did the answer solve the defined task? | exact validator, F1, execution tests | broad safety |
    | behavior or policy | Did it follow the desired behavioral rule? | format validity, refusal recall | factuality |
    | preference | Which output wins under a rubric? | pairwise win rate, margin | objective truth |
    | system quality | Did the whole application work? | groundedness, citation support, retrieval success | model-only quality |
    | operations | Can we serve it reliably? | latency, cost, failures, memory | answer quality |

    Do not average these lanes into one magic score. Name one primary outcome for the
    release decision and keep non-negotiable requirements as guardrails. Compare against
    a meaningful baseline: the current production model, an unchanged prompt, a simple
    heuristic, or another predeclared reference point.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Held-out loss and perplexity

    For average held-out token loss $L$, perplexity is

    $$PP=\exp(L).$$

    **Read aloud:** perplexity equals the exponential of average negative log-likelihood.
    $PP$ and $L$ are positive scalars; $\exp$ is the exponential function. If
    $L=\ln 4\approx1.386$, then $PP=\exp(1.386)\approx4$. The model behaves as though it
    faces roughly four equally plausible next-token choices on average. Lower is better
    only on the same tokenization, text, and loss definition. It does not measure truth,
    safety, or instruction following.

    ### 4.2 Exact match and token F1

    After a declared normalization, exact match is 1 when prediction and reference are
    identical and 0 otherwise. For overlapping token counts:

    $$P=\frac{o}{m},\qquad R=\frac{o}{n},\qquad F_1=\frac{2PR}{P+R}.$$

    **Read aloud:** precision is overlap divided by predicted tokens; recall is overlap
    divided by reference tokens; F1 is their harmonic mean. $o$ is the clipped overlap
    count, $m$ is prediction length, and $n$ is reference length; all are non-negative
    integers. For prediction `orbit around star` and reference `planet orbit around star`,
    $o=3,m=3,n=4$, so $P=1$, $R=0.75$, and $F_1=0.857$. F1 gives partial lexical credit,
    but word order, meaning, and factual correctness can still be wrong.

    ### 4.3 ROUGE-L

    Let $\ell$ be the length of the longest common subsequence (LCS): matching tokens in
    the same order, with gaps allowed. Set $P_L=\ell/m$, $R_L=\ell/n$, and apply the same
    F1 formula. For `cat sat mat` versus `the cat sat on mat`, $\ell=3,m=3,n=5$;
    $P_L=1$, $R_L=0.6$, and ROUGE-L F1 is `0.75`. It helps with ordered lexical coverage,
    but a faithful paraphrase may score poorly and an incorrect copied sentence may score highly.

    ### 4.4 Paired change and bootstrap interval

    For the same $N$ examples under models A and B, let $d_i=s_{B,i}-s_{A,i}$ and
    $\bar d=N^{-1}\sum_i d_i$. Resample the **paired rows** with replacement, recompute
    $\bar d$, and take the 2.5th and 97.5th percentiles for a 95% bootstrap interval.
    Pairing keeps each prompt matched across models. The interval measures sampling
    uncertainty in this dataset; it cannot repair biased, leaked, or tiny coverage.

    | Symbol | Plain-language meaning |
    |---|---|
    | $N$ | number of paired evaluation examples |
    | $i$ | one example index |
    | $s_{A,i}$ | score from model A on example $i$ |
    | $s_{B,i}$ | score from model B on that same example |
    | $d_i$ | candidate-minus-baseline change for that pair |
    | $\bar d$ | average paired change |

    A confidence interval addresses sampling uncertainty under its assumptions. A
    release decision also needs a **minimum useful change**. With enough rows, a tiny
    improvement can be statistically clear but too small to justify cost or risk.
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Start with a four-row answer set. The code prints normalized text, token overlap,
    and both metrics so every intermediate result can be inspected.
    """),
    code(r"""
    import math
    import re
    from collections import Counter

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    def normalize_answer(text):
        lowercase = text.casefold()
        letters_numbers_spaces = re.sub(r"[^\w\s]", " ", lowercase)
        return " ".join(letters_numbers_spaces.split())


    def exact_match(prediction, reference):
        return int(normalize_answer(prediction) == normalize_answer(reference))


    def token_f1(prediction, reference):
        predicted_tokens = normalize_answer(prediction).split()
        reference_tokens = normalize_answer(reference).split()
        if not predicted_tokens or not reference_tokens:
            return float(predicted_tokens == reference_tokens)
        overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(reference_tokens)
        return 2 * precision * recall / (precision + recall)


    answer_rows = [
        ("Paris", "Paris", "same answer"),
        ("Paris, France", "Paris", "correct core plus detail"),
        ("The answer is Paris", "Paris", "verbose answer"),
        ("London", "Paris", "wrong answer"),
    ]

    for prediction, reference, note in answer_rows:
        print({
            "note": note,
            "normalized_prediction": normalize_answer(prediction),
            "normalized_reference": normalize_answer(reference),
            "exact_match": exact_match(prediction, reference),
            "token_f1": round(token_f1(prediction, reference), 3),
        })
    """),
    md(r"""
    **Expected result.** Exact match is 1 only for the identical normalized answer.
    Token F1 gives partial credit to the two answers containing `Paris`. A value outside
    `[0,1]` indicates broken metric code; a low valid value may instead expose an
    unsuitable reference or genuinely weak output.
    """),
    code(r"""
    def lcs_length(left_tokens, right_tokens):
        # Each cell stores the best subsequence length for two token prefixes.
        table = [[0] * (len(right_tokens) + 1) for _ in range(len(left_tokens) + 1)]
        for left_index, left_token in enumerate(left_tokens, start=1):
            for right_index, right_token in enumerate(right_tokens, start=1):
                if left_token == right_token:
                    table[left_index][right_index] = table[left_index - 1][right_index - 1] + 1
                else:
                    table[left_index][right_index] = max(
                        table[left_index - 1][right_index],
                        table[left_index][right_index - 1],
                    )
        return table[-1][-1], table


    def rouge_l_f1(candidate, reference):
        candidate_tokens = normalize_answer(candidate).split()
        reference_tokens = normalize_answer(reference).split()
        if not candidate_tokens or not reference_tokens:
            return float(candidate_tokens == reference_tokens)
        lcs, _ = lcs_length(candidate_tokens, reference_tokens)
        precision = lcs / len(candidate_tokens)
        recall = lcs / len(reference_tokens)
        return 0.0 if lcs == 0 else 2 * precision * recall / (precision + recall)


    candidate = "cat sat mat"
    reference = "the cat sat on mat"
    lcs, dynamic_programming_table = lcs_length(candidate.split(), reference.split())
    print("LCS length:", lcs)
    print("Last table row:", dynamic_programming_table[-1])
    print("ROUGE-L F1:", round(rouge_l_f1(candidate, reference), 3))
    assert lcs == 3 and math.isclose(rouge_l_f1(candidate, reference), 0.75)
    """),
    md(r"""
    The table stores the best subsequence length for every pair of prefixes. Equal
    tokens extend the diagonal result; unequal tokens keep the best
    result from skipping one side. The final cell must print LCS `3` and ROUGE-L `0.75`.
    This is quadratic in both sequence lengths, so production libraries use optimized
    implementations and batching where appropriate.
    """),
    md(r"""
    ### References, validators, and stochastic outputs

    A reference-based score is suitable only when the reference set represents the valid
    answer space. Short factual questions may accept several normalized strings. A JSON
    task should use schema and field validators. Code should be executed in an isolated
    test environment. Open-ended advice may require a human rubric because lexical
    similarity cannot establish correctness.

    Generation adds another source of variation. Greedy decoding is usually
    deterministic for a fixed implementation. Sampling can produce different answers
    from the same prompt, so one generation is one draw—not the model’s entire behavior.
    Freeze decoding parameters and software, then generate multiple samples or repeat
    the full evaluation under several seeds. Report the distribution and failure rate.
    A seed improves reproducibility but does not guarantee identical results across all
    hardware, kernels, and library versions.
    """),
    code(r"""
    candidate_responses = np.array(["correct", "incomplete", "incorrect"])
    candidate_probabilities = np.array([0.65, 0.25, 0.10])

    def sampled_accuracy(seed, draws=20):
        # This mimics repeated generation from one fixed categorical distribution.
        random_generator = np.random.default_rng(seed)
        samples = random_generator.choice(
            candidate_responses,
            size=draws,
            p=candidate_probabilities,
        )
        return samples, float(np.mean(samples == "correct"))


    for seed in [3, 17, 91]:
        samples, accuracy = sampled_accuracy(seed)
        print(
            f"seed={seed:>2}: correct={np.sum(samples == 'correct'):>2}/20, "
            f"sample accuracy={accuracy:.2f}"
        )

    print("Expected long-run accuracy:", candidate_probabilities[0])
    """),
    md(r"""
    The three measured accuracies can differ even though the underlying candidate did
    not change. More samples reduce Monte Carlo noise, but they do not add missing prompt
    categories. Repeat outputs and broader evaluation coverage solve different problems.
    """),
    md(r"""
    ### Real local model evidence

    The next cell reruns the actual adaptation laboratory. These are measured losses and
    preference margins from trained PyTorch models—not fabricated outputs, API responses,
    or a lexical proxy renamed as model quality.
    """),
    code(r"""
    import sys
    from pathlib import Path

    candidate_roots = [Path.cwd(), *Path.cwd().parents]
    repo_root = next(path for path in candidate_roots if (path / "projects/language_model_adaptation").exists())
    sys.path[:0] = [
        str(repo_root / "projects/language_model_adaptation/src"),
        str(repo_root / "projects/tiny_language_model/src"),
    ]

    from language_model_adaptation.lab import run_adaptation_lab

    adaptation_report = run_adaptation_lab(seed=42)
    continued = adaptation_report["continued_pretraining"]
    tuning = adaptation_report["instruction_tuning"]
    alignment = adaptation_report["preference_alignment"]

    measured_rows = [
        ("Base → continued", "domain perplexity", math.exp(continued["domain_loss_before"]), math.exp(continued["domain_loss_after"]), "lower"),
        ("Base → continued", "base-retention perplexity", math.exp(continued["base_retention_loss_before"]), math.exp(continued["base_retention_loss_after"]), "lower"),
        ("Full SFT → LoRA", "held-out SFT loss", tuning["full"]["held_out_loss"], tuning["lora"]["held_out_loss"], "lower"),
        ("SFT → DPO", "preference accuracy", alignment["held_out_preference_accuracy_before"], alignment["held_out_preference_accuracy_after"], "higher"),
        ("SFT → DPO", "SFT-retention loss", alignment["sft_retention_loss_before"], alignment["sft_retention_loss_after"], "lower"),
    ]

    print(f'{"comparison":20s} {"metric":28s} {"before":>10s} {"after":>10s}  direction')
    for comparison, metric, before, after, direction in measured_rows:
        print(f"{comparison:20s} {metric:28s} {before:10.3f} {after:10.3f}  {direction}")
    """),
    md(r"""
    **Interpretation.** Continued pretraining improves domain perplexity but harms base
    retention. LoRA has lower held-out instruction loss than full SFT in this tiny run.
    DPO improves the narrow held-out preference signal but worsens SFT retention. There
    is no single winner until the evaluation contract assigns a primary outcome and a
    tolerated regression. Every comparison must use the same held-out examples and metric.
    """),
    code(r"""
    def paired_bootstrap_delta(before_scores, after_scores, draws=4000, seed=42):
        before = np.asarray(before_scores, dtype=float)
        after = np.asarray(after_scores, dtype=float)
        if before.shape != after.shape or before.ndim != 1 or len(before) == 0:
            raise ValueError("before and after must be non-empty paired one-dimensional arrays")
        # Resample row indices once and apply them to both systems to preserve pairing.
        random_generator = np.random.default_rng(seed)
        sampled_indices = random_generator.integers(0, len(before), size=(draws, len(before)))
        sampled_deltas = (after[sampled_indices] - before[sampled_indices]).mean(axis=1)
        return {
            "observed_delta": float((after - before).mean()),
            "ci_low": float(np.quantile(sampled_deltas, 0.025)),
            "ci_high": float(np.quantile(sampled_deltas, 0.975)),
        }


    held_out = alignment["held_out_examples"]
    accuracy_before = [float(row["correct_before"]) for row in held_out]
    accuracy_after = [float(row["correct_after"]) for row in held_out]
    interval = paired_bootstrap_delta(accuracy_before, accuracy_after)

    for row in held_out:
        print({
            "prompt": row["prompt"],
            "margin_before": round(row["margin_before"], 3),
            "margin_after": round(row["margin_after"], 3),
            "correct_before": row["correct_before"],
            "correct_after": row["correct_after"],
        })
    print("Paired accuracy delta and interval:", interval)
    print("Held-out examples:", len(held_out), "— far too narrow for a deployment claim.")
    """),
    md(r"""
    The interval may look decisive because both tiny held-out examples move in the same
    direction. This is a **mechanical demonstration**, not evidence of broad alignment.
    Bootstrap resampling cannot invent missing prompt types, independent labels, safety
    coverage, or a larger population.
    """),
    md(r"""
    ## 6 · Visualization and Slice Inspection
    """),
    code(r"""
    labels = ["domain PPL", "base-retention PPL", "preference accuracy", "SFT-retention loss"]
    before_values = [
        math.exp(continued["domain_loss_before"]),
        math.exp(continued["base_retention_loss_before"]),
        alignment["held_out_preference_accuracy_before"],
        alignment["sft_retention_loss_before"],
    ]
    after_values = [
        math.exp(continued["domain_loss_after"]),
        math.exp(continued["base_retention_loss_after"]),
        alignment["held_out_preference_accuracy_after"],
        alignment["sft_retention_loss_after"],
    ]

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for axis, label, before, after in zip(axes, labels, before_values, after_values):
        axis.bar(["before", "after"], [before, after], color=["#8da0cb", "#66c2a5"])
        axis.set_title(label, fontsize=9)
        axis.bar_label(axis.containers[0], fmt="%.2f", fontsize=8)
    figure.suptitle("Real local measurements: improvement and regression coexist")
    figure.tight_layout()
    plt.show()
    """),
    md(r"""
    Each panel keeps its own scale because these metrics have different units and must
    not be averaged. Always inspect per-example rows and meaningful slices after the
    dashboard. A global mean can improve while one language, prompt type, response
    length, or risk category regresses.
    """),
    code(r"""
    # Ninety-nine routine cases improve slightly, while one critical case fails.
    slice_evidence = pd.DataFrame(
        [
            {"slice": "routine", "count": 99, "before": 0.70, "after": 0.75},
            {"slice": "critical", "count": 1, "before": 1.00, "after": 0.00},
        ]
    )
    slice_evidence["delta"] = slice_evidence["after"] - slice_evidence["before"]
    weighted_before = np.average(slice_evidence["before"], weights=slice_evidence["count"])
    weighted_after = np.average(slice_evidence["after"], weights=slice_evidence["count"])

    display(slice_evidence)
    print("global score:", round(weighted_before, 3), "->", round(weighted_after, 3))
    print("critical slice passes:", bool(slice_evidence.loc[1, "after"] == 1.0))
    assert weighted_after > weighted_before
    assert slice_evidence.loc[1, "delta"] < 0
    """),
    md(r"""
    The global score improves because routine rows dominate the average, yet the only
    critical example regresses. The right response is not to hide the critical result
    inside a weighted mean. Declare critical slices before testing, require enough rows
    to evaluate them, and make their thresholds release guardrails.
    """),
    md(r"""
    ## 7 · Failure patterns worth recognizing

    | Symptom | Likely cause | Inspect | Scoped repair |
    |---|---|---|---|
    | Excellent evaluation score | train/evaluation overlap | hashes, duplicate entities, templates | rebuild the split before tuning |
    | Paraphrase scores near zero | lexical metric used for semantic judgment | candidate/reference tokens | add human review or a validated semantic metric |
    | “Semantic” score changes across runs | random/hash vectors mislabeled as embeddings | encoder and seed | use a trained pinned encoder; name proxies honestly |
    | Lower perplexity, worse answers | fluency objective differs from task | task rows and retention slices | retain perplexity as diagnostic, add task metrics |
    | Mean improves, critical slice falls | aggregation hides heterogeneity | per-slice counts and deltas | set slice-specific guardrails |
    | CI is narrow on weak data | examples are duplicated or unrepresentative | unique prompts and collection process | improve coverage; uncertainty cannot fix bias |
    | Regression gate flips randomly | sampling or versions uncontrolled | seed, decoding, weights, prompt, code | version the full evaluation manifest |

    Common mistakes: selecting metrics after seeing results, comparing perplexity across
    tokenizers, treating F1 as factuality, using one reference for open-ended writing,
    reporting only the mean, and calling preference accuracy “alignment.”
    """),
    md(r"""
    ## 8 · Library and Tool Implementation

    Production libraries reduce implementation mistakes but do not choose the right
    contract. `evaluate`, `rouge-score`, and `bert-score` can compute established
    metrics when installed and version-pinned. **Real BERTScore** performs contextual
    token matching with a pretrained encoder; it is not random hash cosine and may
    require a model download. This offline core therefore explains it but does not
    pretend to execute it.

    Conceptually, BERTScore encodes candidate and reference tokens in context, computes
    cosine similarity between token vectors, matches every candidate token to its most
    similar reference token for precision, and reverses the direction for recall. Their
    harmonic mean gives an F1-like score; optional inverse-document-frequency weights
    give rarer tokens more influence. Encoder checkpoint, layer, language, baseline
    rescaling, and library version are part of the metric contract. Semantic similarity
    still does not prove that a claim is factually correct.

    NLP-02 provides a real locally trained sentence bi-encoder and held-out retrieval
    evaluation. Its cosine score is useful for its validated retrieval task, but it is
    still **not BERTScore** and is not automatic proof that a generated answer is correct.
    """),
    code(r"""
    sentence_project = repo_root / "projects" / "sentence_embeddings"
    family_project = repo_root / "projects" / "transformer_families"
    sys.path[:0] = [str(sentence_project / "src"), str(family_project / "src")]

    from sentence_embeddings.training import run_experiment

    embedding_report = run_experiment(seed=42, steps=320)
    print("TF-IDF held-out MRR:", round(embedding_report["tfidf_baseline"]["mrr"], 3))
    print("Untrained encoder MRR:", round(embedding_report["untrained_transformer"]["mrr"], 3))
    print("Trained local encoder MRR:", round(embedding_report["trained_contrastive_encoder"]["mrr"], 3))
    print("Metric name: held-out retrieval MRR — not BERTScore and not answer correctness.")
    """),
    md(r"""
    ## 9 · Realistic Case Study — A Prompt-Change Regression Gate

    A support team wants to change a prompt after EVAL-02. Before editing it, the team
    freezes an evaluation manifest containing dataset hash, prompt version, model
    checkpoint, tokenizer, decoding settings, metric code version, seed, and acceptance
    rule. The suite has answerable, ambiguous, adversarial, short, and long prompts.

    The primary metric is task correctness. Guardrails cover refusal behavior, critical
    slices, latency, and cost. The new prompt is accepted only when its paired primary
    change meets the declared rule and no critical guardrail fails. Borderline cases go
    to the human protocol taught in EVAL-04. This process supports one release decision;
    it does not prove universal model quality.
    """),
    code(r"""
    def regression_decision(
        primary_delta,
        primary_ci_low,
        minimum_useful_delta,
        guardrail_deltas,
        limits,
    ):
        # Require both statistical direction and enough improvement to matter.
        primary_passes = (
            primary_ci_low >= 0
            and primary_delta >= minimum_useful_delta
        )
        # In this contract, every guardrail delta is defined so positive means worse.
        failed_guardrails = {
            name: change
            for name, change in guardrail_deltas.items()
            if change > limits[name]
        }
        return {
            "decision": "accept" if primary_passes and not failed_guardrails else "review_or_reject",
            "primary_passes": primary_passes,
            "failed_guardrails": failed_guardrails,
        }


    # Limits are declared before evaluating the candidate. Positive loss delta is worse.
    retention_change = alignment["sft_retention_loss_after"] - alignment["sft_retention_loss_before"]
    decision = regression_decision(
        interval["observed_delta"],
        interval["ci_low"],
        0.10,
        {"sft_retention_loss": retention_change},
        {"sft_retention_loss": 0.25},
    )
    print("Retention loss change:", round(retention_change, 3))
    print("Decision:", decision)
    assert decision["decision"] == "review_or_reject"
    """),
    md(r"""
    **Expected result:** the narrow preference metric improves, but the declared retention
    budget fails, so the system prints `review_or_reject`. A gate is not “model B has a
    higher average.” It is an explicit policy for resolving practical improvement,
    sampling uncertainty, and multiple measured outcomes.
    """),
    md(r"""
    ## 10 · Learning and Production Considerations

    - Freeze the test set; use a development set while iterating.
    - Remove exact duplicates and audit semantic/template contamination separately.
    - Version examples, labels, rubrics, model weights, tokenizer, prompt, decoding, and metric code.
    - Use deterministic likelihood metrics where possible; otherwise repeat stochastic generation.
    - Report counts, distributions, paired deltas, uncertainty, failures, and slices—not only means.
    - Calibrate thresholds on historical decisions before release; do not copy universal numbers.
    - Protect evaluation data containing personal, licensed, or security-sensitive information.
    - Separate this teaching-scale implementation from a production service with access control,
      audit logs, resilient execution, review ownership, and incident response.
    """),
    md(r"""
    ## 11 · Tradeoff and Related-Concept Analysis

    | Concept | Main purpose | Strength | Weakness | Use when |
    |---|---|---|---|---|
    | Loss / perplexity | next-token prediction fit | reference-free and cheap | not task correctness | same tokenizer and held-out text |
    | Exact match | strict structured correctness | transparent | rejects valid variants | canonical short answers |
    | Token F1 | partial answer overlap | simple partial credit | ignores order and meaning | extractive QA |
    | ROUGE-L | ordered lexical coverage | interpretable | weak on paraphrases/factuality | reference summaries with caveats |
    | BLEU | corpus n-gram precision | reproducible historical baseline | weak sentence semantics | comparable translation experiments |
    | Sentence-embedding cosine | whole-text vector similarity | handles some paraphrases | encoder/domain dependent | encoder validated for the task |
    | BERTScore | contextual token similarity | softer token matching | model/download/version dependent | references exist and encoder is suitable |
    | Human evaluation | rubric-based judgment | captures nuanced outcomes | costly and variable | automated metrics miss the decision |
    | LLM-as-judge | scalable rubric proxy | flexible | bias, leakage, instability | validated against humans first |

    pass@k belongs in a code-generation extension after students understand sampling,
    executable tests, and combination notation. Benchmark catalogs belong after this
    lesson’s core decision process; a benchmark name is not an evaluation design.
    """),
    md(r"""
    ## 12 · Readiness Check

    You are ready for NLP-04 Prompt Engineering when you can:

    1. write the evaluation contract before changing a prompt;
    2. explain why the local DPO policy both improves and regresses;
    3. calculate token F1 and ROUGE-L for a tiny example;
    4. state why hash vectors cannot demonstrate BERTScore;
    5. produce paired results with a coverage warning;
    6. apply a predeclared regression gate and defend its guardrails.

    Running every cell is insufficient. Score at least `17/20` on
    `projects/language_model_adaptation/EVALUATION_CHECKPOINT.md`, earn non-zero points
    in every category, and complete the independent exercise with a fresh held-out set.
    """),
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. What decision does an evaluation contract make explicit?
    2. Why can perplexity only be compared under the same tokenization and data contract?
    3. Calculate precision, recall, and F1 when overlap is 2, prediction length 4, reference length 5.
    4. How does ROUGE-L differ from token F1?
    5. Why is a random hash embedding not a semantic metric?
    6. Why must a model comparison be paired by prompt?
    7. What can a bootstrap interval establish, and what can it never repair?
    8. Why did the example gate reject the DPO policy despite improved preference accuracy?
    9. Why should sampled generation be repeated under fixed decoding settings?
    10. How can a statistically clear improvement still fail a release gate?
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked example (15 min).** Prediction `orbit around star`, reference
    `planet orbit around star`: overlap `3`, precision `1`, recall `0.75`, F1 `0.857`.
    Common mistake: counting duplicate tokens without clipping to reference counts.

    **Guided practice (25 min).** Compare `cat mat sat` with `the cat sat on mat` using
    token F1 and ROUGE-L. Hint: token F1 ignores order; LCS does not. **Self-check:**
    token F1 must exceed ROUGE-L. Solution: token overlap is 3, so F1 is `0.75`; LCS is
    2, so ROUGE-L is `0.50`.

    **Independent practice (45 min).** Create six untouched prompt/reference/prediction
    rows. Compute exact match, token F1, and ROUGE-L per row, then report the worst row
    and one slice. **Expected evidence:** assertions keep every score in `[0,1]`, and the
    written decision explains why one metric is primary. Rubric: data separation 2,
    correct metrics 3, row/slice diagnosis 3, limitation 2.

    **Intermediate exercise (45 min).** Add one regression and two unchanged rows to a
    paired comparison. Recompute the bootstrap interval under three seeds. Explain why
    the observed delta stays fixed while Monte Carlo percentiles can move slightly.

    **Challenge mini-project (90 min).** Evaluate two prompt versions without an API.
    Use a deterministic local function or stored outputs, freeze a manifest, declare one
    primary metric and two guardrails, inspect slices, apply a gate, and write a one-page
    decision. Expected output: reproducible code, per-row results, uncertainty, rejected
    examples, and a scoped conclusion—not “model B is universally better.”

    **Summary:** evaluation connects a declared decision to untouched examples, suitable
    metrics, paired evidence, uncertainty, failure analysis, and a predeclared gate.
    **Memory aid:** *Define the decision, freeze the data, inspect pairs, then trust no
    metric beyond the behavior it actually measures.*
    """),
]


build("08_evaluation/02_llm_evaluation.ipynb", cells)
