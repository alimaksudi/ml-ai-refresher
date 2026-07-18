"""Builder for NLP-04 — Controlled Prompt Engineering."""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # NLP-04 · Controlled Prompt Engineering
    ### Change context without changing weights—and measure the result

    Prompt engineering is not a collection of magic phrases. It is controlled
    experimentation on the tokens that condition a frozen model. This lesson trains one
    tiny local instruction model, freezes it, versions three prompt candidates, selects
    one on development data, and opens the final test once. The candidate looks better
    but uncertainty remains large, so the regression gate refuses an unsupported claim.

    **Prerequisites:** NLP-08 and EVAL-02. You must already understand token
    conditioning, SFT formatting, train/development/test separation, paired evaluation,
    and regression gates. No API key or model download is required. **Core time:** 5–8 hours.
    """),
    md(r"""
    ## 1 · Learning Objectives

    After this lesson, you can:

    - explain why prompt tokens change output probabilities without changing weights;
    - separate instructions, trusted context, untrusted input, examples, and output contracts;
    - create stable prompt versions and hashes;
    - freeze model, tokenizer, decoding, seed, data, and metrics before comparison;
    - establish a minimal baseline and change one prompt component at a time;
    - use few-shot examples without leaking final evaluation labels;
    - validate structured syntax separately from semantic correctness;
    - select on development data and evaluate once on paired final-test cases;
    - test prompt paraphrases and reject unnecessary prompt complexity;
    - decide when deterministic code, retrieval, or SFT is more appropriate than prompting.
    """),
    md(r"""
    ## 2 · Historical Motivation and Practical Problem

    **Decision:** should a sentiment service replace its minimal prompt with a clearer
    instruction or a longer few-shot prompt?

    Early language models were prompted as text completions. Larger pretrained models
    made zero-shot and few-shot task specification practical, and later instruction
    tuning made conversational instructions more reliable. The engineering problem did
    not disappear: prompt behavior still depends on model, training distribution,
    tokenizer, chat template, decoding, and evaluation cases.

    Trying three prompts on three hand-picked examples is tempting but unreliable. The
    examples can be selected after seeing results, random decoding can confound the
    comparison, and the final test can quietly become training data. EVAL-02 supplied the
    missing baseline: declare the decision, freeze the evidence path, inspect paired
    failures, and apply a predeclared gate.
    """),
    md(r"""
    ## 3 · Intuition, Prompt Anatomy, and Analogy

    A prompt is the model's visible working context. Changing it is like giving the same
    employee a revised task sheet: the employee is unchanged, but the available
    instructions and examples differ. The analogy stops at agency—the model predicts
    tokens from learned patterns and does not independently understand organizational intent.

    | Component | Purpose | Example | Trust rule |
    |---|---|---|---|
    | Instruction | state the task | classify sentiment | authored and versioned |
    | Trusted context | supply approved facts/rules | allowed labels | verified source |
    | Untrusted input | data to process | customer review | treat as data, not authority |
    | Examples | demonstrate mapping/format | `good → positive` | development only; audit leakage |
    | Output contract | define machine-readable result | one enum label | validate after generation |

    Start with the shortest prompt that specifies the task. Add one component only when
    a named failure justifies it. Few-shot examples are useful for formats and edge cases,
    but irrelevant or leaking examples can hurt. Detailed prompt-injection threat models
    belong in NLP-05; ReAct and search trees belong in the agent phase.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Conditioning with fixed weights

    $$p_\theta(y_t\mid x,y_{<t})=\operatorname{softmax}(f_\theta(x,y_{<t})).$$

    **Read aloud:** the probability of next output token $y_t$ depends on prompt tokens
    $x$, earlier output tokens $y_{<t}$, and fixed parameters $\theta$. $f_\theta$ returns
    one real-valued logit per vocabulary token; softmax converts logits to probabilities
    summing to 1. If two candidate-token logits are `[2, 1]`, their probabilities are
    approximately `[0.731, 0.269]`. Changing $x$ can change those numbers even when
    $\theta$ never changes. This describes token conditioning, not correctness or obedience.

    ### 4.2 One-variable prompt comparison

    For paired case score $s_{v,i}$ under prompt version $v$, candidate change is

    $$d_i=s_{candidate,i}-s_{baseline,i},\qquad
      \bar d=\frac{1}{N}\sum_{i=1}^{N}d_i.$$

    **Read aloud:** subtract baseline from candidate on each of $N$ identical cases,
    then average those paired differences. Scores are scalars such as 0/1 correctness.
    If differences are `[0,1,0,-1]`, the mean is `0`; two changed outcomes cancel.
    Use the paired bootstrap from EVAL-02 for uncertainty. This estimate is invalid when
    models, decoding, or cases also change.

    ### 4.3 Context budget

    $$T_{available}=C-T_{template}-T_{input}-T_{examples}-T_{tools}.$$

    **Read aloud:** available generation tokens equal model context capacity minus every
    encoded input component. All $T$ values and $C$ are non-negative token counts under
    the **actual tokenizer and chat template**. For capacity `192` and components
    `40, 30, 70, 0`, only `52` tokens remain. A character-count estimate is not a token
    count. Reserve output space before sending; truncation can remove instructions or evidence.
    """),
    md(r"""
    ## 5 · Manual Implementation and Real Local Experiment

    First build stable prompt specifications. A production renderer should reject missing
    variables, preserve untrusted input as data, and hash the full definition. The project
    uses one `{review}` field and records the exact template plus SHA-256 hash.
    """),
    code(r"""
    import hashlib
    import json
    import string


    class PromptSpec:
        def __init__(self, version, template, declared_change):
            self.version = version
            self.template = template
            self.declared_change = declared_change
            fields = [name for _, name, _, _ in string.Formatter().parse(template) if name]
            if fields != ["review"]:
                raise ValueError("template must contain exactly one {review} field")

        def render(self, review):
            if not isinstance(review, str) or not review.strip():
                raise ValueError("review must be non-empty text")
            return self.template.format(review=review)

        def sha256(self):
            payload = json.dumps(
                {"version": self.version, "template": self.template, "change": self.declared_change},
                sort_keys=True,
            )
            return hashlib.sha256(payload.encode()).hexdigest()


    baseline_spec = PromptSpec("v0", "review: {review}\nlabel:", "minimal task cue")
    print(baseline_spec.render("good service"))
    print("Prompt hash:", baseline_spec.sha256())
    """),
    md(r"""
    **Expected result:** rendering retains the review exactly and the hash is a stable
    64-character identifier. A missing or additional placeholder is a template-contract
    failure. A valid rendered prompt can still perform poorly; that is an experimental outcome.

    The next cell runs the real local project. One tiny Transformer is trained on authored
    sentiment data and all three prompt formats, then frozen. Development cases select a
    non-baseline challenger. Only the baseline and selected challenger see final-test data.
    Greedy decoding removes sampling as a confounder.
    """),
    code(r"""
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    roots = [Path.cwd(), *Path.cwd().parents]
    repo_root = next(path for path in roots if (path / "projects/prompt_evaluation").exists())
    sys.path[:0] = [
        str(repo_root / "projects/prompt_evaluation/src"),
        str(repo_root / "projects/language_model_adaptation/src"),
        str(repo_root / "projects/tiny_language_model/src"),
    ]

    from prompt_evaluation.lab import run_prompt_lab

    report = run_prompt_lab(seed=42)
    contract = report["experiment_contract"]
    print(json.dumps(contract, indent=2))
    print("Data:", report["data"])
    print("Selected challenger:", report["selected_on_development"])
    """),
    md(r"""
    ### Prompt versions: one declared change at a time

    - `v0_baseline`: minimal `review → label` cue.
    - `v1_explicit_contract`: adds the task and allowed labels.
    - `v2_few_shot`: retains v1 and adds two short demonstrations.

    All three formats appear during toy-model training. Therefore this experiment shows
    controlled prompt comparison for one known model, **not** broad zero-shot or
    in-context-learning capability. Few-shot labels come from training/development design;
    no final-test review or answer appears in the prompt bank.
    """),
    code(r"""
    for version, definition in report["prompt_versions"].items():
        development = report["development"][version]
        print({
            "version": version,
            "declared_change": definition["change"],
            "hash_prefix": definition["sha256"][:12],
            "development_accuracy": round(development["accuracy"], 3),
            "schema_valid_rate": round(development["schema_valid_rate"], 3),
            "mean_response_loss": round(development["mean_response_loss"], 3),
        })
    """),
    md(r"""
    **Expected result:** the explicit-contract prompt wins the predeclared development
    rule; adding few-shot examples does not automatically help. This is a local measured
    outcome, not a ranking of techniques. Inspect examples before trying another prompt.
    """),
    code(r"""
    selected = report["selected_on_development"]
    final_results = report["test_opened_after_selection"]
    for version, evaluation in final_results.items():
        print("\n", version, {
            "accuracy": round(evaluation["accuracy"], 3),
            "schema_valid_rate": round(evaluation["schema_valid_rate"], 3),
            "mean_response_loss": round(evaluation["mean_response_loss"], 3),
        })
        for row in evaluation["examples"]:
            print(row["review"], "->", repr(row["raw_output"]), "expected", row["expected"])

    print("Paired accuracy:", report["paired_test_accuracy"])
    print("Release decision:", report["release_decision"])
    """),
    md(r"""
    The challenger improves final accuracy from about `0.33` to `0.50` and schema
    validity from `0.50` to `0.83`, but the six-case paired interval crosses zero.
    The correct release decision is `review_or_reject`, not “the explicit prompt is
    better.” Collect more representative cases or keep the simpler baseline.
    """),
    md(r"""
    ## 6 · Visualization and Conditioning Evidence
    """),
    code(r"""
    versions = list(report["development"])
    development_accuracy = [report["development"][version]["accuracy"] for version in versions]
    development_schema = [report["development"][version]["schema_valid_rate"] for version in versions]
    x_positions = np.arange(len(versions))

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(x_positions - 0.18, development_accuracy, 0.36, label="accuracy")
    axis.bar(x_positions + 0.18, development_schema, 0.36, label="schema validity")
    axis.set_xticks(x_positions, versions, rotation=12)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("development proportion")
    axis.set_title("Measured local prompt comparison—not simulated technique accuracy")
    axis.legend()
    plt.tight_layout()
    plt.show()
    """),
    code(r"""
    conditioning = report["next_token_conditioning"]
    print("Review:", conditioning["review"])
    print("Baseline next-token distribution:", conditioning["baseline_top_tokens"])
    print("Selected next-token distribution:", conditioning["selected_top_tokens"])
    print("Total variation distance:", round(conditioning["total_variation_distance"], 4))
    """),
    md(r"""
    **Meaning:** prompt versions change the real next-token distribution while model
    weights stay fixed. Total variation distance is half the sum of absolute probability
    differences and lies in `[0,1]`. A non-zero value proves conditioning changed; it does
    not prove the change is useful.
    """),
    code(r"""
    print("Robustness on development paraphrases:")
    for name, evaluation in report["robustness_on_development"].items():
        print(name, {
            "accuracy": round(evaluation["accuracy"], 3),
            "schema_validity": round(evaluation["schema_valid_rate"], 3),
            "loss": round(evaluation["mean_response_loss"], 3),
        })
    """),
    md(r"""
    Prompt paraphrases with the same intended task can behave differently. Do not pick
    the best paraphrase on the final test. Use development data for prompt iteration,
    record every attempted version, and reserve a final test for one decision.
    """),
    md(r"""
    ## 7 · Failure Modes, Beginner Mistakes, and Debugging

    | Symptom | Likely cause | Inspect | Scoped repair |
    |---|---|---|---|
    | candidate “wins” only on chosen examples | selection leakage | prompt/eval history | rebuild an untouched final test |
    | output changes between repeats | decoding not fixed | temperature, seed, model mode | freeze settings or repeat intentionally |
    | few-shot loss is `NaN` | prompt consumed response window | encoded boundaries and labels | shorten examples or expand supported context |
    | valid label but wrong sentiment | syntax confused with semantics | per-case expected/output | keep correctness as separate metric |
    | good mean, bad subgroup | aggregate hides slice | paired rows by slice | add slice guardrails and data |
    | tiny wording change causes failure | prompt brittleness | paraphrase results | simplify, add representative training, or reject |
    | delimiter treated as security boundary | untrusted text still influences model | instruction/data flow and tool permissions | apply NLP-05 threat model and least privilege |

    Common mistakes: changing prompt and decoding together, tuning on final test,
    assuming longer prompts are better, using irrelevant examples, trusting model-stated
    confidence, counting characters instead of tokens, and logging sensitive prompts.
    """),
    md(r"""
    ## 8 · Structured Output and Tooling

    Structured output has two independent checks:

    1. **Syntactic/schema validity:** can software parse it and is the value allowed?
    2. **Semantic correctness:** is the parsed value correct for this case?

    The local lab uses the enum `{positive, negative, neutral}`. Actual outputs such as
    `neutive` fail schema validation; `negative` for a positive review passes schema but
    fails correctness. JSON Schema, Pydantic, or provider-constrained decoding can enforce
    richer syntax, but none guarantees factual or task correctness.
    """),
    code(r"""
    allowed_labels = {"positive", "negative", "neutral"}


    def validate_enum_output(raw_output):
        normalized = raw_output.casefold().strip()
        if normalized not in allowed_labels:
            return {"valid": False, "value": None, "error": "not an allowed label"}
        return {"valid": True, "value": normalized, "error": None}


    actual_outputs = final_results[selected]["examples"]
    for row in actual_outputs:
        print(row["review"], validate_enum_output(row["raw_output"]), "correct=", row["correct"])
    """),
    md(r"""
    Use the model's documented tokenizer and chat template rather than inventing
    `[SYSTEM]` text markers. A framework may help render messages or validate schemas,
    but keep prompt specifications and evaluation contracts independent of one vendor.
    Hosted APIs are optional extensions and must pin the model revision and settings.
    """),
    md(r"""
    ## 9 · Realistic Case Study — Support Ticket Routing

    A support system routes short tickets to `billing`, `technical`, or `account`. The
    baseline prompt omits allowed labels and sometimes emits prose. Engineers freeze a
    representative evaluation set, model revision, tokenizer, greedy decoding, and enum
    validator. They add allowed labels as one change, then separately test two examples.

    Development data selects one challenger. The final paired report includes accuracy,
    schema validity, each changed case, product/language slices, latency, and token count.
    The release gate requires non-negative paired evidence and no critical-slice regression.
    Human review handles ambiguous tickets. No universal accuracy, cost, or threshold is
    assumed; the organization calibrates those values from its own risk and history.
    """),
    md(r"""
    ## 10 · Learning and Production Considerations

    - Store prompt text, version, hash, owner, intended change, and rollback version.
    - Store model/tokenizer/chat-template revision and decoding configuration beside results.
    - Separate development iteration from one-time final testing.
    - Count tokens with the actual tokenizer, including role markers and tool schemas.
    - Redact or control access to prompts containing personal, proprietary, or secret data.
    - Treat retrieved documents and user text as untrusted data, not higher-priority instructions.
    - Validate tool arguments and authorize each action outside the language model.
    - Track latency and cost from measured deployment traces, not copied universal numbers.
    - Roll back when guardrails fail; do not edit thresholds after seeing a regression.
    """),
    md(r"""
    ## 11 · Tradeoff and Related-Concept Analysis

    | Approach | Main purpose | Strength | Weakness | Prefer when |
    |---|---|---|---|---|
    | Minimal prompt | establish baseline | cheap and interpretable | underspecified tasks fail | task/model already clear |
    | Explicit contract | state labels/format/constraints | easy controlled change | may still be ignored | failure is ambiguity or format |
    | Few-shot examples | demonstrate edge cases | concrete behavior signal | token cost and leakage risk | examples represent real failures |
    | Deterministic code | enforce known rules | reliable and testable | cannot handle broad language | behavior is fully specified |
    | Retrieval | supply changing evidence | current/domain context | retrieval can fail or inject noise | missing knowledge is the problem |
    | SFT/LoRA | change model behavior | shorter inference prompts | data/training/evaluation cost | repeated behavior is not prompt-fixable |
    | Structured validator | enforce output interface | catches invalid syntax | cannot prove meaning | downstream software needs a contract |

    Reasoning scaffolds should expose verifiable intermediate results only when the task
    benefits. More generated reasoning is not automatically faithful. Self-consistency is
    an optional extension for discrete, independently checkable answers after sampling and
    cost are understood. ReAct is taught with tool authorization in AGT-02.
    """),
    md(r"""
    ## 12 · Readiness Check

    You are ready for NLP-05 when you can:

    1. explain prompt conditioning without claiming weights changed;
    2. identify all five prompt components and their trust boundaries;
    3. reproduce prompt hashes and the frozen experiment contract;
    4. explain why development selects and final test confirms only once;
    5. distinguish enum validity from sentiment correctness;
    6. diagnose the few-shot context-window failure;
    7. defend `review_or_reject` despite a higher candidate point estimate;
    8. choose prompting, code, retrieval, or SFT for a named failure.
    """),
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. How can prompt tokens change next-token probabilities when parameters are frozen?
    2. What belongs in instruction, trusted context, untrusted input, examples, and output contract?
    3. Why must prompt versions use identical model and decoding settings?
    4. Why should the final test remain closed during prompt iteration?
    5. Give one few-shot leakage failure and one context-budget failure.
    6. Why can a schema-valid output still be wrong?
    7. What does the paired interval say about this lesson's challenger?
    8. When should deterministic code, retrieval, or SFT replace a longer prompt?
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, Solutions, and Mini-Project

    **Worked example (15 min).** With baseline scores `[1,0,1,0]` and candidate
    `[1,1,0,0]`, paired differences are `[0,1,-1,0]` and mean change is `0`. The
    candidate has one win and one regression. Common mistake: comparing only averages
    and failing to inspect changed rows.

    **Guided practice (25 min).** Add one `neutral` example to v1 without changing its
    instruction. Hint: make a new version and hash; never overwrite v1. Predict token
    count and which rows might change. **Self-check:** model, decoding, and evaluation
    hashes remain identical while only the prompt hash changes.

    **Independent practice (45 min).** Add six untouched development reviews with a
    `negation` slice. Compare v0 and v1 per case. Expected evidence: zero exact overlap,
    scores in `[0,1]`, a slice table, and a decision that acknowledges the small sample.
    Rubric: split discipline 2, controlled version 2, correct paired results 2, failure
    diagnosis 2, limitations 2.

    **Intermediate exercise (45 min).** Replace greedy decoding with three declared
    sampling seeds. Report within-prompt variation separately from between-prompt change.
    Do not average them into one unexplained number.

    **Challenge mini-project (90 min).** Complete `projects/prompt_evaluation`: add one
    output contract, one robustness paraphrase, and one critical slice. Select on
    development, open final test once, apply the gate, and write a rollback decision.
    Expected output: versioned manifest, prompt/data hashes, actual local outputs,
    paired uncertainty, invalid outputs, slice failures, and a scoped conclusion.

    **Summary:** controlled prompt engineering changes one context component, freezes
    every confounder, and trusts only paired evidence from untouched cases.
    **Memory aid:** *Version the prompt, freeze everything else, inspect every changed
    case, and reject complexity without evidence.*
    """),
]


build("05_nlp_and_llms/04_prompt_engineering.ipynb", cells)
