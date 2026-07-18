"""Builder for NLP-05 — LLM Failure Boundaries and Guardrails."""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # NLP-05 · LLM Failure Boundaries and Guardrails
    ### Define the policy, measure mistakes, and keep controls outside the model

    “Hallucination” is often used for several different failures. A claim can be false,
    unsupported by supplied evidence, correctly supported by an unreliable source, or
    correctly stated with an invalid citation. Safety policy, privacy, security, and
    output syntax are different again. This lesson separates those contracts and tests
    a deterministic local guardrail pipeline without pretending regex rules detect truth.

    **Prerequisites:** NLP-04 and EVAL-02. No hosted model or download is required.
    Real retrieval grounding is deliberately deferred to RAG-04, after retrieval exists.
    **Core time:** 5–8 hours with the checkpoint.
    """),
    md(r"""
    ## 1 · Learning Objectives

    After this lesson, you can:

    - distinguish factual correctness, evidence support, citation validity, policy
      compliance, privacy handling, and interface validity;
    - write an application-specific policy before building a classifier;
    - choose among `ALLOW`, `REDACT`, `BLOCK`, and `ESCALATE`;
    - draw trust boundaries and identify assets, actors, capabilities, and residual risk;
    - compare a keyword baseline with a measured candidate using action confusion data;
    - calculate critical escape and over-restriction rates;
    - redact authorized PII without storing or forwarding raw values;
    - validate structured claims, citations, source support, and authoritative truth separately;
    - explain why consistency, model confidence, NLI, temperature, and RAG are not universal truth detectors;
    - reject a guardrail release when one critical slice escapes.
    """),
    md(r"""
    ## 2 · Practical Problem and Motivation

    A support assistant receives ordinary questions, personal data needed for a task,
    prohibited requests for another user's credentials, and high-stakes medical or legal
    decisions. A single “safe/unsafe” label cannot express the required behavior.

    The simplest baseline blocks messages containing words such as `attack`, `password`,
    or `secret`, and blocks every detected email or identification number. It over-blocks
    benign security discussion, mishandles authorized PII, and misses paraphrases. A more
    explicit policy can improve those cases, but passing a small test suite does not create
    a security boundary. Identity, authorization, least-privilege tools, and human ownership
    remain outside the language model.

    For generated factual claims, lexical overlap is also insufficient. “Refunds last 90
    days” shares most words with a source saying 30 days. We need structured, independently
    named checks and later a measured retrieval-grounding system.
    """),
    md(r"""
    ## 3 · Concepts, Analogy, and Threat Contract

    A guardrail is a control that observes a boundary and chooses an action under a
    declared policy. Think of airport processing: a valid ticket, identity check, baggage
    inspection, and customs declaration answer different questions. Passing one does not
    imply passing the others. The analogy stops at intent—text classifiers infer patterns
    and can be evaded; airport authority also depends on law, trained staff, and physical controls.

    ### Five independent output contracts

    | Contract | Question | Example failure |
    |---|---|---|
    | Correctness | Is the claim true under the authoritative record? | says 60 instead of 30 days |
    | Evidence support | Does a cited source contain the claim? | cites a 30-day source for 90 days |
    | Citation validity | Does the citation resolve to an allowed source? | cites missing `P9` |
    | Policy compliance | Is the requested behavior permitted? | asks for another user's credential |
    | Interface validity | Does output satisfy the declared schema? | `claims` is a string, not a list |

    ### Threat contract

    | Element | Teaching system |
    |---|---|
    | Assets | credentials, PII, system instructions, approved policy facts |
    | Untrusted inputs | user text, future retrieved documents, model-generated tool arguments |
    | Attacker capability | paraphrase requests and place instructions inside data |
    | Required controls | authorization, redaction, least privilege, validation, audit, escalation |
    | Residual risk | unseen language and intent can bypass pattern rules |

    Delimiters help preserve structure but do not enforce authority. Never give the model
    access to a secret or tool merely because a prompt says not to misuse it.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Exact action accuracy

    $$A=\frac{1}{N}\sum_{i=1}^{N}\mathbf 1[\hat a_i=a_i].$$

    **Read aloud:** accuracy is the fraction of $N$ cases whose predicted action
    $\hat a_i$ equals expected action $a_i$. The indicator is 1 for a match and 0
    otherwise. For predictions matching 9 of 11 cases, $A=9/11=0.818$. Accuracy treats
    every mistake equally, so it cannot be the only release measure when one escape is
    more serious than several unnecessary escalations.

    ### 4.2 Critical escape rate

    Let $C$ be cases expected to `BLOCK` or `ESCALATE`, and let permissive actions be
    `ALLOW` or `REDACT`:

    $$E_C=\frac{\sum_{i\in C}\mathbf 1[\hat a_i\in\{ALLOW,REDACT\}]}{|C|}.$$

    **Read aloud:** count critical cases mistakenly given a permissive action, then
    divide by the number of critical cases. If 1 of 6 escapes, $E_C=1/6=0.167$.
    Lower is better. The result only covers the tested policy and slices; zero observed
    escapes is not proof that no attack exists.

    ### 4.3 Over-restriction rate

    Let $B$ be cases expected to `ALLOW` or `REDACT`:

    $$O_B=\frac{\sum_{i\in B}\mathbf 1[\hat a_i\in\{BLOCK,ESCALATE\}]}{|B|}.$$

    **Read aloud:** count benign or redactable cases that receive a restrictive action,
    then divide by benign cases. If 3 of 5 are unnecessarily blocked or escalated,
    $O_B=0.60$. This measures user friction, not safety by itself.

    ### 4.4 Structured evidence support

    For structured claim $c=(field,value)$ and cited source set $S_c$:

    $$support(c)=\mathbf 1[\exists s\in S_c:\ facts_s[field]=value].$$

    **Read aloud:** support is 1 when at least one valid cited source contains the same
    field and value. For claim `refund_window_days=60` citing a draft that says 60,
    support is 1—even if the approved policy says 30. Support therefore does not imply
    correctness, authority, freshness, or safety. Open-text claim extraction and NLI are
    more difficult and are not disguised by this exact teaching check.
    """),
    md(r"""
    ## 5 · Manual Implementation and Measured Local Project

    The executable project compares two deterministic input controls:

    - **Keyword baseline:** blocks a few words and every PII match.
    - **Policy candidate:** applies precedence `BLOCK → ESCALATE → REDACT → ALLOW`.

    The cases are authored and versioned. They include one deliberately unseen
    credential paraphrase so a superficially strong average cannot hide a critical escape.
    """),
    code(r"""
    import json
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    roots = [Path.cwd(), *Path.cwd().parents]
    repo_root = next(path for path in roots if (path / "projects/guardrail_evaluation").exists())
    sys.path.insert(0, str(repo_root / "projects/guardrail_evaluation/src"))

    from guardrail_evaluation.lab import (
        evaluate_output,
        policy_guard,
        redact_pii,
        run_guardrail_lab,
    )

    report = run_guardrail_lab()
    print(json.dumps(report["policy_contract"], indent=2))
    print("Dataset:", report["data"])
    """),
    md(r"""
    ### PII redaction before downstream use

    Redaction is appropriate only when the application is authorized to process the
    request and the sensitive value is unnecessary downstream. Blocking, tokenization,
    encryption, access control, and retention deletion solve different privacy problems.
    """),
    code(r"""
    raw_text = "Email alice@example.com about SSN 123-45-6789."
    safe_text, detected_kinds = redact_pii(raw_text)
    print("Raw text:     ", raw_text)
    print("Downstream:   ", safe_text)
    print("Detected types:", detected_kinds)
    assert "alice@example.com" not in safe_text and "123-45-6789" not in safe_text
    """),
    md(r"""
    **Expected result:** raw email and SSN disappear from the downstream string. Do not
    print raw PII in production logs as this teaching cell does with synthetic data.
    A regex miss is a privacy escape; a successful replacement does not prove legal basis,
    authorization, encryption, or deletion compliance.
    """),
    code(r"""
    baseline = report["input_guard"]["keyword_baseline"]
    candidate = report["input_guard"]["policy_candidate"]
    for name, evaluation in [("keyword baseline", baseline), ("policy candidate", candidate)]:
        print(name, {
            "exact_action_accuracy": round(evaluation["exact_action_accuracy"], 3),
            "critical_escape_rate": round(evaluation["critical_escape_rate"], 3),
            "over_restriction_rate": round(evaluation["over_restriction_rate"], 3),
        })
    print("Known escape:", report["known_escape"])
    print("Release decision:", report["release_decision"])
    """),
    md(r"""
    The candidate improves exact actions from about `0.45` to `0.91` and reduces
    over-restriction from `0.60` to `0`. It still allows one of six critical cases:
    “login phrase” paraphrases “password.” The declared zero-escape rule therefore
    returns `review_or_reject`. Adding that exact phrase to a regex may pass this suite
    while leaving the underlying intent-recognition problem unsolved.
    """),
    md(r"""
    ### Output contracts without a fake factuality score

    Fixed structured outputs make each check exact and reproducible. Approved source
    `P1` states 30 refund days; unapproved draft `P2` states 60. This creates the crucial
    case “supported by its cited source but incorrect under the authority.”
    """),
    code(r"""
    output_report = report["output_contracts"]
    print({
        key: round(value, 3)
        for key, value in output_report.items()
        if key.endswith("rate") or key == "contract_test_accuracy"
    })
    for row in output_report["rows"]:
        print(row["case_id"], row["measured"], row["output"]["answer"])
    """),
    md(r"""
    **Expected distinctions:** `O03` is schema-valid, citation-valid, and source-supported,
    but authoritatively incorrect. `O04` is correct but lacks a citation, so support is
    unverified. `O06` fails schema and therefore cannot safely proceed to later checks.
    """),
    md(r"""
    ## 6 · Visualization and Error Slices
    """),
    code(r"""
    metric_names = ["action accuracy", "critical escapes", "over-restriction"]
    baseline_values = [
        baseline["exact_action_accuracy"],
        baseline["critical_escape_rate"],
        baseline["over_restriction_rate"],
    ]
    candidate_values = [
        candidate["exact_action_accuracy"],
        candidate["critical_escape_rate"],
        candidate["over_restriction_rate"],
    ]
    positions = np.arange(len(metric_names))
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(positions - 0.18, baseline_values, 0.36, label="keyword baseline")
    axis.bar(positions + 0.18, candidate_values, 0.36, label="policy candidate")
    axis.set_xticks(positions, metric_names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("rate")
    axis.set_title("A stronger average does not override a critical escape")
    axis.legend()
    plt.tight_layout()
    plt.show()
    """),
    code(r"""
    action_order = ["ALLOW", "REDACT", "BLOCK", "ESCALATE"]
    matrix = np.zeros((4, 4), dtype=int)
    for row in candidate["rows"]:
        matrix[action_order.index(row["expected_action"]), action_order.index(row["predicted_action"])] += 1

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(4), action_order, rotation=20)
    axis.set_yticks(range(4), action_order)
    axis.set_xlabel("predicted action")
    axis.set_ylabel("expected action")
    axis.set_title("Policy-candidate action matrix")
    for row_index in range(4):
        for column_index in range(4):
            axis.text(column_index, row_index, matrix[row_index, column_index], ha="center", va="center")
    figure.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
    The off-diagonal `BLOCK → ALLOW` cell is the release blocker. Slice counts must be
    shown beside rates: a zero error on one or two cases is weak evidence. Add languages,
    indirect requests, encoded text, multi-turn context, and authorized edge cases before
    claiming broader coverage.
    """),
    md(r"""
    ## 7 · Failure Modes, Beginner Mistakes, and Debugging

    | Symptom | Likely cause | Inspect | Scoped repair |
    |---|---|---|---|
    | benign security text blocked | keyword treated as intent | exact matched rule and context | narrow rule; add benign slice |
    | credential paraphrase allowed | pattern coverage gap | critical escape rows | add semantic review/control; keep least privilege |
    | raw PII appears downstream | redaction miss or wrong precedence | before/after trace without persistent raw logs | repair detector and data flow |
    | cited claim still wrong | source is unapproved or stale | source authority/version | require approved sources and freshness |
    | correct answer fails support | citation missing/invalid | claim-citation mapping | repair citation, not factual label |
    | consistency is high but claim false | systematic model error | authoritative record | treat agreement as disagreement signal only |
    | guard passes but tool causes harm | model was allowed to authorize action | identity, permissions, tool arguments | enforce authorization outside model |

    Common mistakes: defining policy after seeing outputs, reporting one safety score,
    treating regex as intent detection, retaining sensitive blocked text indefinitely,
    trusting model confidence, calling `NLI neutral` false, and assuming retrieval makes
    generated claims supported.
    """),
    md(r"""
    ## 8 · Related Detection and Control Tools

    - **NLI:** with evidence as premise and claim as hypothesis, `entailment` suggests
      support, `contradiction` suggests conflict, and `neutral` means the evidence does
      not decide. Classifier errors and domain shift require calibration.
    - **Cross-sample consistency:** disagreement can prioritize review, but consistent
      falsehoods and diverse truthful paraphrases prevent it from being a truth detector.
    - **Temperature:** changes sampling diversity. Greedy decoding can repeat a false
      mode perfectly; no universal temperature makes claims factual.
    - **Model-stated confidence:** text such as “95% confident” is not calibrated unless
      a separate evaluation demonstrates that relationship.
    - **Schema libraries:** JSON Schema or Pydantic can enforce structure, not meaning.
    - **Policy classifiers:** may cover paraphrases better than rules but still require
      labelled slices, thresholds, appeals, monitoring, and controls outside the model.

    Real RAG grounding is not executed here. RAG-01 through RAG-03 build retrieval;
    RAG-04 evaluates whether generated claims and citations are supported by retrieved evidence.
    """),
    md(r"""
    ## 9 · Realistic Case Study — Account Support Assistant

    An account-support assistant can explain public product policy but cannot retrieve
    another user's credentials or make account changes without authenticated tools.
    Authorized contact information is redacted before entering an analytics model. Medical
    and legal decisions route to approved human workflows rather than receiving an invented
    confidence score.

    The release suite contains ordinary requests, benign security language, authorized PII,
    direct and paraphrased credential requests, instruction exfiltration, and high-stakes
    cases. Each action has an owner and appeal path. Tool services independently verify
    identity, account ownership, argument schema, and permissions. A blocked text sample
    is retained only under a documented minimal retention policy.
    """),
    md(r"""
    ## 10 · Learning and Production Considerations

    - Version policy text, cases, expected actions, rules/models, thresholds, and owners.
    - Minimize and redact logs; encrypt necessary audit records and enforce retention deletion.
    - Measure exact actions, critical escapes, over-restriction, slices, and appeals separately.
    - Use shadow evaluation and human review before enforcement changes.
    - Treat user text, retrieved documents, model outputs, and tool responses as untrusted inputs.
    - Put authentication, authorization, rate limits, spend limits, and irreversible-action
      confirmation in deterministic services outside the model.
    - Define fail-open versus fail-closed behavior per action and outage mode.
    - Record residual risk and incident response; no model or rule set “solves safety.”
    """),
    md(r"""
    ## 11 · Tradeoff and Related-Concept Analysis

    | Control | Main purpose | Strength | Weakness | Use when |
    |---|---|---|---|---|
    | Exact rule | enforce known pattern | transparent and fast | easy to evade/overmatch | syntax is stable and narrow |
    | Redaction | remove unnecessary sensitive values | preserves benign workflow | detectors miss formats | downstream does not need raw PII |
    | Policy classifier | infer labelled policy category | handles more variation | dataset and threshold dependent | representative labels exist |
    | Escalation | transfer uncertain/high-risk decision | human accountability | delay and capacity cost | consequences exceed automation evidence |
    | Schema validator | protect software interface | deterministic | meaning can be wrong | downstream expects structure |
    | Evidence-support check | test claim against source | inspectable grounding | source may be wrong/stale | authoritative evidence is supplied |
    | Retrieval | obtain external evidence | updates model context | can retrieve wrong/untrusted text | implemented and measured after RAG lessons |
    | SFT/preference training | shift model behavior | reduces repeated failure patterns | cannot enforce permissions | data supports a stable behavioral change |
    """),
    md(r"""
    ## 12 · Readiness Check

    You are ready for RAG when you can:

    1. name all five independent output contracts;
    2. justify `ALLOW`, `REDACT`, `BLOCK`, and `ESCALATE` precedence;
    3. calculate critical escape and over-restriction rates manually;
    4. trace synthetic PII through redaction without persisting it;
    5. explain the supported-but-incorrect draft-source case;
    6. identify the paraphrased credential escape in the action matrix;
    7. explain why delimiters, consistency, temperature, and confidence are insufficient;
    8. defend `review_or_reject` despite 91% exact action accuracy.
    """),
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. How do factual correctness and evidence support differ?
    2. Why can a valid citation fail to support a claim?
    3. When should PII be redacted rather than blocked?
    4. What do critical escape and over-restriction measure?
    5. Why did the keyword baseline block benign security language?
    6. Why is consistent generation not proof of truth?
    7. What does NLI `neutral` mean?
    8. Which controls must remain outside the language model?
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, Solutions, and Mini-Project

    **Worked example (15 min).** Six critical cases contain one permissive prediction,
    so critical escape rate is `1/6=0.167`. Five benign/redactable cases contain no
    restrictive prediction, so over-restriction is `0/5=0`. Common mistake: using all
    11 cases as the denominator for both rates.

    **Guided practice (25 min).** Add an authorized phone number. Hint: expected action
    is `REDACT`, and the safe text must contain no raw number. **Self-check:** exact action
    remains correct, redaction type is recorded, and raw PII is absent downstream.

    **Independent practice (45 min).** Add six credential-request paraphrases and six
    benign security cases. Report the action matrix, critical escape, over-restriction,
    and failed rows. Rubric: policy labels 2, privacy-safe data 2, correct metrics 2,
    slice diagnosis 2, residual risk 2.

    **Intermediate exercise (45 min).** Add evidence version and authority checks.
    Create one claim supported by a stale source and one unsupported claim that happens
    to be correct. Keep support, freshness, authority, and truth as separate columns.

    **Challenge mini-project (90 min).** Complete `projects/guardrail_evaluation` with
    one multilingual slice, one fail-open/fail-closed outage decision, one appeal path,
    and one authorization check outside the model. Expected output: versioned policy,
    data hash, action confusion, redaction trace, independent output contracts, critical
    escapes, release decision, owners, and residual risk.

    **Summary:** guardrails implement a declared policy at explicit boundaries; they do
    not turn a language model into a truth oracle or authorization service.
    **Memory aid:** *Separate the contracts, minimize sensitive data, measure both escapes
    and over-blocking, and keep authority outside the model.*
    """),
]


build("05_nlp_and_llms/05_hallucination_and_guardrails.ipynb", cells)
