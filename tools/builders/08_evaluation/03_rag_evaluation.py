"""Builder for Lesson EVAL-03 — RAG Evaluation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # EVAL-03 · RAG Evaluation
    ### Section 08 — Evaluation · Diagnose retrieval, answering, citation, and refusal separately

    > A final answer can be wrong because evidence was not retrieved, the wrong
    > retrieved passage was selected, the answer was unsupported, a citation was
    > invalid, or the system should have refused. One overall score cannot tell these
    > failures apart.

    This lesson evaluates the same versioned corpus, queries, answers, citations, and
    abstention cases used in the earlier RAG lessons. It starts with deterministic
    labelled metrics. LLM judges are an extension that must be calibrated against human
    labels; they are not treated as automatic truth.
    """),

    md(r"""
    ## 1 · Learning Objectives

    By the end of the lesson, you will be able to:

    - trace a RAG response through retrieval → answer selection → support → citation → refusal;
    - calculate labelled context precision@k and context recall@k manually;
    - distinguish answer correctness from evidence support;
    - label a metric honestly as a gold-label metric, deterministic proxy, or judge score;
    - compare lexical, dense LSA, and hybrid RRF under one versioned contract;
    - diagnose the first failed component from per-query traces;
    - build an explicit regression gate without pretending its thresholds are universal;
    - prepare RAG traces for an external evaluation library without making hidden model calls.

    **Why now:** RAG-03 taught retrieval labels and ranking metrics, RAG-04 taught
    grounded answers and citations, RAG-05 taught persistence and filters, and RAG-06
    taught measured fusion. EVAL-02 supplied general LLM evaluation vocabulary. This
    lesson combines those foundations before RAG-07 adds reranking.
    """),

    md(r"""
    ## 2 · Historical Motivation and Evidence

    RAG evaluation became a component problem because retrieval and generation can
    fail independently. RAGAS proposed metrics for retrieved context, faithfulness,
    and response relevance; ARES trains lightweight task-specific judges and uses a
    small human-labelled set to correct evaluation estimates.

    Primary sources:

    - Es et al., [RAGAs: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/)
    - Saad-Falcon et al., [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://aclanthology.org/2024.naacl-long.20/)

    These frameworks are useful patterns, not proof that an automated judge is correct
    for every domain. Judge prompts, models, metric APIs, and score distributions can
    change. Production use requires versioning and comparison with human decisions.
    This notebook makes no universal cost, latency, or human-agreement claim.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    Think of a research assistant:

    ```text
    question
       ↓
    retrieved evidence ── did the right sections arrive?
       ↓
    selected answer ───── did it contain the required information?
       ↓
    cited evidence ────── does the citation exist and support the text?
       ↓
    answer or refusal ─── was answering allowed by the available evidence?
    ```

    A librarian may bring the correct shelf but hand over the wrong book. Conversely,
    an answer may sound correct despite missing evidence because the model remembered
    it elsewhere. The analogy stops there: real systems also have authorization,
    freshness, injection, version, latency, and judge-calibration constraints.

    | Question | Metric contract | Stage |
    |---|---|---|
    | How much returned evidence is labelled relevant? | Context precision@k | Retrieval |
    | How much labelled relevant evidence was recovered? | Context recall@k | Retrieval |
    | Does the answer satisfy the declared answer label? | Correctness or correctness proxy | Answer selection |
    | Is the answer supported by cited evidence? | Support or faithfulness judge | Grounding |
    | Are citation IDs valid retrieved IDs? | Citation validity | Attribution |
    | Did the system answer only answerable questions? | Abstention accuracy | Decision policy |
    """),

    code(r"""
    import json
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    REPOSITORY_ROOT = next(
        path for path in (Path.cwd(), *Path.cwd().parents)
        if (path / 'projects/rag_foundations/src').exists()
    )
    PROJECT_SOURCE = REPOSITORY_ROOT / 'projects/rag_foundations/src'
    if str(PROJECT_SOURCE) not in sys.path:
        sys.path.insert(0, str(PROJECT_SOURCE))

    from rag_foundations.evaluation import load_json
    from rag_foundations.rag_evaluation import (
        DEFAULT_THRESHOLDS,
        apply_quality_gate,
        context_precision_at_k,
        evaluate_rag_systems,
    )

    DATA_DIRECTORY = REPOSITORY_ROOT / 'projects/rag_foundations/data'
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Loaded versioned local RAG evaluation tools.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Labelled context precision@k

    $$P@k=\frac{|R_k\cap G|}{|R_k|}$$

    **Read it aloud:** count the unique retrieved section IDs in the first $k$ that
    are also gold-relevant, then divide by the number returned.

    - $R_k$: set of unique retrieved section IDs in the first $k$ results.
    - $G$: set of gold-relevant section IDs.
    - $|\cdot|$: number of elements in a set.

    If five sections are returned and two are relevant, precision@5 is $2/5=0.4$.
    This requires relevance labels. For an unanswerable query with $G=\varnothing$,
    this lesson reports precision as **not applicable**, then evaluates abstention.

    ### 4.2 Labelled context recall@k

    $$R@k=\frac{|R_k\cap G|}{|G|}$$

    **Read it aloud:** divide the relevant evidence recovered in the first $k$ by all
    labelled relevant evidence. If two of three labelled sections are recovered,
    recall is $2/3\approx0.667$. Incomplete gold labels make the result incomplete;
    do not call it total real-world recall.

    ### 4.3 Required-term correctness proxy

    $$C_{\text{terms}}=\mathbb{1}[\text{every required normalized term occurs in the answer}]$$

    $\mathbb{1}[\cdot]$ is 1 when the condition is true and 0 otherwise. This detects
    whether a deterministic extractive answer contains declared terms. It is **not**
    semantic correctness: negation, context, and contradictory statements can fool it.

    ### 4.4 Extractive support proxy

    $$S_{\text{text}}=\mathbb{1}[\operatorname{normalize}(A)\subseteq
    \operatorname{normalize}(E_{\text{cited}})]$$

    $A$ is the answer and $E_{\text{cited}}$ is cited evidence text. The subset symbol
    here means normalized answer text occurs inside normalized evidence. This proves
    extractive containment, not semantic entailment or factual truth.

    ### 4.5 Citation validity and abstention accuracy

    Citation validity is 1 when every citation ID is among retrieved IDs. Abstention
    accuracy is the number of correct answer/refuse decisions divided by all decisions.
    A system can have perfect support yet low correctness because it faithfully quotes
    the wrong retrieved passage.
    """),

    code(r"""
    # Manual retrieval metric example.
    retrieved_ids = ['policy.returns', 'policy.shipping', 'faq.account', 'policy.refunds', 'faq.login']
    relevant_ids = {'policy.returns', 'policy.refunds', 'policy.warranty'}

    manual_precision = context_precision_at_k(retrieved_ids, relevant_ids)
    manual_recall = len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)

    print('Relevant retrieved:', sorted(set(retrieved_ids) & relevant_ids))
    print(f'Precision@5 = 2/5 = {manual_precision:.3f}')
    print(f'Recall@5 = 2/3 = {manual_recall:.3f}')
    assert np.isclose(manual_precision, 0.4)
    assert np.isclose(manual_recall, 2 / 3)
    """),

    md(r"""
    ## 5 · Manual Implementation and Metric Contracts

    A quality gate is only a declared policy over measured values. It should return
    the observed value, required minimum, and every violation instead of hiding the
    decision inside a single boolean.
    """),

    code(r"""
    teaching_metrics = {
        'context_precision_at_k': 0.20,
        'context_recall_at_k': 0.75,
        'answer_correctness_proxy': 0.50,
        'evidence_support_proxy': 1.00,
        'citation_validity': 1.00,
        'abstention_accuracy': 0.85,
    }
    teaching_gate = apply_quality_gate(teaching_metrics, DEFAULT_THRESHOLDS)
    print('Gate passed:', teaching_gate['passed'])
    print('Violations:')
    for violation in teaching_gate['violations']:
        print(' ', violation)

    assert not teaching_gate['passed']
    assert {item['metric'] for item in teaching_gate['violations']} == {
        'context_recall_at_k', 'abstention_accuracy'
    }
    """),

    md(r"""
    **Expected result.** The example fails recall and abstention only. A missing metric,
    silent default, or one undifferentiated “failed” message is a contract bug. The
    thresholds are teaching values used to demonstrate regression mechanics; they are
    not medical, legal, financial, or customer-support SLAs.
    """),

    md(r"""
    ## 6 · Versioned Project Evaluation and Visualization

    Evaluate lexical, dense LSA, and hybrid RRF using identical corpus, query, answer,
    chunking, `top_k`, metric, and threshold contracts. The answerer remains deliberately
    extractive so support and citation behavior are deterministic.
    """),

    code(r"""
    rag_report = evaluate_rag_systems(DATA_DIRECTORY)
    print('Corpus hash:', rag_report['corpus_sha256'])
    print('Query hash: ', rag_report['queries_sha256'])
    print('Answer hash:', rag_report['answers_sha256'])
    print('Strategy:', rag_report['strategy'], '| top_k:', rag_report['top_k'])

    display_metrics = (
        'context_precision_at_k', 'context_recall_at_k',
        'answer_correctness_proxy', 'evidence_support_proxy',
        'citation_validity', 'abstention_accuracy', 'successful_case_rate',
    )
    print(f"\n{'system':14s} " + ' '.join(f'{name[:9]:>9s}' for name in display_metrics) + ' gate')
    for system_name, system in rag_report['systems'].items():
        values = ' '.join(f"{system['metrics'][name]:9.3f}" for name in display_metrics)
        print(f"{system_name:14s} {values} {system['quality_gate']['passed']}")
    """),

    md(r"""
    **Expected committed result:**

    | System | Precision@5 | Recall@5 | Correctness proxy | Support proxy | Citation validity | Abstention | Success rate | Gate |
    |---|---:|---:|---:|---:|---:|---:|---:|---|
    | Lexical | 0.175 | 0.875 | 0.389 | 1.0 | 1.0 | 0.944 | 0.389 | Fail |
    | Dense LSA | 0.188 | 0.938 | 0.444 | 1.0 | 1.0 | 0.944 | 0.444 | Pass |
    | Hybrid RRF | 0.175 | 0.875 | 0.444 | 1.0 | 1.0 | 0.944 | 0.444 | Pass |

    Dense LSA has the highest retrieval recall on this corpus. Hybrid RRF does not
    improve it, even though RRF won the small RAG-06 diagnostic benchmark. Results are
    dataset-dependent. Perfect support is expected from extraction and must not hide
    the much lower answer correctness proxy.
    """),

    code(r"""
    metric_order = [
        'context_precision_at_k', 'context_recall_at_k',
        'answer_correctness_proxy', 'evidence_support_proxy',
        'citation_validity', 'abstention_accuracy',
    ]
    systems = list(rag_report['systems'])
    x = np.arange(len(metric_order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 5))
    for index, system_name in enumerate(systems):
        values = [rag_report['systems'][system_name]['metrics'][metric] for metric in metric_order]
        ax.bar(x + (index - 1) * width, values, width, label=system_name)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in metric_order], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Metric value')
    ax.set_title('Measured RAG components under one versioned contract')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    code(r"""
    # Plot real first-failure counts rather than simulated quadrants.
    failure_names = [
        'success', 'retrieval_failure', 'answer_failure',
        'attribution_failure', 'grounding_failure', 'abstention_failure',
    ]
    x = np.arange(len(failure_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5))
    for index, system_name in enumerate(systems):
        counts = rag_report['systems'][system_name]['failure_counts']
        ax.bar(x + (index - 1) * width, [counts[name] for name in failure_names], width, label=system_name)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in failure_names], fontsize=8)
    ax.set_ylabel('Number of 18 labelled cases')
    ax.set_title('First failed component by retrieval system')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Interpretation.** Most remaining failures are answer-selection failures: the
    relevant section may be in the five retrieved results, but the extractive baseline
    answers from rank 1. This is different from retrieval failure. Attribution and
    support remain perfect by construction. One answerable case is incorrectly refused,
    lowering abstention accuracy, but the first-failure taxonomy assigns it to retrieval
    when no relevant section was recovered. Independent metrics and first-failure labels
    answer different questions.
    """),

    code(r"""
    # Inspect component rows instead of diagnosing from averages alone.
    dense_rows = rag_report['systems']['dense_lsa']['rows']
    for outcome in ('retrieval_failure', 'answer_failure'):
        row = next(item for item in dense_rows if item['outcome'] == outcome)
        print(f"\n{outcome}: {row['query_id']}")
        print('  retrieved:', row['retrieved_sections'])
        print('  answer:', row['answer'][:100])
        print('  citations:', row['citations'])
        print('  recall:', row['retrieval_recall_at_k'])
        print('  correctness proxy:', row['answer_correctness'])
        print('  support proxy:', row['evidence_support'])
    """),

    md(r"""
    ## 7 · Failure Modes and Debugging

    | Symptom | Likely cause | Inspect | Scoped fix |
    |---|---|---|---|
    | Support is 1.0 but answers are wrong | Extractive answer copied the wrong passage | Relevant, retrieved, selected, and cited IDs | Improve selection/reranking; do not alter support labels |
    | Recall is high but precision is low | Large `k` returns many distractors | Metric by `k` and candidate ranks | Tune `k` on development labels or rerank |
    | Unanswerable precision looks perfect | Empty gold set was scored as retrieval success | Answerability and abstention fields | Mark precision not applicable; score refusal |
    | Judge score changes with no system change | Judge model, prompt, or decoding drifted | Judge version and human calibration set | Pin and recalibrate the judge |
    | Quality gate flips after threshold change | Policy changed, not model quality | Threshold version and violation rows | Version the policy separately from the system |
    | Average improves while a safety slice regresses | Aggregate hides subgroup failure | Per-query slices and confidence intervals | Gate important slices explicitly |
    | Implausibly strong semantic score | Random/hash embedding or leakage | Representation, seeds, fit boundaries | Replace proxy; rebuild evaluation before tuning |

    **Executable debugging rule:** find the first component whose input contract was
    satisfied but output contract failed. Do not “improve RAG” as one undifferentiated
    action.
    """),

    md(r"""
    ## 8 · Library-Compatible Trace Preparation

    The deterministic project report is the executable reference implementation. The
    optional RAGAS cell below constructs its current single-turn dataset schema but does
    not call a remote model or claim a judge score. Scoring requires a declared judge,
    prompt, embedding model, credentials, cost boundary, and human calibration set.
    APIs can change, so use the installed version's official documentation.
    """),

    code(r"""
    try:
        import ragas
        from ragas import EvaluationDataset, SingleTurnSample

        queries = load_json(DATA_DIRECTORY / 'queries.json')['queries']
        sample_row = rag_report['systems']['dense_lsa']['rows'][0]
        sample_query = next(item for item in queries if item['id'] == sample_row['query_id'])
        sample = SingleTurnSample(
            user_input=sample_query['query'],
            retrieved_context_ids=sample_row['retrieved_sections'],
            response=sample_row['answer'],
            reference_context_ids=sample_query['relevant_sections'],
        )
        evaluation_dataset = EvaluationDataset(samples=[sample], name='rag-foundations-demo')
        print('RAGAS version:', ragas.__version__)
        print('Prepared samples:', len(evaluation_dataset.samples))
        print('User input:', evaluation_dataset.samples[0].user_input)
        assert evaluation_dataset.samples[0].retrieved_context_ids == sample_row['retrieved_sections']
    except ImportError:
        print('RAGAS is optional in this kernel.')
        print('Install requirements-evaluation-production.txt and rerun this cell.')
    """),

    md(r"""
    ## 9 · Realistic Case Study — Curriculum Assistant Regression Gate

    **Measured teaching case.** Eighteen versioned questions cover direct,
    paraphrased, multi-concept, and unanswerable behavior. Three retrievers feed the
    same extractive answerer.

    The result shows why component evaluation matters:

    - dense LSA retrieves the most labelled evidence, Recall@5 0.938;
    - its answer correctness proxy is only 0.444 because top-one selection remains weak;
    - evidence support and citation validity are 1.0 because answers are extracted from
      their cited passages;
    - abstention accuracy is 0.944, so one answer/refuse decision still needs review.

    The next justified experiment is selection or reranking, not another retrieval
    change. These local results are not business impact or production capacity evidence.
    """),

    md(r"""
    ## 10 · Production and Safety Considerations

    - Version corpus, queries, gold evidence, answers, chunker, retriever, generator,
      prompt, judge, metrics, and threshold policy separately.
    - Calibrate automated judges against domain-expert labels; report false-pass and
      false-failure rates, not only correlation or an average score.
    - Preserve per-query traces and important slices. A mean can hide unsafe regressions.
    - Apply authorization, freshness, and unsafe-content filters before generation and
      before exporting traces to an evaluator.
    - Treat retrieved document instructions as data, not evaluator instructions.
    - Sample for expensive evaluation only after defining a sampling strategy; include
      failures and normal traffic without presenting the sample as the whole population.
    - Use end-to-end production percentiles for latency. The local report is a teaching
      measurement over a tiny corpus.
    - Review label coverage and disagreement. Gold sets can be incomplete or wrong.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    | Evaluation method | Strengths | Weaknesses | Use when | Avoid when |
    |---|---|---|---|---|
    | Gold evidence IDs | Deterministic retrieval diagnosis | Annotation is expensive and may be incomplete | Tuning retrieval and reranking | No evidence labels exist |
    | Required-term proxy | Cheap regression signal | Misses negation and semantics | Stable extractive baseline | Claiming true answer correctness |
    | Extractive support proxy | Auditable containment check | Not entailment or truth | Testing citation plumbing | Evaluating free-form synthesis alone |
    | LLM judge | Handles richer semantic rubrics | Model/prompt bias, cost, drift | Calibrated sampled evaluation | Unversioned or unvalidated high-stakes decisions |
    | Human review | Domain nuance and rubric refinement | Slow, costly, disagreement | Calibration and critical cases | Treating one reviewer as infallible gold |

    Choose the cheapest method that measures the actual risk, then validate its failure
    modes with a stronger method.
    """),

    md(r"""
    ## 12 · Readiness and Interview Preparation

    A strong diagnosis follows this order:

    1. verify data, label, and system hashes;
    2. inspect whether relevant evidence entered `top_k`;
    3. inspect which evidence the answerer selected;
    4. check correctness, support, and citations independently;
    5. check whether answering versus refusing was correct;
    6. change one component and rerun the same contract.

    If asked whether a faithfulness score of 1.0 proves the answer is true, answer no:
    it only means the answer is supported under that metric's evidence and judge
    contract. The evidence itself may be stale, wrong, malicious, or misinterpreted.
    """),

    md(r"""
    ## 13 · Teach-Back — Six Checks

    1. Calculate precision@5 and recall@5 when two of five retrieved sections are in a
       gold set containing three sections.
    2. How can support equal 1.0 while answer correctness equals 0?
    3. Why is context precision not applicable to an unanswerable query in this lesson?
    4. What is the first component to inspect when retrieval recall is zero?
    5. Why must judge model, prompt, and threshold policy have separate versions?
    6. What evidence would justify adding RAG-07 reranking?

    **Answer key:** 0.4 and 0.667; faithfully copying the wrong passage; there is no
    relevant set to be precise against, so refusal is the contract; retrieval and
    filters; otherwise score drift is misattributed; relevant evidence is in the
    candidate set but top-one answer selection or ranking remains weak.
    """),

    md(r"""
    ## 14 · Exercises, Solutions, and Mini Project

    ### Beginner 1 — trace classification · 10 minutes

    Relevant evidence is absent, the system refuses, and the question is answerable.
    Classify the first failure and the refusal decision.

    **Solution:** first failure is retrieval; abstention decision is also incorrect.
    First-failure taxonomy and independent abstention accuracy can both be true.

    ### Beginner 2 — metric calculation · 15 minutes

    `top_k=[a,b,c,d]`, gold evidence is `{b,d,e}`. Calculate precision and recall.

    **Solution:** intersection `{b,d}`; precision $2/4=0.5$; recall $2/3\approx0.667$.
    Common mistake: using the same denominator for both.

    ### Intermediate 1 — guided quality gate · 20 minutes

    Add `successful_case_rate` to a copied threshold policy. **Hint:** keep the policy
    separate from measured metrics and print the exact violation.

    **Self-check:** lowering a threshold changes the policy decision but not the saved
    system metric. Common mistake: rewriting the metric to make a gate pass.

    ### Intermediate 2 — independent component diagnosis · 30 minutes

    Run `make rag-system-evaluate`. Select one `answer_failure`. Explain why retrieval
    succeeded, why support still equals 1, and which one-variable experiment should run.

    **Rubric:** 2 points each for trace evidence, correct metric meaning, first failed
    component, controlled fix, and limitation. Pass: 8/10.

    ### Challenge — judge calibration design · 45–60 minutes

    Design a 50-case human calibration study for a future faithfulness judge. Include
    rubric, blinded review, disagreement handling, false-pass cost, false-failure cost,
    judge version, slices, and promotion rule. Do not invent agreement results.

    **Rubric:** 2 points each for label protocol, error analysis, risk-weighted
    threshold, versioning, and deployment/rollback rule. Pass: 8/10.

    ### Mini project — justify reranking

    - **Goal:** decide whether RAG-07 should rerank existing candidates.
    - **Data:** versioned query ID, gold evidence IDs, retrieved IDs, selected evidence,
      answer, required terms, citations, answerability, slice, and system version.
    - **Workflow:** preserve the best retrieval baseline → identify answer-selection
      failures with relevant evidence already present → propose a reranker → measure
      retrieval/ranking and answer metrics under the same labels.
    - **Expected output:** before/after component report, per-query deltas, failure
      taxonomy, latency limitation, and a keep/reject decision.
    - **Evaluation:** no label leakage, no hidden threshold change, no unsupported
      semantic claim, and no deployment if gains exist only on the tuning cases.
    """),

    md(r"""
    ---
    ### Summary

    RAG evaluation is a chain of contracts. Precision and recall evaluate labelled
    retrieval; correctness evaluates the answer contract; support and citations
    evaluate grounding and attribution; abstention evaluates whether answering was
    allowed. Proxies must be named as proxies, automated judges must be calibrated,
    and every comparison must preserve versioned inputs and policies.

    **Memory aid:** *Find the first broken component, keep every metric honest, and
    change one system variable under the same labels.*

    **Next after mastery:** RAG-07 reranking, justified by traces where relevant evidence
    is present but ranked or selected poorly.
    """),
]


build("08_evaluation/03_rag_evaluation.ipynb", cells)
