"""Builder for Lesson RAG-07 — Reranking."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md


cells = [
    md(r"""
    # RAG-07 · Reranking
    ### Section 06 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    EVAL-03 found a specific failure: relevant evidence was often inside the candidate
    list, but the extractive answerer selected the wrong passage. This lesson changes
    one component only. It freezes the retrieved candidates, assigns a new
    query-passage score, and checks whether the useful passages move earlier.

    **Decision rule:** rerank only after measured candidate recall is adequate and
    ordering errors remain. A reranker cannot retrieve missing evidence.
    """),

    md(r"""
    ## 1 · Learning Objectives

    By the end of the core lesson, you will be able to:

    1. distinguish candidate retrieval from candidate reranking;
    2. reorder a three-passage list manually and calculate its reciprocal-rank change;
    3. explain why candidate recall is a hard ceiling;
    4. compare an original ranking and reranked list without changing the candidates;
    5. separate a transparent pair scorer from a real neural cross-encoder;
    6. tune on development labels without looking at held-out evaluation labels;
    7. diagnose an improvement, a regression, and a candidate miss; and
    8. treat model scores and local latency measurements within their declared limits.

    **Prerequisite check:** complete RAG-06 and EVAL-03 first. You should already be
    able to explain candidate depth, stable evidence IDs, Recall@k, MRR, nDCG, and why
    evaluation data must not choose a model setting.
    """),

    md(r"""
    ## 2 · Historical Motivation and the Practical Problem

    ### Practical problem before history

    Imagine the query: “Should softmax be applied before cross entropy?” Hybrid
    retrieval returns these candidates:

    1. `neural.logits::sentence::0` — “Logits and cross entropy.”
    2. `neural.logits::sentence::2` — the sentence explaining that cross entropy
       expects raw logits and applies log-softmax internally.
    3. `leakage.definition::sentence::0` — “Definition.”

    Candidate recall is successful because the answer passage is present. Rank-one
    selection still fails because a heading appears before the explanatory sentence.
    Increasing retrieval depth does not directly repair this ordering problem.

    A **reranker** receives the query and a bounded candidate list, gives every pair a
    new relevance score, and sorts the same IDs. This is a second-stage precision
    decision, not another retrieval system.

    BERT passage reranking established the pattern of scoring query-passage pairs with
    a joint transformer on standard retrieval benchmarks. Read the paper as historical
    evidence, not as a guaranteed improvement on this corpus:
    <https://arxiv.org/abs/1901.04085>.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    ### Concept, analogy, and analogy limit

    A first-stage retriever is like a recruiter who quickly produces a shortlist.
    A reranker is the slower interviewer who reads each shortlisted résumé together
    with the exact job requirement and changes the order.

    The analogy stops here: candidates are passages, relevance comes from labels and
    user intent, and a model score is not a human judgment or a probability by default.

    ```text
    corpus ── fast retrieval ──> fixed candidate IDs ── pair scoring ──> new order
                 recall stage                            ordering stage
    ```

    Three boundaries matter:

    - **Membership:** reranking must preserve the candidate ID set.
    - **Ceiling:** an absent labelled passage cannot be moved into the result.
    - **Evaluation:** settings are chosen on development queries, then frozen before
      held-out evaluation.

    A bi-encoder scores separately created query and passage representations. A neural
    cross-encoder instead processes the query and passage jointly and returns one
    score for that pair. Joint processing can model richer interactions, but it must
    run for every candidate at query time.
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Reciprocal rank

    $$RR = \frac{1}{r}$$

    **Read aloud:** reciprocal rank is one divided by the position of the first
    relevant passage.

    **Symbols:** $RR$ is a scalar from 0 to 1. $r$ is the positive integer rank of the
    first labelled passage. If no labelled passage is present, this course records
    $RR=0$.

    **Small example:** before reranking, the answer passage is at rank 2, so
    $RR=1/2=0.5$. After it moves to rank 1, $RR=1/1=1.0$.

    **Meaning:** the useful passage moved earlier. **Use and limit:** RR is useful when
    the first relevant result matters. It ignores later relevant passages, so use
    nDCG or recall when a query needs several pieces of evidence.

    ### 4.2 Discounted cumulative gain

    $$DCG@k = \sum_{i=1}^{k}\frac{rel_i}{\log_2(i+1)}$$

    **Read aloud:** add each relevance label after discounting lower positions by a
    base-two logarithm.

    **Symbols:** $k$ is the number of ranks inspected; $i$ is a one-based rank;
    $rel_i$ is 1 for a labelled passage and 0 otherwise; $\sum$ means add the terms;
    $\log_2$ is the base-two logarithm. All values are scalars.

    **Small example:** for labels `[1, 0, 1]`,
    $DCG@3=1/\log_2(2)+0/\log_2(3)+1/\log_2(4)=1+0+0.5=1.5$.

    $$nDCG@k = \frac{DCG@k}{IDCG@k}$$

    **Read aloud:** normalized DCG is the observed discounted gain divided by the
    ideal discounted gain. $IDCG@k$ is DCG for the best possible ordering of the same
    labels. For two relevant passages in three ranks, $IDCG@3=1+1/\log_2(3)\approx
    1.631$, so $nDCG@3\approx1.5/1.631=0.920$.

    **Meaning:** 1 is ideal for the available labels; lower values indicate poorer
    ordering. **Use and limit:** nDCG handles multiple or graded labels, but its value
    is only as trustworthy as the relevance judgments.

    ### 4.3 Pair score and bounded cost

    $$s_j = f(q, d_j), \qquad C_{rerank} \approx m\,t_{pair}$$

    **Read aloud:** candidate $j$ receives the score produced from query $q$ and
    passage $d_j$; approximate reranking cost is the number of candidates times the
    measured time per pair.

    **Symbols:** $s_j$ is an ordering score, $f$ is the pair-scoring model, $q$ is the
    query, $d_j$ is candidate passage $j$, $m$ is candidate count, $t_{pair}$ is local
    time per pair, and $C_{rerank}$ is approximate local reranking time.

    **Small example:** 15 candidates at 2 ms per pair gives about $15\times2=30$ ms
    before batching and overhead. **Meaning:** cost grows with candidate width.
    **Use and limit:** this is a planning approximation, not a service SLA; batching,
    hardware, token lengths, concurrency, and network calls change real latency.
    """),

    code(r"""
    # Smallest manual example: reorder the same three IDs.
    query = "Should softmax be applied before cross entropy?"
    candidates = [
        {"id": "heading", "text": "Logits and cross entropy.", "pair_score": 0.32},
        {"id": "answer", "text": "Cross entropy expects raw logits and applies log-softmax internally.", "pair_score": 0.91},
        {"id": "distractor", "text": "Definition.", "pair_score": 0.04},
    ]

    original_ids = [candidate["id"] for candidate in candidates]
    reranked = sorted(candidates, key=lambda candidate: -candidate["pair_score"])
    reranked_ids = [candidate["id"] for candidate in reranked]

    original_rank = original_ids.index("answer") + 1
    reranked_rank = reranked_ids.index("answer") + 1
    print("Input candidate IDs:", original_ids)
    print("Pair scores:", [candidate["pair_score"] for candidate in candidates])
    print("Reranked IDs:", reranked_ids)
    print(f"Reciprocal rank: {1/original_rank:.1f} -> {1/reranked_rank:.1f}")
    assert set(original_ids) == set(reranked_ids)
    """),

    md(r"""
    ## 5 · Manual Implementation and Measured Project Benchmark

    The first code cell demonstrates the central operation: score, sort, preserve IDs,
    then compare ranking metrics. Its expected output is `['answer', 'heading',
    'distractor']` and reciprocal rank `0.5 -> 1.0`. A missing or new ID is a code
    failure. An unchanged metric is valid evidence that the scorer did not help.

    The project benchmark now applies the same contract to eighteen versioned queries.
    It uses passage-level labels and a fixed hybrid-RRF candidate set. A transparent
    character n-gram pair scorer teaches the mechanism offline. It is **not** a neural
    cross-encoder. Development labels select the blend weight; evaluation labels stay
    held out until final scoring.
    """),

    code(r"""
    import json
    from pathlib import Path

    report_path = Path("projects/rag_foundations/artifacts/reranking_evaluation.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = report["systems"]["evaluation"]
    baseline = evaluation["baseline_metrics"]
    local_reranker = evaluation["local_pair_reranker_metrics"]

    print("Corpus hash:", report["corpus_sha256"][:12])
    print("Query hash:", report["queries_sha256"][:12])
    print("Passage-label hash:", report["reranking_labels_sha256"][:12])
    print("Selected development alpha:", report["local_pair_scorer"]["selected_alpha"])
    print(f"Candidate hit rate: {baseline['candidate_hit_rate']:.4f}")
    print(f"MRR: {baseline['mrr']:.4f} -> {local_reranker['mrr']:.4f}")
    print(f"nDCG@5: {baseline['ndcg_at_k']:.4f} -> {local_reranker['ndcg_at_k']:.4f}")
    print(f"Top-1 accuracy: {baseline['top_1_accuracy']:.4f} -> {local_reranker['top_1_accuracy']:.4f}")
    """),

    md(r"""
    ### Reading the measured result

    1. Load one committed artifact rather than rebuilding retrieval inside the lesson.
    2. Verify three hashes before comparing experiments.
    3. Read candidate hit rate first; it is unchanged by reranking.
    4. Compare MRR and nDCG only on the held-out evaluation split.
    5. Notice that top-1 accuracy may remain flat even when average ordering improves.

    Expected held-out values are approximately MRR `0.5667 -> 0.6042`, nDCG@5
    `0.6249 -> 0.6436`, and candidate hit rate `0.8750` for both orders. Different
    hashes mean you are evaluating different data. Matching hashes with weaker metrics
    is a valid result to diagnose, not a reason to alter the labels.
    """),

    code(r"""
    # Inspect three different outcomes instead of showing only a win.
    rows = {row["query_id"]: row for row in evaluation["rows"]}
    for query_id in ("q09", "q13", "q14"):
        row = rows[query_id]
        relevant = set(row["relevant_passages"])

        def first_relevant_rank(ranking):
            return next((rank for rank, item in enumerate(ranking, 1) if item in relevant), None)

        print(f"\n{query_id}")
        print("  labelled passages:", row["relevant_passages"])
        print("  original rank:", first_relevant_rank(row["baseline_ranking"]))
        print("  reranked rank:", first_relevant_rank(row["reranked_ranking"]))
        print("  candidate hit:", row["candidate_hit"])
    """),

    md(r"""
    ## 6 · Visualization

    A useful visualization shows rank changes per query and keeps candidate misses
    visible. An aggregate-only bar can hide a regression.
    """),

    code(r"""
    import matplotlib.pyplot as plt

    query_ids, original_rr, reranked_rr = [], [], []
    for row in evaluation["rows"]:
        if not row["answerable"]:
            continue
        relevant = set(row["relevant_passages"])

        def reciprocal_rank(ranking):
            rank = next((i for i, item in enumerate(ranking, 1) if item in relevant), None)
            return 0.0 if rank is None else 1.0 / rank

        query_ids.append(row["query_id"])
        original_rr.append(reciprocal_rank(row["baseline_ranking"]))
        reranked_rr.append(reciprocal_rank(row["reranked_ranking"]))

    positions = range(len(query_ids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([position - width / 2 for position in positions], original_rr, width, label="Original")
    ax.bar([position + width / 2 for position in positions], reranked_rr, width, label="Reranked")
    ax.set_xticks(list(positions), query_ids)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Reciprocal rank")
    ax.set_title("Held-out passage rank changes; zero means candidate miss")
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    The plot should show `q09` improving, `q13` regressing, several no-gain rows, and
    `q14` remaining at zero. This supports a narrow claim: the local pair scorer makes
    a modest average improvement, but it neither helps every query nor repairs missing
    candidates.
    """),

    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    | Symptom | Likely cause | Evidence to inspect | Scoped fix |
    |---|---|---|---|
    | Relevant passage absent before and after | Candidate retrieval miss | Candidate IDs and candidate recall | Repair retrieval; do not tune reranker |
    | Candidate set changes | Pipeline contract violation | ID sets before/after | Freeze membership; sort only existing IDs |
    | Development improves, evaluation declines | Overfit or domain mismatch | Split metrics and query slices | Reduce tuning or collect representative labels |
    | Implausibly strong evaluation | Evaluation-label leakage | Alpha/model selection history | Recreate an untouched split before trusting results |
    | Raw score treated as probability | Uncalibrated model output | Activation and calibration evidence | Use scores for order only or calibrate separately |
    | Long passages dominate or truncate | Length/model limit mismatch | Token counts and truncation logs | Rerank appropriate chunks or evaluate windowing |
    | Latency claim does not reproduce | Hardware/workload mismatch | Device, batch, tokens, percentiles | Measure the actual serving path |
    | Correct evidence ranks high but answer fails | Downstream selection/composition issue | Selected IDs, answer, citations | Fix answer assembly; do not relabel ranking |

    Common beginner mistakes are claiming reranking improves recall, evaluating on
    training/development rows, changing retrieval and reranking simultaneously,
    comparing raw scores from different models, and reporting one successful query as
    a system-wide result.
    """),

    md(r"""
    ## 8 · Real Neural Cross-Encoder with Sentence Transformers

    A real cross-encoder receives pairs such as `(query, passage)` and runs joint
    transformer inference. It does not create reusable passage embeddings. The
    maintained library usage guide is
    <https://www.sbert.net/docs/cross_encoder/usage/usage.html>.

    The following cell never downloads silently. It runs only when the pinned revision
    is already cached; otherwise it prints the explicit project command. Raw MS MARCO
    logits are ordering scores, not calibrated probabilities.
    """),

    code(r"""
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_revision = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    demo_passages = [candidate["text"] for candidate in candidates]

    try:
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(
            model_name,
            revision=model_revision,
            local_files_only=True,
        )
        pairs = [(query, passage) for passage in demo_passages]
        neural_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        neural_order = sorted(
            zip(original_ids, neural_scores),
            key=lambda item: -float(item[1]),
        )
        print("Pinned neural order:", [item[0] for item in neural_order])
        print("Raw ordering logits:", [round(float(item[1]), 4) for item in neural_order])
    except (ImportError, OSError) as error:
        print("Pinned model is not available locally; no download was attempted.")
        print("Run the explicit neural-reranking command in the project README.")
        print("Reason:", type(error).__name__)
    """),

    md(r"""
    ## 9 · Realistic Case Study — Fix the Course RAG Answer Selector

    **Decision:** should the current course pipeline add a second-stage reranker?

    **Evidence:** EVAL-03 showed answer-selection failures with relevant sections
    already retrieved. RAG-07 then froze the candidate lists and added passage labels.
    On held-out queries, the transparent scorer increased MRR from `0.5667` to
    `0.6042` and nDCG@5 from `0.6249` to `0.6436`, while top-1 accuracy remained
    `0.3750`. The pinned neural extension measured MRR `0.6500`, nDCG@5 `0.6874`, and
    top-1 accuracy `0.5000` on the same machine and labels.

    **Decision:** keep reranking as a controlled candidate-stage experiment, not a
    universal default. Expand passage labels and diagnose `q14` retrieval before
    making a deployment decision. The current set is too small for a production SLA
    or domain-wide quality claim.

    **Expected workflow:** retrieve → freeze candidate IDs → rerank → verify the same
    IDs → compare held-out MRR/nDCG/top-1 → inspect regressions and misses → measure
    local latency with model revision and device metadata.
    """),

    md(r"""
    ## 10 · Production and Learning Considerations

    - Keep authorization, tenant, freshness, and safety filters before reranking; a
      high relevance score must not restore forbidden evidence.
    - Preserve stable IDs and provenance through every reorder.
    - Label passages at the same unit the model ranks. Broad document labels can hide
      whether the answer-bearing passage moved.
    - Mine hard negatives from realistic first-stage candidates, but keep entities or
      time periods separated where leakage is possible.
    - Record model name, immutable revision, tokenizer, maximum length, truncation,
      device, precision, batch size, candidate width, and label hashes.
    - Measure quality and latency together. Local mean time on a tiny notebook is not
      p95/p99 service latency under concurrency.
    - Calibrate only when a downstream decision needs probabilities or thresholds.
      Ranking by raw score does not itself require probability calibration.
    - Monitor slices. Aggregate improvement can coexist with domain, language, length,
      or query-type regressions.
    """),

    md(r"""
    ## 11 · Tradeoff and Related-Concept Comparison

    | Concept | Main purpose | Strengths | Weaknesses | Use when |
    |---|---|---|---|---|
    | BM25 / dense retrieval | Find candidates from a large corpus | Fast enough for broad search | Coarser interaction | Evidence may not yet be in a bounded list |
    | RRF hybrid fusion | Combine rank evidence from branches | No raw-score normalization | Still uses first-stage signals | Sparse and dense branches are complementary |
    | Transparent pair scorer | Teach and debug reorder mechanics | Offline, inspectable, deterministic | Not a neural cross-encoder; limited semantics | Learning contracts and regression tests |
    | Neural cross-encoder | Jointly score query-passage pairs | Rich query-conditioned interactions | Per-pair compute, domain and truncation risk | Candidate recall is adequate and ordering is weak |
    | Late interaction | Retain token-level matching with partial precomputation | Middle ground in quality/efficiency | More indexing/storage complexity | Full cross-encoding is too expensive |
    | No reranker | Preserve the first-stage order | Simplest, lowest added cost | Leaves ordering failures unchanged | Baseline already passes quality and latency needs |

    **Use reranking when:** candidate recall is adequate, rank errors are measured,
    and quality value justifies extra compute. **Avoid it when:** evidence is missing,
    the baseline already passes, or no representative labels exist. Prefer retrieval
    repair for candidate misses and answer composition repair when correct passages are
    already selected but the final response is wrong.
    """),

    md(r"""
    ## 12 · Readiness and Debugging Checklist

    Before proposing reranking, answer yes to all of these:

    1. Is candidate recall measured on the target unit and cutoff?
    2. Do traces show ordering failures rather than primarily candidate misses?
    3. Are candidate IDs frozen for the comparison?
    4. Are development and evaluation labels separated?
    5. Are MRR/nDCG/top-1 interpreted alongside regressions and slices?
    6. Is the scorer named honestly as heuristic, pair model, or cross-encoder?
    7. Are raw scores kept separate from calibrated probabilities?
    8. Are model revision, truncation, device, and local latency conditions recorded?

    If any answer is no, repair that evidence contract before tuning a larger model.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. What is the difference between retrieval and reranking?
    2. Why must the candidate ID set stay unchanged in a controlled comparison?
    3. Manually calculate RR when the first relevant passage moves from rank 4 to 2.
    4. Why can MRR improve while candidate recall remains exactly unchanged?
    5. Why is the local character pair scorer not a neural cross-encoder?
    6. What information leaks if alpha is chosen on held-out evaluation queries?
    7. What does `q14` prove about the reranker's ceiling?
    8. Why is a cross-encoder logit not automatically a probability?

    **Self-check:** a complete answer names the first-stage candidate set, the
    development/evaluation boundary, at least one ordering metric, and one limitation.
    """),

    md(r"""
    ## 14 · Exercises, Mini Project, Self-Check, and Solutions

    ### Worked example (10 minutes)

    Ranking `[noise, relevant, relevant]` has first relevant rank 2, so RR is `0.5`.
    Its DCG@3 is `0 + 1/log2(3) + 1/log2(4) ≈ 1.131`. The ideal DCG is
    `1 + 1/log2(3) ≈ 1.631`, so nDCG@3 is approximately `0.693`.

    ### Guided practice (20 minutes)

    Move the two relevant items to ranks 1 and 3. **Hint:** calculate DCG with the
    same denominator. **Expected result:** RR becomes `1.0`; nDCG@3 is approximately
    `0.920`. Common mistake: changing the candidate membership while changing order.

    ### Independent practice (35 minutes)

    Run `make reranking-evaluate`. Choose a held-out query other than `q09`, `q13`, or
    `q14`. Calculate the original and reranked RR manually, check the artifact, and
    explain whether the change is an improvement, regression, or no-gain result.
    **Self-check:** hashes match and both rankings contain exactly the same IDs.

    ### Challenge mini project (60 minutes)

    **Goal:** determine whether a neural cross-encoder is justified for this corpus.

    **Dataset columns:** `query_id`, `split`, `query`, `candidate_passage_id`,
    `passage_text`, `relevance_label`, `original_rank`, `reranked_rank`, and
    `reranker_score`.

    **Workflow:** keep the held-out split fixed → run the pinned neural model → record
    revision/device/batch/candidate width → compare MRR, nDCG@5, top-1, regressions,
    misses, and local latency → recommend adopt, reject, or collect more evidence.

    **Expected output:** a hash-bound comparison table plus one improved trace, one
    regression/no-gain trace, and one candidate miss. **Evaluation criteria:** 2 points
    each for contract integrity, correct metrics, leakage prevention, failure
    diagnosis, and appropriately limited recommendation.

    ### Solution and scoring rubric

    - Guided RR: `1.0`; guided nDCG@3: about `0.920`.
    - `q14` must remain a miss because the labelled passages are absent from the fixed
      candidate set.
    - Full credit requires unchanged candidate membership and no evaluation-driven
      tuning.
    - **Readiness threshold: 8/10**, including leakage prevention and candidate-ceiling
      diagnosis. Below 8, revisit EVAL-03 metrics and RAG-06 candidate tracing.

    **Plain-language summary:** retrieval finds the shortlist; reranking changes only
    its order. Measure the ceiling first, tune on development labels, and inspect both
    wins and regressions on held-out queries.

    **One-sentence memory aid:** A reranker can move good evidence up, but it cannot
    bring missing evidence in.
    """),
]


build("06_rag/07_reranking.ipynb", cells)
