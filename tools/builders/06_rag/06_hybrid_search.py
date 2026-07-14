"""Builder for Lesson RAG-06 — Hybrid Search."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # RAG-06 · Hybrid Search
    ### Section 06 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    > A dense retriever can understand paraphrases yet blur exact identifiers. A
    > lexical retriever can preserve exact words yet miss vocabulary changes. Hybrid
    > search asks a measurable question: **does combining their candidate lists improve
    > retrieval on our labelled queries enough to justify the added complexity?**

    This lesson extends the measured `rag_foundations` project from RAG-03 and RAG-05.
    It does not assume hybrid is automatically better.
    """),

    md(r"""
    ## 1 · Learning Objectives

    By the end of this lesson, you will be able to:

    - predict when BM25 or dense retrieval is likely to help;
    - calculate one BM25 term contribution and one RRF score by hand;
    - implement candidate union, stable-ID deduplication, RRF, and alpha fusion;
    - compare BM25, dense LSA, RRF, and alpha fusion on the same labelled queries;
    - interpret Recall@k, MRR, nDCG, query slices, abstention, and candidate depth;
    - keep authorization, freshness, safety, and provenance intact in both branches;
    - decide when hybrid retrieval should **not** be deployed.

    **Prerequisite check.** NLP-01 supplied BM25; NLP-02 and RAG-01 supplied vectors
    and similarity; RAG-03 supplied labelled retrieval evaluation; RAG-05 supplied
    persistent storage and filters. If those ideas are not familiar, revisit them
    before continuing.
    """),

    md(r"""
    ## 2 · Historical Motivation

    Traditional sparse retrieval ranks exact terms. BM25 added term-frequency
    saturation and document-length normalization to this family. Dense Passage
    Retrieval later showed how learned vector representations could retrieve passages
    without requiring every query word to appear literally. Neither signal dominates
    every domain, so rank fusion combines independently produced candidate lists.

    Primary reading:

    - Robertson and Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
    - Karpukhin et al., [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
    - Cormack, Clarke, and Büttcher, [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
    - Thakur et al., [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation](https://arxiv.org/abs/2104.08663)

    These sources support the algorithms and benchmark motivation. Results later in
    this notebook come from the repository's small local teaching dataset; they are
    not vendor benchmarks or production capacity claims.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    ### The problem

    Imagine two librarians. One remembers exact catalogue labels such as `ZX-410`.
    The other understands that “restore the converter” is similar to “reset the
    inverter.” Asking both may recover evidence that either one misses. A fusion rule
    merges their lists by stable document ID.

    The analogy stops here: real retrievers return noisy numerical rankings, enforce
    access policies, and can fail together because they share the same bad corpus.

    | Query condition | BM25 tendency | Dense tendency | Sensible first choice |
    |---|---|---|---|
    | Product code or error code | Preserves exact token | May blur nearby identifiers | BM25 baseline, then measure hybrid |
    | Paraphrase with different wording | May miss vocabulary change | Can capture related meaning | Dense baseline, then measure hybrid |
    | Mixed identifier and intent | Captures identifier | Captures intent | Hybrid candidate generation |
    | Tiny corpus with stable exact vocabulary | Often sufficient | Extra model and index may add little | BM25 only |
    | Strict latency budget with no measured hybrid gain | Simpler | One branch may be sufficient | Best measured single retriever |

    **Core process:** retrieve more than the final `top_k` from both branches → apply
    equivalent policy filters → take the candidate union → deduplicate by stable ID →
    fuse → keep the final `top_k` → evaluate by query slice.
    """),

    code(r"""
    import math
    import re
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

    from rag_foundations.evaluation import BM25Index, RetrievalIndex, build_chunks, load_json
    from rag_foundations.hybrid import evaluate_hybrid, minmax_score_fusion

    plt.rcParams['figure.figsize'] = (9, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Loaded deterministic local retrieval tools.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 BM25 recap

    For a query $q$ and document $d$:

    $$
    \operatorname{BM25}(d,q)=\sum_{t\in q}\operatorname{IDF}(t)
    \frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}
    $$

    **Read it aloud:** for each query term, multiply how rare the term is by its
    saturated, length-adjusted frequency in the document, then add the contributions.

    - $t$: one query term; $q$: the set or sequence of query terms.
    - $f(t,d)$: number of occurrences of term $t$ in document $d$; a count.
    - $|d|$: document length in tokens; $\operatorname{avgdl}$: average corpus length.
    - $k_1\geq0$: term-frequency saturation control; larger values saturate more slowly.
    - $b\in[0,1]$: document-length normalization control.
    - $N$: number of documents; $n(t)$: documents containing $t$.

    We use Robertson's non-negative IDF variant:

    $$\operatorname{IDF}(t)=\log\left(\frac{N-n(t)+0.5}{n(t)+0.5}+1\right).$$

    **Manual example.** Let $N=10$, $n(t)=2$, $f(t,d)=3$, $|d|=100$,
    $\operatorname{avgdl}=80$, $k_1=1.5$, and $b=0.75$.

    $$\operatorname{IDF}(t)=\log(4.4)\approx1.482$$

    $$1-b+b|d|/\operatorname{avgdl}=0.25+0.75(100/80)=1.1875$$

    $$\text{saturated TF}=\frac{3(2.5)}{3+1.5(1.1875)}\approx1.569$$

    The term contributes about $1.482\times1.569=2.325$ points. This is a ranking
    score, not a probability. BM25 cannot match a query term absent from its index.

    ### 4.2 Reciprocal Rank Fusion

    $$\operatorname{RRF}(d)=\sum_{r:d\in L_r}\frac{1}{k+\operatorname{rank}_r(d)}$$

    **Read it aloud:** for every ranked list containing document $d$, add the inverse
    of the constant $k$ plus that document's one-based rank.

    - $L_r$: ranked candidate list number $r$.
    - $\operatorname{rank}_r(d)$: one-based position of $d$ in list $L_r$.
    - $k$: rank constant; commonly 60, but still a parameter.
    - A document missing from a list receives **no contribution from that list**.

    If a document is rank 2 in BM25 and rank 5 in dense retrieval with $k=60$:

    $$\operatorname{RRF}(d)=1/62+1/65\approx0.03152.$$

    RRF avoids comparing incompatible raw scores. It does not calibrate relevance,
    guarantee improvement, or remove the need to choose candidate depth.

    ### 4.3 Alpha-weighted fusion

    $$s_{\text{hybrid}}(d)=\alpha\hat{s}_{\text{dense}}(d)+(1-\alpha)\hat{s}_{\text{BM25}}(d)$$

    Here $s$ is a scalar score, a hat means the score was normalized, and
    $\alpha\in[0,1]$ controls dense weight. With normalized dense score 0.4, normalized
    BM25 score 0.8, and $\alpha=0.25$, the fused score is
    $0.25(0.4)+0.75(0.8)=0.7$.

    Min-max normalization depends on the returned candidate set and is sensitive to
    outliers. Tune alpha only on labelled development queries; never choose it from
    the final test set.
    """),

    code(r"""
    # Verify the manual BM25 calculation before implementing fusion.
    document_count = 10
    matching_documents = 2
    term_frequency = 3
    document_length = 100
    average_document_length = 80
    k1 = 1.5
    b = 0.75

    inverse_document_frequency = math.log(
        (document_count - matching_documents + 0.5) / (matching_documents + 0.5) + 1
    )
    length_factor = 1 - b + b * document_length / average_document_length
    saturated_frequency = term_frequency * (k1 + 1) / (term_frequency + k1 * length_factor)
    term_contribution = inverse_document_frequency * saturated_frequency

    print(f'IDF: {inverse_document_frequency:.3f}')
    print(f'Length factor: {length_factor:.4f}')
    print(f'Saturated term frequency: {saturated_frequency:.3f}')
    print(f'BM25 term contribution: {term_contribution:.3f}')
    assert np.isclose(term_contribution, 2.325, atol=0.001)
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    The first example isolates fusion. The rankings are deliberately supplied by hand,
    so no simulated vector is mistaken for a real encoder. Section 6 then runs real
    BM25 and dense LSA retrieval on the project's labelled data.
    """),

    code(r"""
    # Each tuple is (stable_document_id, branch_score).
    bm25_candidates = [('doc-zx410', 8.2), ('doc-safety', 3.1), ('doc-overload', 1.2)]
    dense_candidates = [('doc-overload', 0.91), ('doc-zx410', 0.73), ('doc-restart', 0.61)]

    def reciprocal_rank_fusion_by_id(ranked_lists, rank_constant=60):
        fused_scores = {}
        for ranked_list in ranked_lists:
            for one_based_rank, (document_id, _branch_score) in enumerate(ranked_list, start=1):
                fused_scores[document_id] = fused_scores.get(document_id, 0.0) + (
                    1.0 / (rank_constant + one_based_rank)
                )
        return sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))

    fused_candidates = reciprocal_rank_fusion_by_id([bm25_candidates, dense_candidates])
    print('Candidate union:', [document_id for document_id, _ in fused_candidates])
    for rank, (document_id, score) in enumerate(fused_candidates, start=1):
        print(f'{rank}. {document_id:12s} RRF={score:.5f}')

    expected_union = {'doc-zx410', 'doc-safety', 'doc-overload', 'doc-restart'}
    assert {document_id for document_id, _ in fused_candidates} == expected_union
    assert fused_candidates[0][0] == 'doc-zx410'
    """),

    md(r"""
    **Expected result.** Four unique IDs should remain. `doc-zx410` ranks first because
    both branches support it. `doc-safety` and `doc-restart` still remain candidates;
    missing from one list means no contribution from that list, not an invented worst
    rank. A duplicate ID in the final list or an ID outside the union means fusion is
    broken.
    """),

    code(r"""
    # A small executable BM25 check: an unseen query should return no arbitrary documents.
    data_directory = REPOSITORY_ROOT / 'projects/rag_foundations/data'
    corpus_data = load_json(data_directory / 'corpus.json')
    structure_chunks = build_chunks(corpus_data, 'structure')
    bm25_index = BM25Index(structure_chunks)

    exact_results = bm25_index.search('log-softmax', k=3)
    unknown_results = bm25_index.search('zxqv-9999', k=3)

    print('Exact-token top result:', exact_results[0][0].section_id, round(exact_results[0][1], 3))
    print('Unknown-token results:', unknown_results)
    assert exact_results[0][0].section_id == 'neural.logits'
    assert unknown_results == []
    """),

    md(r"""
    ## 6 · Visualization and Measured Project Comparison

    Now evaluate every method on the versioned `hybrid_corpus.json` and
    `hybrid_queries.json` benchmark with the same structure-aware chunks, labels,
    `top_k=3`, and candidate depth 9. Its slices cover exact identifier, paraphrase,
    mixed intent, no-gain control, and unanswerable behavior. Four-component dense
    LSA is a real statistical dense representation, but it is not a neural sentence
    encoder.
    """),

    code(r"""
    hybrid_report = evaluate_hybrid(data_directory)
    metric_names = ('recall_at_k', 'mrr', 'ndcg_at_k', 'unanswerable_abstention_rate')

    print('top_k =', hybrid_report['top_k'], '| candidate_k =', hybrid_report['candidate_k'])
    print(f"{'method':24s}  recall   MRR   nDCG  abstain")
    for method, result in hybrid_report['experiments'].items():
        values = [result[name] for name in metric_names]
        print(f'{method:24s}  ' + '  '.join(f'{value:.3f}' for value in values))

    base_recall = max(
        hybrid_report['experiments'][name]['recall_at_k']
        for name in ('bm25', 'dense_lsa')
    )
    fused_recall = max(
        result['recall_at_k']
        for name, result in hybrid_report['experiments'].items()
        if name.startswith('hybrid_')
    )
    assert fused_recall >= base_recall
    assert all(
        result['unanswerable_abstention_rate'] == 1.0
        for result in hybrid_report['experiments'].values()
    )
    """),

    md(r"""
    **Expected result for the committed diagnostic benchmark.** BM25 reaches
    Recall@3 0.875, dense LSA reaches 0.75, and RRF reaches 1.0. RRF also has the best
    MRR (0.875) and nDCG@3 (about 0.923). Every method abstains on `h05`. If labels or
    corpus hashes change, treat the printed report—not these historical expected
    values—as authoritative and review the diff.

    A valid but weaker result is not a code failure. It means the method did not earn
    deployment on this evaluation set.
    """),

    code(r"""
    alpha_methods = [
        name for name in hybrid_report['experiments']
        if name.startswith('hybrid_alpha_')
    ]
    alpha_values = [float(name.rsplit('_', 1)[1]) for name in alpha_methods]
    alpha_recall = [hybrid_report['experiments'][name]['recall_at_k'] for name in alpha_methods]
    alpha_mrr = [hybrid_report['experiments'][name]['mrr'] for name in alpha_methods]
    rrf_recall = hybrid_report['experiments']['hybrid_rrf']['recall_at_k']

    fig, ax = plt.subplots()
    ax.plot(alpha_values, alpha_recall, 'o-', label='Alpha fusion Recall@3')
    ax.plot(alpha_values, alpha_mrr, 's-', label='Alpha fusion MRR')
    ax.axhline(rrf_recall, color='tab:red', linestyle='--', label=f'RRF Recall@3={rrf_recall:.3f}')
    ax.set_xlabel('alpha (0 = BM25 only, 1 = dense only)')
    ax.set_ylabel('Metric value')
    ax.set_ylim(0, 1.05)
    ax.set_title('Measured fusion tradeoff on labelled project queries')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Interpretation.** RRF earns further evaluation on this diagnostic benchmark:
    `h01` shows BM25's identifier ranking advantage, `h02` shows dense LSA's paraphrase
    ranking advantage, and RRF recovers both relevant sections for mixed-intent `h03`.
    All methods already rank `h04` correctly, so hybrid adds no value there. Weighted
    alpha fusion does not match RRF on this set; a familiar alpha such as 0.5 is not
    automatically a good default. Five teaching queries are still too small for a
    deployment claim.
    """),

    code(r"""
    # Compare query-level changes instead of hiding behind one average.
    rows_by_method = {
        method: {row['query_id']: row for row in result['rows']}
        for method, result in hybrid_report['experiments'].items()
    }
    assert rows_by_method['bm25']['h01']['reciprocal_rank'] > rows_by_method['dense_lsa']['h01']['reciprocal_rank']
    assert rows_by_method['dense_lsa']['h02']['reciprocal_rank'] > rows_by_method['bm25']['h02']['reciprocal_rank']
    assert rows_by_method['hybrid_rrf']['h03']['recall_at_k'] == 1.0
    assert rows_by_method['bm25']['h04']['reciprocal_rank'] == 1.0
    assert rows_by_method['dense_lsa']['h04']['reciprocal_rank'] == 1.0
    assert rows_by_method['hybrid_rrf']['h05']['abstained']

    print('h01: BM25 ranks the exact identifier higher than dense LSA.')
    print('h02: dense LSA ranks the paraphrase higher than BM25.')
    print('h03: RRF retrieves both mixed-intent evidence sections.')
    print('h04: all three methods rank the control correctly; fusion adds no gain.')
    print('h05: RRF abstains when neither branch supplies evidence.')

    improvements = []
    for query_id in rows_by_method['hybrid_rrf']:
        bm25_value = rows_by_method['bm25'][query_id]['recall_at_k']
        dense_value = rows_by_method['dense_lsa'][query_id]['recall_at_k']
        rrf_value = rows_by_method['hybrid_rrf'][query_id]['recall_at_k']
        if rrf_value > min(bm25_value, dense_value):
            improvements.append((query_id, bm25_value, dense_value, rrf_value))
    if improvements:
        print('Queries where RRF improves Recall@3 over at least one base retriever:')
        for query_id, bm25_value, dense_value, rrf_value in improvements:
            print(f'  {query_id}: BM25={bm25_value:.1f}, dense={dense_value:.1f}, RRF={rrf_value:.1f}')
    else:
        print('No query-level RRF recall improvement. Add missing query slices before claiming a gain.')
    """),

    md(r"""
    ## 7 · Failure Modes and Debugging

    | Symptom | Likely cause | Evidence to inspect | Scoped fix |
    |---|---|---|---|
    | Unknown query returns arbitrary documents | Zero-score results were sliced into `top_k` | Branch scores and abstention row | Remove non-positive BM25 hits; define evidence threshold |
    | Same chunk appears twice | Fusion used list position instead of stable ID | Candidate IDs before and after union | Deduplicate by stable chunk ID before final ranking |
    | Restricted chunk appears after fusion | Policy applied after one branch or after fusion | Sparse and dense candidate traces | Apply equivalent policy filters before both branch rankings |
    | One branch always dominates alpha fusion | Raw score ranges differ or outlier controls normalization | Per-branch score histograms | Normalize on a declared candidate set or use RRF |
    | Relevant chunk never enters fused list | Candidate depth is too small | Branch candidate IDs at several depths | Tune candidate depth on development labels |
    | Good average, poor identifier queries | Aggregate hides a failure slice | Exact/entity/paraphrase slice metrics | Add labelled slice coverage; adjust retriever only with evidence |
    | No gain over one branch | Signals overlap or one branch is weak | Per-query deltas and latency | Keep the simpler measured winner |

    **Important correction:** BM25+ does not solve unseen vocabulary. It lower-bounds
    the term-frequency contribution for matching terms so very long matching documents
    are not over-penalized. For unseen terms, use query expansion, synonym handling,
    a suitable tokenizer, or a complementary dense representation. See Lv and Zhai,
    [Lower-Bounding Term Frequency Normalization](https://timan.cs.illinois.edu/czhai/pub/cikm11-bm25.pdf).
    """),

    code(r"""
    # Executable debugging check: the union must be unique and deterministic.
    first_run = reciprocal_rank_fusion_by_id([bm25_candidates, dense_candidates])
    second_run = reciprocal_rank_fusion_by_id([bm25_candidates, dense_candidates])
    first_ids = [document_id for document_id, _ in first_run]
    assert first_run == second_run
    assert len(first_ids) == len(set(first_ids))
    print('Deterministic stable-ID fusion check passed:', first_ids)
    """),

    md(r"""
    ## 8 · Library and Project Implementation

    The project implementation you just ran uses scikit-learn's TF-IDF, truncated SVD,
    and normalization for the dense LSA branch, plus the repository's tested BM25 and
    fusion functions. The next cell runs `rank_bm25` when the NLP/RAG requirements are
    installed; it never prints unexecuted code as if it were a result.
    """),

    code(r"""
    try:
        from rank_bm25 import BM25Okapi

        library_documents = [
            'ZX-410 inverter reset procedure',
            'restore the power converter after an overload',
            'employee travel reimbursement policy',
        ]
        library_tokens = [re.findall(r'[a-z0-9]+(?:[-.][a-z0-9]+)*', text.lower())
                          for text in library_documents]
        library_bm25 = BM25Okapi(library_tokens, k1=1.5, b=0.75)
        query_tokens = ['zx-410', 'reset']
        library_scores = library_bm25.get_scores(query_tokens)
        ordered_indices = np.argsort(-library_scores, kind='stable')
        print('rank_bm25 results:')
        for index in ordered_indices:
            print(f'  score={library_scores[index]:.3f}  {library_documents[index]}')
        assert ordered_indices[0] == 0
    except ImportError:
        print('rank_bm25 is optional and not installed in this kernel.')
        print('Install requirements-nlp-rag.txt, restart the kernel, and rerun this cell.')
    """),

    md(r"""
    ## 9 · Realistic Case Study — Curriculum Assistant

    **Teaching scenario, not a production benchmark.** The local curriculum assistant
    receives exact-identifier, paraphrased, mixed-intent, control, and unanswerable
    questions. Its structure-aware chunks retain document and section provenance.

    **Measured workflow:**

    1. BM25 and dense LSA each retrieve 9 candidates for a final `top_k` of 3.
    2. RRF or alpha fusion combines the candidate union by stable chunk ID.
    3. If neither branch has meaningful evidence, the system abstains.
    4. Recall, MRR, nDCG, abstention, latency, slices, and component rows are saved.

    **Observed local result:** RRF reaches Recall@3 1.0 versus BM25 0.875 and dense
    LSA 0.75. It preserves abstention on the unanswerable query and improves the
    mixed-intent case without changing the no-gain control. This supports a larger
    evaluation, not immediate deployment. It does not establish neural-embedding
    quality, distributed throughput, business lift, or safety for another corpus.
    """),

    md(r"""
    ## 10 · Production and Safety Considerations

    - **Branch-specific preprocessing.** BM25 tokenization must match its sparse index;
      the dense branch must use the tokenizer expected by its encoder. The branches do
      not need identical tokenization.
    - **Policy before ranking.** Authorization, tenant, freshness, and unsafe-content
      constraints must restrict both branches before fusion. Post-filtering can expose
      forbidden candidates and reduce the final result count.
    - **Stable identity and provenance.** Fuse by immutable chunk ID, retain document,
      source, version, and branch scores, and define deterministic tie-breaking.
    - **Candidate depth.** Retrieve more than final `top_k`; measure recall and latency
      as candidate depth changes.
    - **Abstention.** RRF always ranks supplied candidates. Decide whether either branch
      supplied meaningful evidence before returning them.
    - **Concurrent latency.** Sparse and dense branches usually run concurrently. A
      useful first model is `max(branch latency) + fusion + orchestration`, followed by
      measured end-to-end percentiles. Component p99 values cannot be blindly added.
    - **Index lifecycle.** Update document frequencies with corpus changes, version both
      indexes together, and evaluate before and after migrations.
    - **Monitoring.** Track query slices, zero-result rate, candidate overlap, policy
      filtering, latency, and retrieval quality when delayed labels arrive.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    | Method | Main purpose | Strengths | Weaknesses | Use when |
    |---|---|---|---|---|
    | BM25 | Exact lexical ranking | Transparent, inexpensive, identifier-friendly | Vocabulary mismatch | Exact language dominates and it wins measured slices |
    | Dense | Semantic ranking | Paraphrase and conceptual similarity | Model/index cost; can blur identifiers | Semantic slices justify it |
    | RRF | Rank-based candidate fusion | No raw-score calibration; simple candidate union | Has `k`; discards score magnitude | Labels are limited or score scales differ |
    | Alpha fusion | Weighted normalized scores | Can favor a known branch; uses score magnitude | Needs normalization and labelled tuning | Development data supports a stable alpha |
    | Reranker | Reorder retrieved candidates | Rich query-document interaction | Slower; cannot recover missing evidence | After EVAL-03 confirms candidate recall |

    **Use hybrid when:** query slices need complementary signals, both branches add
    relevant candidates, and measured gains justify cost.

    **Avoid hybrid when:** one branch already meets the target, both branches fail on
    the same corpus gap, policy parity cannot be guaranteed, or the latency/cost budget
    is more valuable than the measured gain.
    """),

    md(r"""
    ## 12 · Readiness and Interview Preparation

    A strong answer to “Should we replace BM25 with vectors?” is:

    1. preserve BM25 as a baseline;
    2. label representative exact, paraphrase, mixed, and unanswerable queries;
    3. compare base retrievers and fusion under identical evaluation conditions;
    4. inspect per-query and slice failures, not only one average;
    5. deploy the simplest method that meets retrieval, policy, latency, and cost goals.

    You are ready to continue when you can explain why raw BM25 and cosine scores
    should not be added, calculate RRF manually, trace one fused result back to both
    candidate lists, and justify a non-hybrid choice when evidence supports it.
    """),

    md(r"""
    ## 13 · Teach-Back — Five Checks

    1. Why can BM25 retrieve an identifier that a dense encoder blurs?
    2. What contribution does a document receive from an RRF list in which it is absent?
    3. Why must candidate depth be larger than the final `top_k`?
    4. Why do policy filters apply before both branch rankings?
    5. Which project evidence would make you keep only one retriever?

    **Answer key:** exact token/IDF signal; zero contribution; fusion cannot recover a
    candidate neither branch supplied; forbidden evidence must never enter ranking;
    one branch meets targets while fusion adds no reliable quality gain worth its cost.
    """),

    md(r"""
    ## 14 · Exercises, Solutions, and Mini Project

    ### Beginner exercise 1 — manual RRF · 10 minutes

    Document A is rank 1 in BM25 and absent from dense. Document B is rank 3 in BM25
    and rank 2 in dense. Use $k=60$. Which ranks first?

    **Expected result:** A = $1/61\approx0.01639$; B = $1/63+1/62\approx0.03200$;
    B ranks first. Common mistake: inventing a dense rank for A.

    ### Beginner exercise 2 — candidate union · 10 minutes

    Sparse IDs are `[a, b, c]`; dense IDs are `[b, d, a]`. Write the candidate set
    and explain why fusion must use IDs rather than text equality.

    **Expected result:** `{a,b,c,d}`. IDs preserve provenance and avoid merging two
    distinct chunks with identical text.

    ### Intermediate exercise 1 — guided alpha calculation · 20 minutes

    Add a function for alpha fusion to the manual string-ID example. **Hint:** normalize
    each branch independently, then score every ID in the union with missing branch
    score zero. Test alpha 0, 0.5, and 1.

    **Self-check:** alpha 0 follows normalized BM25; alpha 1 follows normalized dense;
    every output ID is unique. Common mistakes: normalizing after combining scores and
    silently dropping candidates missing from one branch.

    ### Intermediate exercise 2 — independent failure analysis · 30 minutes

    Run `make hybrid-rag-evaluate`. Select one query whose result differs across
    methods. Record branch candidates, relevant IDs, rank changes, and the smallest
    justified change.

    **Rubric:** 2 points each for evidence trace, correct metric interpretation,
    controlled recommendation, and explicit limitation. Pass: 6/8.

    ### Challenge — policy-safe hybrid retrieval · 45–60 minutes

    Extend a tiny corpus with public, restricted, stale, and unsafe chunks. Apply one
    shared policy predicate before building or querying both branch candidate sets.
    Assert that public search never returns forbidden IDs and authorized search may
    return the restricted current safe ID.

    **Rubric:** 2 points each for pre-fusion filtering, stable-ID deduplication,
    assertions, provenance, and explanation of why post-filtering is insufficient.
    Pass: 8/10.

    ### Mini project — measured hybrid upgrade

    - **Goal:** decide whether hybrid retrieval should replace the best single branch.
    - **Dataset columns:** query ID, query text, slice, relevant section IDs,
      answerable flag; candidate rows include stable section IDs and branch rankings.
    - **Workflow:** predict the outcomes for committed cases `h01`–`h05` → evaluate
      BM25, dense, RRF, and alpha → verify the expected diagnostic behavior → add one
      fresh labelled query from a new domain → inspect slices → recommend one next
      experiment or deployment choice.
    - **Expected output:** a versioned JSON report and a short decision note citing
      Recall@k, MRR, nDCG, abstention, candidate depth, latency limitations, and at
      least two query-level examples.
    - **Evaluation criteria:** reproducible hashes; no invented labels or claims;
      policy-safe candidates; correct metrics; recommendation matches evidence.
    """),

    md(r"""
    ---
    ### Summary

    Hybrid search combines complementary candidate lists; it does not guarantee a
    better system. BM25 preserves exact lexical evidence, dense retrieval can recover
    semantic matches, RRF combines ranks without score calibration, and alpha fusion
    uses normalized scores with a tuned weight. Measure all methods on identical labels,
    preserve policy and provenance in both branches, and keep the simplest winner.

    **Memory aid:** *Retrieve with complementary signals, fuse by stable ID, and keep
    hybrid only when labelled evidence earns the complexity.*

    **Next after mastery:** EVAL-03 evaluates the full RAG system. RAG-07 reranking
    follows only after candidate retrieval has been evaluated.
    """),
]


build("06_rag/06_hybrid_search.ipynb", cells)
