"""Builder for NLP-02 — Sentence Embeddings from scratch."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # NLP-02 · Sentence Embeddings from Scratch
    ### Section 05 — From contextual tokens to retrieval-ready meaning

    NLP-06 built a BERT-style bidirectional encoder. It returns one contextual vector
    per token, but semantic search needs **one comparable vector per sentence**. This
    lesson builds that missing bridge locally: tokenize → encode → pool → normalize →
    train with semantic pairs → evaluate retrieval. No hosted API or downloaded model
    is required.

    **Prerequisites:** NLP-01, DL-07, DL-08, NLP-06, cosine similarity, cross-entropy,
    and train/evaluation separation. **Estimated time:** 5–8 hours with the checkpoint.
    """),
    md(r"""
    ## 1 · Learning Objectives

    After this lesson, you can:

    - explain why token representations are not yet sentence embeddings;
    - implement `[CLS]`, padding-aware mean, and max pooling;
    - build a bi-encoder whose query and document towers share weights;
    - derive cosine similarity, L2 normalization, and MNR/InfoNCE loss;
    - explain temperature, in-batch negatives, hard negatives, and false negatives;
    - compare TF-IDF, an untrained Transformer, and a trained dense encoder fairly;
    - evaluate retrieval with Recall@k, MRR, margin, leakage checks, and slices;
    - decide when sparse retrieval or a cross-encoder is a better tool.
    """),
    md(r"""
    ## 2 · The Problem We Are Trying to Solve

    A support user writes, “Failed login attempts locked my account.” The correct help
    article says, “Verify your identity to unlock a blocked profile.” Keyword overlap
    is incomplete, so exact matching may miss it.

    NLP-01 gave words vectors; NLP-06 gave every token a context-aware vector. Neither
    directly gives one fixed-size point for the whole sentence. Averaging static word
    vectors ignores word order and context. Comparing every query-document token pair
    with a cross-encoder is accurate but too expensive for first-pass search over a
    large corpus. We need a representation that can be computed for documents in
    advance and compared quickly when a query arrives.
    """),
    md(r"""
    ## 3 · What Is the Concept?

    A **sentence embedding** is a fixed-length vector for a variable-length text. A
    useful embedding space places texts with the same meaning near one another and
    unrelated texts farther apart.

    A **bi-encoder** sends a query and a document through the same encoder separately.
    “Same encoder” means shared parameters—not two independently trained models. The
    document vectors can be stored before any query arrives. At search time, encode one
    query and rank stored documents by cosine similarity.

    **Analogy.** Think of organizing books on a map. Pooling writes one location for
    each book; contrastive training moves books about the same problem closer together;
    retrieval finds the nearest shelves. The analogy ends there: vector distance is a
    learned numeric signal, not human understanding.
    """),
    md(r"""
    ## 4 · When Should We Use It?

    Use a sentence bi-encoder for:

    1. semantic search over support articles or product descriptions;
    2. first-stage retrieval for RAG, before optional reranking;
    3. clustering, duplicate detection, or matching short texts at scale.

    Do **not** make it the automatic choice when:

    - exact identifiers and rare keywords dominate—TF-IDF/BM25 may be stronger;
    - the corpus is tiny and maximum pairwise accuracy matters—a cross-encoder may fit;
    - exact sequence generation is required—use an encoder-decoder or causal model;
    - texts exceed the model context—chunk or use a long-context/late-interaction design;
    - you have no representative relevance labels—start with a measured sparse baseline.
    """),
    md(r"""
    ## 5 · How It Works, Step by Step

    1. Tokenize each query and document and create a padding mask.
    2. Run the BERT-style bidirectional encoder from NLP-06.
    3. Pool valid token states into one vector; padding must contribute nothing.
    4. L2-normalize the vector so dot product equals cosine similarity.
    5. Form a batch of matching `(query, document)` pairs.
    6. Build the all-pairs similarity matrix. Its diagonal contains correct matches.
    7. Apply cross-entropy so correct pairs move closer and other candidates move away.
    8. Add carefully checked hard negatives—confusable but genuinely irrelevant texts.
    9. Evaluate once on held-out queries and documents; compare against simpler baselines.

    ```mermaid
    flowchart LR
      Q[query tokens] --> E[shared BERT encoder]
      D[document tokens] --> E
      E --> P[masked pooling]
      P --> N[L2 normalization]
      N --> S[cosine scores]
      S --> R[rank documents]
    ```
    """),
    md(r"""
    ## 6 · Mathematics After the Intuition

    Let token state $\mathbf h_t\in\mathbb R^D$ be the encoder output at position $t$,
    and let mask $m_t$ equal 1 for a real token and 0 for padding. Padding-aware mean
    pooling is

    $$\mathbf s=\frac{\sum_{t=1}^{T}m_t\mathbf h_t}{\sum_{t=1}^{T}m_t}.$$

    Here $T$ is padded sequence length, $D$ is hidden width, and $\mathbf s$ is the
    sentence vector. For scalar token states `[2, 4, 100, 100]` and mask `[1,1,0,0]`,
    the result is $(2+4)/2=3`, not `51.5`.

    L2 normalization is $\hat{\mathbf s}=\mathbf s/\|\mathbf s\|_2$. Cosine is

    $$\cos(\mathbf a,\mathbf b)=\frac{\mathbf a^\top\mathbf b}
      {\|\mathbf a\|_2\|\mathbf b\|_2}=\hat{\mathbf a}^\top\hat{\mathbf b}.$$

    For $\mathbf a=(3,4)$, the norm is 5 and $\hat{\mathbf a}=(0.6,0.8)$. For
    $\mathbf b=(0,5)$, $\hat{\mathbf b}=(0,1)$, so cosine is `0.8`.
    """),
    md(r"""
    ### Multiple Negatives Ranking loss

    For batch pairs $(q_i,d_i)$, create score $z_{ij}=\cos(q_i,d_j)/\tau$. The correct
    document for query row $i$ is column $i$:

    $$J=-\frac1B\sum_{i=1}^{B}\log
      \frac{\exp(z_{ii})}{\sum_{j=1}^{B}\exp(z_{ij})}.$$

    $B$ is batch size and $\tau>0$ is temperature. Smaller $\tau$ sharpens score
    differences and multiplies logit sensitivity by $1/\tau$, but it can also produce
    saturated probabilities or unstable updates. Temperature is validated, not assumed.
    Off-diagonal documents become **in-batch negatives**.
    This is efficient, but a mislabeled off-diagonal relevant document becomes a
    **false negative** and teaches the wrong geometry.

    An explicit hard negative $n_i$ is handled per query in this project. We add a
    softplus penalty:

    $$
    \ell_{hard,i}=\operatorname{softplus}\left(
    \frac{\cos(q_i,n_i)-\cos(q_i,d_i)}{\tau}
    \right).
    $$

    If the negative outranks the positive, the argument is positive and the penalty
    grows. Softplus remains above zero even when ordering is correct, so compare values
    rather than expecting an exact zero. Scoring every hard negative against every
    query can create accidental false negatives.
    """),
    code(r"""
    # Verify masked pooling and normalization with a batch-shaped example.
    import numpy as np

    token_states = np.array([[[2.0, 0.0], [4.0, 2.0], [100.0, 100.0], [100.0, 100.0]]])
    padding_mask = np.array([[1.0, 1.0, 0.0, 0.0]])
    mask_with_feature_axis = padding_mask[..., None]
    pooled = (token_states * mask_with_feature_axis).sum(axis=1) / mask_with_feature_axis.sum(axis=1)
    normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    print("pooled vector:", pooled)
    print("normalized vector:", normalized)
    print("normalized norm:", np.linalg.norm(normalized, axis=1))
    assert np.allclose(pooled, [[3.0, 1.0]])
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    """),
    code(r"""
    import numpy as np

    # Manual three-pair similarity matrix. Rows are queries; columns are documents.
    query_vectors = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    document_vectors = np.array([[.9, .1], [.1, .9], [-.8, .2]])
    document_vectors /= np.linalg.norm(document_vectors, axis=1, keepdims=True)
    similarity_matrix = query_vectors @ document_vectors.T
    print(np.round(similarity_matrix, 3))
    print("Correct scores are diagonal:", np.round(np.diag(similarity_matrix), 3))

    temperature = 0.2
    logits = similarity_matrix / temperature
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    row_probabilities = np.exp(shifted_logits) / np.exp(shifted_logits).sum(axis=1, keepdims=True)
    correct_probabilities = row_probabilities[np.arange(len(row_probabilities)), np.arange(len(row_probabilities))]
    losses = -np.log(correct_probabilities)
    print("\nrow probabilities:\n", np.round(row_probabilities, 3))
    print("correct-match probabilities:", np.round(correct_probabilities, 3))
    print("per-query losses:", np.round(losses, 3))
    print("mean MNR loss:", round(float(losses.mean()), 3))
    assert np.allclose(row_probabilities.sum(axis=1), 1.0)
    """),
    md(r"""
    ## 7 · Python Implementation from Scratch

    The project keeps each responsibility visible:

    - `tokenizer.py`: lowercase word tokenization, `[UNK]`, `[CLS]`, padding, mask;
    - `model.py`: shared NLP-06 encoder, pooling, normalization, MNR loss;
    - `data.py`: explicit train/evaluation text and confusable intent mapping;
    - `evaluation.py`: TF-IDF and ranking metrics implemented directly;
    - `training.py`: optimizer loop, baselines, held-out evaluation, artifacts.

    The first executable example uses actual text and actual gradient updates. It does
    not inject hand-designed semantic directions or simulate Transformer outputs.
    """),
    code(r"""
    import sys
    from pathlib import Path

    candidates = [Path.cwd(), *Path.cwd().parents]
    repo_root = next(path for path in candidates if (path / "projects/sentence_embeddings").exists())
    sentence_root = repo_root / "projects" / "sentence_embeddings"
    family_root = repo_root / "projects" / "transformer_families"
    sys.path.insert(0, str(sentence_root / "src"))
    sys.path.insert(0, str(family_root / "src"))

    from sentence_embeddings.data import EXAMPLES, assert_no_text_leakage
    from sentence_embeddings.training import run_experiment

    assert_no_text_leakage()
    print("Intents:", [row.intent for row in EXAMPLES])
    print("Held-out example:", EXAMPLES[0].evaluation_query)
    """),
    code(r"""
    report = run_experiment(seed=42, steps=320)
    print("Initial loss:", round(report["training"]["initial_loss"], 4))
    print("Final loss:  ", round(report["training"]["final_loss"], 4))
    for name in ["tfidf_baseline", "untrained_transformer", "trained_contrastive_encoder"]:
        metrics = report[name]
        print(f"{name:28s} R@1={metrics['recall_at_1']:.3f} "
              f"R@3={metrics['recall_at_3']:.3f} MRR={metrics['mrr']:.3f} "
              f"margin={metrics['mean_positive_negative_margin']:.3f}")
    print("CPU median ms/text:", round(report["latency"]["median_encoding_ms_per_text"], 3))
    print("Per-intent ranks:", {name: row["rank"] for name, row in report["evaluation_slices"].items()})
    """),
    md(r"""
    ## 8 · Reading the Retrieval Evidence

    `run_experiment` first evaluates random encoder weights. It then fits TF-IDF only
    for comparison, trains the dense encoder, and evaluates all systems against the
    same held-out query-to-document mapping.

    Expected deterministic result for seed 42:

    | System | Recall@1 | Recall@3 | MRR | Mean margin |
    |---|---:|---:|---:|---:|
    | TF-IDF | 0.250 | 0.625 | 0.476 | -0.060 |
    | Untrained Transformer | 0.000 | 0.500 | 0.332 | -0.086 |
    | Trained contrastive encoder | 0.875 | 1.000 | 0.917 | 0.477 |

    Recall@1 asks whether the correct document ranks first. MRR rewards moving it
    toward the top. Positive margin means the correct score beats the strongest wrong
    score on average. Loss reduction alone is insufficient; held-out retrieval must
    improve. Because there are only eight queries, one error changes Recall@1 by 0.125.
    """),
    code(r"""
    import matplotlib.pyplot as plt

    names = ["TF-IDF", "Untrained", "Contrastive"]
    keys = ["tfidf_baseline", "untrained_transformer", "trained_contrastive_encoder"]
    recall1 = [report[key]["recall_at_1"] for key in keys]
    mrr = [report[key]["mrr"] for key in keys]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - .18, recall1, .36, label="Recall@1")
    ax.bar(x + .18, mrr, .36, label="MRR")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out retrieval score")
    ax.set_title("Training must beat the untrained encoder on unseen pairs")
    ax.legend()
    plt.show()
    """),
    md(r"""
    ## 9 · Pooling Choices

    | Pooling | Main idea | Strength | Weakness | Use when |
    |---|---|---|---|---|
    | `[CLS]` | use first special-token state | simple; can be trained as summary | raw `[CLS]` is not guaranteed semantic | objective explicitly trains it |
    | Mean | average valid token states | stable default; all tokens contribute | can dilute rare decisive tokens | sentence similarity and retrieval |
    | Max | keep largest value per dimension | preserves strong activations | noisy; discards frequency and balance | evidence is sparse and validated |

    Pooling is not magic. The training objective determines whether the pooled vector
    has useful geometry. Never average padding, and never claim one strategy is best
    without a held-out comparison.
    """),
    md(r"""
    ## 10 · Practical Library Version

    After understanding the manual system, a production experiment can use a pinned,
    local `sentence-transformers` checkpoint:

    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_MODEL_PATH, local_files_only=True)
    vectors = model.encode(texts, normalize_embeddings=True)
    ```

    This course does not download a checkpoint during the core lesson. A library model
    is a candidate, not an answer: record its exact revision, dataset, pooling, maximum
    length, embedding dimension, latency, memory, and held-out retrieval metrics.
    Hosted embeddings are optional controlled comparisons because privacy, cost, model
    revisions, and network availability change the experiment.
    """),
    md(r"""
    ## 11 · Common Beginner Mistakes and Debugging

    | Symptom | Likely cause | Check and repair |
    |---|---|---|
    | Same sentence changes with extra padding | mask omitted during attention or pooling | run padding-invariance test |
    | Dot products depend mainly on vector length | embeddings not normalized | assert every norm is near 1 |
    | Training loss falls but retrieval does not | overfit or leakage | evaluate held-out pairs; inspect overlap |
    | Loss cannot approach its expected floor | false/duplicate negatives | inspect every similarity-matrix label |
    | All similarities are high | collapsed or anisotropic space | inspect off-diagonal distribution and margins |
    | Hard negatives hurt quality | they are actually relevant | relabel with domain review; do not trust mining blindly |
    | TF-IDF beats dense retrieval | small data, exact vocabulary, or poor training | keep TF-IDF; gather better pairs before scaling |

    Five especially common conceptual errors are: calling raw BERT states sentence
    embeddings; confusing bi-encoder and cross-encoder; fitting on evaluation queries;
    reporting only training loss; and assuming “hard” means “randomly difficult” rather
    than “confusable but irrelevant.”
    """),
    md(r"""
    ## 12 · Comparison with Related Concepts

    | Concept | Main purpose | Strengths | Weaknesses | When to use |
    |---|---|---|---|---|
    | TF-IDF/BM25 | lexical retrieval | fast, inspectable, rare terms | misses paraphrases | exact terms and strong baseline |
    | Word-vector average | sentence baseline | cheap | weak context/order | educational baseline |
    | Sentence bi-encoder | dense retrieval | precomputable, semantic | compresses interaction | first-stage search |
    | Cross-encoder | pair relevance | token-level interaction | full model pass per pair | rerank small candidate set |
    | Late interaction | token-vector retrieval | more detailed matching | larger index/complexity | quality between bi/cross encoder |

    Hybrid retrieval often combines sparse and dense evidence rather than declaring a
    universal winner. RAG-01 teaches the vector index; RAG-06 later measures fusion.
    """),
    md(r"""
    ## 13 · Real-World Use Case

    A support platform indexes approved help articles offline. Each incoming question
    is embedded once, nearest articles are retrieved, and an optional cross-encoder
    reranks only the top candidates. The team evaluates by issue type because an
    overall average can hide weak refund or account-access performance.

    Release evidence includes Recall@10, MRR, per-intent slices, no-answer queries,
    p50/p95 encoding latency, index size, privacy review, and a pinned model revision.
    A false negative hides the correct help article; a false positive wastes user time
    or may surface an unsafe policy. The retrieval threshold must be tuned on validation
    data, not guessed from a cosine number.
    """),
    md(r"""
    ## 14 · Student Check

    1. Why must mean pooling multiply by the padding mask before summing?
    2. Why does the MNR target for row `i` normally equal column `i`?
    3. What changes when temperature becomes smaller, and why can “smaller” become harmful?
    4. Why can TF-IDF outperform a dense model on identifiers or exact product names?
    5. What evidence distinguishes memorizing training pairs from learning useful geometry?

    Do not continue if you cannot draw the `B × B` similarity matrix and explain every
    cell without notes.
    """),
    md(r"""
    ## 15 · Practice Exercises

    **Beginner**

    1. Manually pool token values `[1,3,7,99]` with mask `[1,1,1,0]`.
    2. Normalize vectors `(3,4)` and `(5,0)`, then calculate their cosine.

    **Intermediate**

    3. Add `[CLS]` and max pooling to the experiment and compare held-out MRR.
    4. Remove hard-negative penalties, retrain with three seeds, and report mean/std.

    **Challenge**

    5. Add a deliberately false negative, inspect which loss term increases, then
       design a label rule or mask that prevents the error.
    """),
    md(r"""
    ## 16 · Mini Project — Local Support-Article Retriever

    **Goal:** retrieve the correct support article for unseen user wording without an
    API key.

    **Dataset columns:** `query_id`, `query_text`, `document_id`, `document_text`,
    `relevant`, `split`, `intent`, `negative_source`.

    **Workflow:** freeze the split → audit duplicates → build TF-IDF → train bi-encoder
    → mine and review hard negatives → evaluate overall and per intent → measure latency
    → write data/model cards.

    **Expected output:** versioned model weights, vocabulary, retrieval report, ranked
    error examples, and a reproducible command.

    **Evaluation:** no leakage; Recall@k and MRR improve over the untrained encoder;
    comparison with TF-IDF is honest; padding/normalization tests pass; every hard
    negative is reviewed; limitations are documented.
    """),
    md(r"""
    ## 17 · Summary

    Sentence embeddings are trained representations, not an automatic property of raw
    Transformer states. A bi-encoder shares one encoder across queries and documents,
    pools only valid tokens, normalizes vectors, and learns geometry from positive and
    negative pairs. MNR makes a whole batch useful, but false negatives can teach the
    wrong relationship. Success is held-out retrieval improvement over sparse and
    untrained baselines—not falling training loss.

    **Memory aid:** *Encode separately, pool with the mask, normalize, contrast, then
    trust only held-out retrieval.*

    **Next canonical lesson:** NLP-03 follows the complete LLM pretraining and data
    lifecycle. RAG-01 later uses normalized vectors for exact and approximate similarity
    search. Complete `projects/sentence_embeddings/MASTERY_CHECKPOINT.md` first.
    """),
]

build("05_nlp_and_llms/02_sentence_embeddings.ipynb", cells)
