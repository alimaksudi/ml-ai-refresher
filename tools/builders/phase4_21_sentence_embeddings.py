"""Builder for Notebook 21 — Sentence Embeddings.

Run:  python3 tools/builders/phase4_21_sentence_embeddings.py
Emits: notebooks/phase4_nlp_llms/21_sentence_embeddings.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # 21 · Sentence Embeddings
    ### Phase 4 — Modern NLP and LLMs · *ML/AI Senior Mastery Curriculum*

    > Notebook 20 gave every *word* a dense vector. This notebook lifts that to the
    > *sentence* level: how do we get a single fixed-length vector that captures the
    > meaning of a full sentence, paragraph, or document? The answer evolved from naive
    > averaging of word vectors → BERT [CLS] pooling → **sentence-transformers** (bi-
    > encoders trained with contrastive loss on NLI/STS data). These vectors are the
    > foundation of every modern semantic search system and RAG pipeline (Phase 5).
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - Why word-embedding averages fail for sentences and how mean pooling compares to
      [CLS] pooling.
    - The **bi-encoder architecture**: encode query and document independently; score
      by cosine similarity. Fast at inference — the key property for retrieval.
    - The **cross-encoder architecture**: encode (query, document) jointly; output a
      relevance scalar. Slower but more accurate — the key property for reranking.
    - **Contrastive learning**: how sentence-transformers (SBERT, Reimers & Gurevych
      2019) fine-tune a BERT encoder on NLI/STS pairs with a triplet/cosine loss.
    - How to build a **semantic search engine from scratch** using cosine similarity
      over sentence embeddings.
    - Evaluation: STS benchmarks, MRR, and BEIR.

    **Why it matters**
    - Sentence embeddings are the input to every RAG pipeline (Notebooks 25–30). A
      bad embedding model → bad retrieval → bad generation, no matter how good the LLM.

    **Typical interview questions**
    - "Bi-encoder vs cross-encoder — when do you use each?"
    - "Why can't you just average word2vec vectors for sentence similarity?"
    - "What is contrastive loss and why does it work for embeddings?"
    - "How would you evaluate a sentence embedding model?"
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Averaging word vectors.** The simplest sentence embedding: mean of the word2vec
    vectors in the sentence. Surprisingly competitive on some tasks but ignores
    word order and sentence-level composition ("not bad" ≠ "bad" + "not" / 2).

    **Universal Sentence Encoder (Cer et al., 2018)** and **InferSent (Conneau et al.,
    2017)** showed that training an encoder *specifically* to produce sentence
    representations (via NLI tasks) gave far better semantic similarity scores.

    **BERT (Devlin et al., 2018)** changed everything: its [CLS] token after
    fine-tuning on NLI tasks produced strong sentence embeddings. But naively using
    the BERT [CLS] or mean-pooled output from a pre-trained (unfine-tuned) BERT is
    *worse than averaging GloVe vectors* on STS benchmarks — the Transformer
    encodes position and context, not sentence-level semantics, unless explicitly
    trained to.

    **Sentence-BERT (Reimers & Gurevych, 2019)** solved this: fine-tune a Siamese
    (twin) BERT on Natural Language Inference and Semantic Textual Similarity data
    with a **cosine similarity loss**. The result: a bi-encoder that maps sentences to
    a 768d space where cosine similarity = semantic similarity. Inference: 65ms for
    10K sentences (vs 65 hours with cross-encoder BERT). This made dense retrieval
    practical.

    **Today.** `all-MiniLM-L6-v2`, `bge-m3`, `text-embedding-3-small` (OpenAI), and
    `Voyage-2` are the production defaults. They all follow the same bi-encoder recipe
    with improved data, training objectives (SimCSE, E5, GTE), and architectures.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The key idea: geometry = semantics.** A sentence embedding model maps any
    sentence to a point in $\mathbb{R}^d$ such that the angle between two points
    (cosine similarity) measures their semantic relatedness. "The puppy chased the
    ball" and "A dog ran after a sphere" should be nearby; "The stock market fell"
    should be far away.

    **Bi-encoder vs cross-encoder.**
    - **Bi-encoder:** encode query $q$ and each document $d_i$ *independently* into
      vectors $\mathbf{q}, \mathbf{d}_i$; rank by $\cos(\mathbf{q}, \mathbf{d}_i)$.
      Documents can be pre-encoded and stored in an index. Query encoding takes
      microseconds at inference. This is the *retrieval* step.
    - **Cross-encoder:** feed $(q, d_i)$ *jointly* into BERT with a [SEP] separator;
      predict a relevance score from the [CLS] token. Can see interactions between
      query and document tokens — much more accurate but cannot pre-encode. This is the
      *reranking* step (Notebook 28).

    ```mermaid
    flowchart LR
        Q["query text"] --> BiEncQ["bi-encoder\n(BERT/MiniLM)"]
        D["document text"] --> BiEncD["bi-encoder\n(same weights)"]
        BiEncQ --> QVec["q_vec (768d)"]
        BiEncD --> DVec["d_vec (768d)"]
        QVec --> Cos["cosine sim\n(fast, indexable)"]
        DVec --> Cos
        Cos --> Top100["top-100 candidates"]
        Top100 --> XEnc["cross-encoder\n(q,d pair jointly)"]
        XEnc --> Score["relevance score\n(accurate, slow)"]
    ```

    **Contrastive learning intuition.** Show the model pairs: (anchor, positive) =
    semantically similar, (anchor, negative) = semantically different. Loss pushes
    positives close and negatives far. After enough pairs, the geometry self-organises
    into a semantic map.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    import re, math

    rng = np.random.default_rng(42)
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    def cosine_sim(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n > 0 else 0.0

    SENTENCES = [
        "The cat sat on the mat",              # 0
        "A cat rested on a rug",               # 1  paraphrase of 0
        "The dog chased the ball",             # 2
        "A puppy ran after a sphere",          # 3  paraphrase of 2
        "Machine learning models train on data", # 4
        "Neural networks learn from examples",  # 5  paraphrase of 4
        "The stock market fell sharply",        # 6
        "Shares declined on the exchange",      # 7  paraphrase of 6
        "I love sunny weather",                 # 8
        "The weather is great today",           # 9  paraphrase of 8
    ]

    PAIRS = [(0,1),(2,3),(4,5),(6,7),(8,9)]    # expected similar pairs
    print(f"Sentences: {len(SENTENCES)},  Expected similar pairs: {len(PAIRS)}")
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Mean pooling

    Given a sentence of $T$ tokens with Transformer output states
    $\mathbf{h}_1,\dots,\mathbf{h}_T \in \mathbb{R}^d$:
    $$\mathbf{s} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{h}_t$$
    Simple and effective; masks out padding tokens when computing the mean.

    ### 4.2 Cosine similarity
    $$\text{cos}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}$$
    Range $[-1, 1]$; 1 = identical direction; 0 = orthogonal; −1 = opposite.
    Equivalent to dot product after L2 normalisation — which is why production
    systems often pre-normalise vectors and use inner product (faster on GPU/FAISS).

    ### 4.3 Contrastive / triplet loss

    **Triplet loss:** for each (anchor $a$, positive $p$, negative $n$):
    $$J_{\text{triplet}} = \max\!\left(0,\; d(a,p) - d(a,n) + \text{margin}\right)$$
    where $d$ is Euclidean or cosine distance. Loss is zero when the positive is at
    least `margin` closer than the negative. Pushes the model to maintain a margin.

    **Multiple Negatives Ranking (MNR) loss** (Henderson et al.; used in SBERT-v2):
    treat all *other* positives in the batch as negatives. For a batch of $B$ pairs
    $(a_i, p_i)$:
    $$J_{\text{MNR}} = -\frac{1}{B}\sum_i \log \frac{\exp(\cos(a_i,p_i)/\tau)}
        {\sum_j \exp(\cos(a_i,p_j)/\tau)}$$
    This is cross-entropy over the $B$ positives with temperature $\tau$. Large batches
    = more negatives = better training. **In-batch negatives** make this highly
    efficient.

    ### 4.4 STS evaluation (Pearson/Spearman $\rho$)

    STS benchmarks provide sentence pairs with human similarity ratings $[0,5]$.
    Evaluation: compute model cosine similarities, then Spearman $\rho$ between model
    scores and human scores. State-of-the-art on STS-B: $\rho \approx 0.92$.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    ### 5a Mean pooling over toy word vectors
    """),

    code(r"""
    # 5a. Toy word embeddings (random but fixed — imagine pre-trained word2vec).
    # In a real system these would be the outputs of a Transformer layer.
    VOCAB = sorted(set(w.lower() for s in SENTENCES for w in re.findall(r"[a-z]+", s.lower())))
    V = len(VOCAB); D = 32
    w2i = {w: i for i, w in enumerate(VOCAB)}

    # Simulate "pre-trained" word vectors: similar words close together via seeded RNG.
    # We cheat slightly: words that appear in similar sentences share a common base.
    word_vecs = rng.normal(0, 1, (V, D))

    # Give animal-related and sport-related words a common direction bias.
    animal_dir = rng.normal(0, 1, D); animal_dir /= np.linalg.norm(animal_dir)
    sport_dir  = rng.normal(0, 1, D); sport_dir  /= np.linalg.norm(sport_dir)
    finance_dir= rng.normal(0, 1, D); finance_dir/= np.linalg.norm(finance_dir)
    weather_dir= rng.normal(0, 1, D); weather_dir/= np.linalg.norm(weather_dir)
    ml_dir     = rng.normal(0, 1, D); ml_dir     /= np.linalg.norm(ml_dir)

    for w, bias in [
        ("cat", animal_dir), ("dog", animal_dir), ("puppy", animal_dir),
        ("mat", animal_dir), ("rug", animal_dir),
        ("ball", sport_dir), ("sphere", sport_dir), ("chased", sport_dir), ("ran", sport_dir),
        ("stock", finance_dir), ("market", finance_dir), ("shares", finance_dir),
        ("declined", finance_dir), ("exchange", finance_dir),
        ("sunny", weather_dir), ("weather", weather_dir),
        ("machine", ml_dir), ("learning", ml_dir), ("neural", ml_dir), ("networks", ml_dir),
    ]:
        if w in w2i:
            word_vecs[w2i[w]] += 3.0 * bias    # strong domain signal

    def mean_pool(sentence):
        tokens = [w.lower() for w in re.findall(r"[a-z]+", sentence.lower()) if w.lower() in w2i]
        if not tokens:
            return np.zeros(D)
        return np.mean([word_vecs[w2i[t]] for t in tokens], axis=0)

    sentence_vecs = np.array([mean_pool(s) for s in SENTENCES])
    # L2-normalise for cosine via dot product.
    norms = np.linalg.norm(sentence_vecs, axis=1, keepdims=True) + 1e-9
    sentence_vecs_norm = sentence_vecs / norms

    print("Sentence embedding shape:", sentence_vecs_norm.shape)
    print("\nCosine similarities for expected similar pairs:")
    for a, b in PAIRS:
        sim = cosine_sim(sentence_vecs_norm[a], sentence_vecs_norm[b])
        print(f"  [{a}] '{SENTENCES[a][:30]}' vs [{b}] '{SENTENCES[b][:30]}': {sim:.3f}")
    """),

    md(r"""
    ### 5b Semantic search: top-k retrieval by cosine similarity
    """),

    code(r"""
    # 5b.1 Build a tiny "index": pre-encode all sentences.
    # This is exactly what FAISS/pgvector does at scale.
    INDEX = sentence_vecs_norm         # (N, D) pre-encoded

    def semantic_search(query, top_k=3):
        q_vec = mean_pool(query)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        sims = INDEX @ q_vec           # dot product = cosine sim after normalisation
        ranked = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in ranked]

    queries = ["cat resting on rug", "finance news", "machine learning"]
    for q in queries:
        results = semantic_search(q)
        print(f"Query: '{q}'")
        for idx, score in results:
            print(f"  [{idx}] {SENTENCES[idx]}  sim={score:.3f}")
        print()
    """),

    md(r"""
    ### 5c Contrastive loss (Multiple Negatives Ranking) from scratch
    """),

    code(r"""
    # 5c. MNR loss: for a batch of (anchor, positive) pairs,
    # treat other positives as in-batch negatives.
    def softmax_1d(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def mnr_loss(anchors, positives, tau=0.05):
        # anchors, positives: (B, D) L2-normalised
        B = len(anchors)
        sim_matrix = anchors @ positives.T    # (B, B) all-pairs cosine sim
        sim_matrix /= tau
        # Diagonal entries are the positives; off-diagonal are negatives.
        # Cross-entropy: for each row, maximise the diagonal entry.
        total_loss = 0.0
        for i in range(B):
            log_probs = sim_matrix[i] - np.log(np.sum(np.exp(sim_matrix[i] - sim_matrix[i].max())) + 1e-9) - sim_matrix[i].max()
            total_loss += -log_probs[i]
        return total_loss / B

    # Build a batch of (anchor, positive) pairs.
    anchors  = sentence_vecs_norm[[0, 2, 4, 6, 8]]   # one per pair
    positives= sentence_vecs_norm[[1, 3, 5, 7, 9]]
    loss_val = mnr_loss(anchors, positives)
    print(f"MNR loss on perfectly ordered embeddings: {loss_val:.4f}")

    # Shuffle positives to simulate untrained model.
    shuffled = positives[rng.permutation(len(positives))]
    loss_shuffled = mnr_loss(anchors, shuffled)
    print(f"MNR loss on shuffled (wrong) pairs:       {loss_shuffled:.4f}")
    print("Lower is better — the trained model should approach near-zero loss.")
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Cosine similarity heatmap between all sentence pairs.
    sim_matrix = sentence_vecs_norm @ sentence_vecs_norm.T
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax)
    labels = [f"{i}: {s[:20]}..." for i, s in enumerate(SENTENCES)]
    ax.set_xticks(range(len(SENTENCES))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(SENTENCES))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Figure 1 — Cosine similarity matrix (mean-pooled toy word vectors)")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** The cosine similarity matrix between all 10 sentences. Expected
    similar pairs (0–1, 2–3, 4–5, 6–7, 8–9) appear as bright green blocks near the
    diagonal. Unrelated sentence pairs should be near zero or negative. On this toy
    corpus with our biased word vectors, the paraphrase pairs cluster together and
    cross-topic pairs stay separated — demonstrating that even simple mean pooling over
    domain-biased word vectors captures some semantic structure. Real sentence
    embeddings (SBERT, BGE) produce much sharper separation via supervised contrastive
    training.
    """),

    code(r"""
    # Figure 2 — 2D PCA of sentence embeddings, coloured by topic.
    from numpy.linalg import svd

    def pca2d(X):
        X = X - X.mean(0)
        U, S, Vt = svd(X, full_matrices=False)
        return X @ Vt[:2].T

    emb2d = pca2d(sentence_vecs_norm)
    colours = ["red","red","blue","blue","green","green","orange","orange","purple","purple"]
    topics = ["animals","animals","sports","sports","ML","ML","finance","finance","weather","weather"]

    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for i, (x, y) in enumerate(emb2d):
        label = topics[i] if topics[i] not in seen else None
        ax.scatter(x, y, c=colours[i], s=80, label=label, zorder=3)
        ax.annotate(f"[{i}] {SENTENCES[i][:20]}", (x, y), fontsize=7.5, alpha=0.85)
        seen.add(topics[i])

    ax.set_title("Figure 2 — PCA of sentence embeddings: topic clusters emerge")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** PCA projects the 32-dimensional sentence embeddings onto 2D. Same-
    colour dots (same topic) cluster together: the biased word vectors propagate topic
    signal through mean pooling, so the sentence embeddings inherit the geometry. In a
    trained SBERT model this separation is much sharper because the contrastive
    objective directly optimises for paraphrase pairs to be co-located and non-
    paraphrases to be separated — not just inheriting word-vector topic bias.
    """),

    code(r"""
    # Figure 3 — Effect of pooling strategy: [CLS] vs mean vs max.
    # Simulate with random token outputs (as if from a Transformer).
    T = 8                                       # sequence length
    H = rng.normal(0, 1, (T, D))               # fake Transformer hidden states
    cls_pool  = H[0]                            # [CLS] token (position 0)
    mean_pool_H = H.mean(axis=0)
    max_pool  = H.max(axis=0)

    methods = {"[CLS] pool": cls_pool, "mean pool": mean_pool_H, "max pool": max_pool}
    norms = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in methods.items()}

    print("Cosine similarities between pooling strategies (same token sequence):")
    keys = list(norms.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            sim = cosine_sim(norms[keys[i]], norms[keys[j]])
            print(f"  {keys[i]} vs {keys[j]}: {sim:.3f}")
    print("\nKey insight: the three strategies produce *different* vectors from identical")
    print("input. Supervised fine-tuning determines which strategy works best for semantic")
    print("similarity — empirically, mean pooling on SBERT >= [CLS] on raw BERT.")
    """),

    md(r"""
    **Figure 3 (output).** The three pooling strategies — [CLS], mean, max — produce
    different sentence vectors from *identical* Transformer hidden states. For a raw
    pre-trained BERT (not fine-tuned for sentence similarity), Reimers & Gurevych (2019)
    found that mean pooling slightly outperforms [CLS] on STS benchmarks, and that
    *fine-tuning* on NLI/STS data matters far more than the pooling choice. Max pooling
    captures the most "activated" feature per dimension, which can help for tasks with
    sparse, important keywords but hurts for semantic similarity.
    """),

    code(r"""
    # Figure 4 — MNR loss landscape: how temperature tau affects separation.
    taus = np.logspace(-2, 0, 50)           # 0.01 to 1.0
    losses = []
    for tau in taus:
        losses.append(mnr_loss(anchors, positives, tau=tau))

    fig, ax = plt.subplots()
    ax.semilogx(taus, losses)
    ax.set_xlabel("temperature tau (log scale)"); ax.set_ylabel("MNR loss")
    ax.set_title("Figure 4 — MNR loss vs temperature: lower tau -> sharper separation")
    ax.axvline(0.05, color="red", ls="--", label="tau=0.05 (SimCSE default)")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 4.** The MNR loss as a function of temperature $\tau$. At low $\tau$
    (sharper distribution), the model is forced to distinguish similar from dissimilar
    with high confidence — harder training but tighter clusters. At high $\tau$ (softer
    distribution), the model sees all pairs as nearly equally likely, giving near-zero
    gradient signal. The default $\tau=0.05$ (SimCSE, SBERT) is at the steep part of
    this curve: training is numerically challenging but produces the tightest clusters.
    This is why sentence embedding training is sensitive to temperature and batch size
    (larger batches = more hard negatives = better training).
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Domain mismatch** | Finance queries return poor results on general embedding | Training data was general web text | Fine-tune on domain pairs (MTEB domain) |
    | **Anisotropy collapse** | All embeddings cluster near one direction | Pre-trained BERT occupies a narrow cone | Whitening transform; SIMCSE; fine-tune |
    | **Pooling on long docs** | First/last tokens dominate meaning | Mean pool gives equal weight to stopwords | CLS-first, or chunk + pool, or late-interaction (ColBERT) |
    | **Cross-encoder at scale** | 10ms/pair × 1M docs = too slow | No pre-encoding possible | Bi-encoder for retrieval; cross-encoder for top-100 reranking only |
    | **Semantic drift** | "LLM" means different things over time | Static embeddings | Periodic re-indexing; monitor query-index similarity |
    | **Negative collapse** | Bi-encoder learns to ignore negatives | Easy negatives in batch | Hard negative mining (BM25 negatives; in-batch mixing) |
    | **Language mismatch** | English queries miss Spanish documents | Monolingual model | Multilingual model (LaBSE, mE5, mGTE) |
    """),

    code(r"""
    # Demonstrate: anisotropy — raw BERT-style embeddings cluster in a narrow cone.
    raw_embs = rng.normal(0, 1, (100, D))              # random baseline (isotropic)
    # Simulate anisotropic pre-trained BERT: all vecs biased toward one direction.
    aniso_dir = rng.normal(0, 1, D)
    aniso_embs = rng.normal(0, 0.5, (100, D)) + 3.0 * aniso_dir

    def avg_cosine(E):
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sims = E @ E.T
        # Exclude diagonal
        idx = np.triu_indices(len(E), k=1)
        return float(sims[idx].mean())

    print(f"Average cosine sim (isotropic random): {avg_cosine(raw_embs):.3f}  <- near 0")
    print(f"Average cosine sim (anisotropic BERT):  {avg_cosine(aniso_embs):.3f}  <- near 1")
    print("Consequence: anisotropic space compresses all cosine similarities into [0.8, 1.0]")
    print("-> cannot distinguish 'very similar' from 'somewhat similar'.")
    print("Fix: whitening (subtract mean, decorrelate) or fine-tune with contrastive loss.")
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 sentence-transformers: the production standard.
    try:
        from sentence_transformers import SentenceTransformer, util

        model_st = SentenceTransformer("all-MiniLM-L6-v2")  # 22M params, 384d, fast
        embeddings = model_st.encode(SENTENCES, normalize_embeddings=True, show_progress_bar=False)
        print(f"sentence-transformers embedding shape: {embeddings.shape}")

        # Semantic search.
        q_emb = model_st.encode(["cat resting"], normalize_embeddings=True)
        sims = (embeddings @ q_emb.T).squeeze()
        ranked = np.argsort(sims)[::-1][:3]
        print("\nSemantic search for 'cat resting' (MiniLM):")
        for i in ranked:
            print(f"  [{i}] {SENTENCES[i]}  sim={sims[i]:.3f}")

        # STS similarity matrix.
        cos_scores = util.cos_sim(embeddings, embeddings).numpy()
        print(f"\nAll-pairs cosine sim (5 expected pairs, diag excluded):")
        for a, b in PAIRS:
            print(f"  [{a}] vs [{b}]: {cos_scores[a,b]:.3f}")

    except Exception as e:
        print(f"[sentence-transformers not available: {type(e).__name__}]")
        print("In production:")
        print("  from sentence_transformers import SentenceTransformer")
        print("  model = SentenceTransformer('all-MiniLM-L6-v2')  # or bge-m3, e5-large")
        print("  embs  = model.encode(sentences, normalize_embeddings=True)")
        print("  sim   = embs @ embs.T   # cosine via dot after L2-norm")
    """),

    code(r"""
    # 8.2 OpenAI embeddings (guarded — requires API key).
    print("OpenAI embeddings — production pattern (guarded):")
    print("  from openai import OpenAI")
    print("  client = OpenAI()")
    print("  resp = client.embeddings.create(")
    print("    input=sentences, model='text-embedding-3-small')  # 1536d, $0.02/1M tokens")
    print("  embs = np.array([r.embedding for r in resp.data])")
    print()
    print("Key tradeoffs:")
    print("  - Proprietary: data leaves your infra")
    print("  - No fine-tuning control")
    print("  - High quality but ongoing cost at scale")
    print("  Prefer open-source (MiniLM, BGE, E5) for cost-sensitive production.")
    """),

    md(r"""
    **Scratch vs production.** Our mean-pooling over toy word vectors demonstrates the
    *principle* but skips three things that matter enormously in practice: (1) the
    Transformer hidden states that capture context within the sentence; (2) the
    contrastive fine-tuning that aligns the geometry with human semantic similarity
    judgements; (3) the FAISS/pgvector index for sub-millisecond retrieval over millions
    of documents. `sentence-transformers` gives you 1 and 2 in three lines; Phase 5
    (Notebooks 25–30) covers 3.
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Semantic Job Matching

    **Scenario.** A recruiting platform wants to automatically match job seekers' résumé
    summaries to job descriptions — beyond keyword overlap ("Python") to semantic
    understanding ("experience in data pipeline development" ↔ "ETL engineering").

    **Architecture.** Bi-encoder fine-tuned on (résumé summary, matching job) pairs.
    At query time: encode résumé → retrieve top-50 job descriptions from FAISS index
    → cross-encoder rerank (Notebook 28) to produce final ranked list.

    **Training data.** Historical applications where the candidate was hired (positive
    pairs); rejected applications from similar roles (hard negatives); random job
    descriptions from different domains (easy negatives).

    **Why not just BM25?** A résumé may say "designed data infrastructure" and the JD
    says "ETL engineering" — zero keyword overlap, high semantic similarity. Dense
    retrieval catches this; BM25 misses it.

    **Business metrics.** Recall@50 (does the right job appear in the top-50 retrieved
    before reranking?), MRR@10 (how high does the best match rank?), conversion rate
    (application → interview). A 5% lift in Recall@50 can double match quality.

    **Constraints.** Low latency (<100ms p99), privacy (embeddings computed on-premise,
    not sent to OpenAI), multilingual (candidates write in 30 languages → mE5 or
    LaBSE).

    **Cost of mistakes.** False negative: good match never shown → lost revenue. False
    positive: irrelevant job shown → candidate disengagement.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Model selection.** Baseline: `all-MiniLM-L6-v2` (fast, 384d, great for
      English). Better quality: `bge-large-en-v1.5`, `e5-large-v2` (768d, slower).
      Multilingual: `paraphrase-multilingual-MiniLM-L12-v2`, `LaBSE`. Cost-insensitive:
      `text-embedding-3-large` (OpenAI).
    - **Fine-tuning.** Domain adaptation with `sentence-transformers` + MNR loss on
      domain-specific pairs is the recommended approach. Even 1000–10K pairs can shift
      quality significantly. Use `TripletLoss` or `MultipleNegativesRankingLoss`.
    - **Indexing.** Pre-encode the corpus once; store in FAISS (Notebook 25), pgvector,
      or Pinecone. Re-encode only when the model or corpus changes.
    - **Dimensionality.** Trade: lower dim = faster FAISS scan, but quality loss.
      `text-embedding-3-small` supports Matryoshka learning — you can truncate to 512d
      with minimal quality loss. `MRL` (Matryoshka Representation Learning) is now the
      standard for production embedding models.
    - **Batch encoding.** Encode in batches of 32–256 on GPU for throughput.
      `encode(batch, batch_size=128)` in sentence-transformers handles this.
    - **Quantisation.** INT8 quantisation of the encoder reduces memory and speeds
      inference by ~2× with <1% quality loss on STS.
    - **Monitoring.** Track the distribution of query-to-corpus cosine similarities;
      if it shifts (mean drops, variance increases), the index may be stale or the
      query distribution has drifted (Notebook 45).
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Pooling strategy:**

    | Strategy | Quality (STS) | Cost | Best for |
    |---|---|---|---|
    | Word vector mean pool | Low | Very fast | Baseline, no GPU |
    | BERT mean pool (raw) | Low (worse than GloVe) | Slow | Do NOT use without fine-tuning |
    | SBERT mean pool (fine-tuned) | High | Moderate | **Default** |
    | [CLS] fine-tuned | High | Moderate | Classification downstream |
    | Max pool | Medium | Moderate | Keyword-dominated tasks |

    **Model size vs quality vs speed:**

    | Model | Dim | Params | Speed (GPU) | STS-B $\rho$ | When to use |
    |---|---|---|---|---|---|
    | all-MiniLM-L6-v2 | 384 | 22M | ~14K sent/s | ~0.89 | **Production default** |
    | all-mpnet-base-v2 | 768 | 109M | ~2K sent/s | ~0.91 | Higher quality needed |
    | bge-large-en-v1.5 | 1024 | 335M | ~500 sent/s | ~0.93 | Best quality, cost OK |
    | text-embedding-3-large | 3072 | API | API | ~0.93 | Max quality, budget |

    **Bi-encoder vs cross-encoder:**

    | | Bi-encoder | Cross-encoder |
    |---|---|---|
    | Pre-encoding | Yes | No |
    | Inference cost | $O(1)$ per query (dot product) | $O(N)$ per query (full pass) |
    | Accuracy | Good | Better |
    | Use for | **Retrieval (first pass)** | **Reranking (top-K only, Ntbk 28)** |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"Why is raw BERT mean-pooling worse than GloVe for STS?"* → BERT is optimised
      for MLM and NSP (next-sentence prediction, discarded in later models), not for
      producing geometrically meaningful sentence representations. Its hidden states
      occupy an anisotropic cone, compressing all cosines into a narrow range. Fine-
      tuning with contrastive loss on STS/NLI data forces the model to spread vectors
      across the space.
    - *"Bi-encoder vs cross-encoder — when do you use each?"* → Bi-encoder for
      retrieval: pre-encode the corpus, query at $O(1)$ cost. Cross-encoder for
      reranking: can't pre-encode, but sees full (q, d) interaction for accurate
      scoring. The standard production pipeline is bi-encoder → top-100 → cross-encoder
      → rerank → top-5 (Notebook 28).

    **Deep-dive questions**
    - *"What is in-batch negative sampling in MNR loss?"* → All other positives in the
      same mini-batch become negatives for a given anchor. Efficient because it requires
      no special hard-negative mining; quality scales with batch size because larger
      batches = more and harder negatives.
    - *"How would you fine-tune an embedding model for a new domain?"* → Collect (query,
      positive passage) pairs from domain data; use hard negatives (BM25 top-10 non-
      relevant for extra difficulty); train with `MultipleNegativesRankingLoss` in
      `sentence-transformers` for 1–3 epochs; evaluate on a held-out STS-style set.

    **Whiteboard questions**
    - "Write the MNR loss formula and explain why temperature matters." (§4.3)
    - "Draw the bi-encoder architecture and explain how retrieval works at query time."
      (§3 Mermaid)

    **Strong vs weak answers**
    - *"Should we use OpenAI embeddings or self-hosted in production?"*
      - **Weak:** "OpenAI is always better quality."
      - **Strong:** "For most English tasks, `all-mpnet-base-v2` or `bge-large` is
        within 1–2 Spearman points of `text-embedding-3-large` and costs zero per
        query. Self-hosted wins on privacy, latency, and cost at scale (>100M calls/day
        makes API cost prohibitive). OpenAI is the right default for early prototyping
        or when fine-tuning isn't feasible."

    **Common mistakes:** saying BERT embeddings are automatically good for semantic
    search (they're not without fine-tuning); confusing bi-encoder with cross-encoder;
    not knowing that MNR loss requires normalised vectors; forgetting that larger batches
    = better negatives = better training.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** A sentence embedding maps a sentence to a fixed vector. What
       property must that space have for it to be useful for semantic search?
    2. **Why not word-vector mean?** Name two things it misses.
    3. **The anisotropy problem.** What is it, and why does raw BERT suffer from it?
    4. **Bi-encoder.** Explain how you would build a semantic search engine for 10M
       documents using a bi-encoder. What is the offline step? What is the query step?
    5. **Cross-encoder.** Why can't you use a cross-encoder for first-pass retrieval?
    6. **Contrastive loss.** What are anchor, positive, and negative? What does the
       loss do to each?
    7. **In-batch negatives.** Why does larger batch size help?
    8. **Model choice.** Your task needs <50ms p99 latency for 1M documents and
       English-only text. Which embedding model do you pick and why?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. In our toy cosine similarity matrix (Figure 1), find the pair with the highest
       similarity that is NOT an expected paraphrase pair. Why might they be similar?
    2. Why does the anisotropy of raw BERT hidden states make cosine similarity
       unreliable as a semantic signal?

    **Beginner → Intermediate (coding)**
    3. Implement **whitening** on the toy embeddings: subtract the mean embedding and
       multiply by $\Sigma^{-1/2}$ (where $\Sigma$ is the sample covariance). Re-run
       Figure 1 and show that the average off-diagonal cosine similarity decreases.
    4. Add max pooling as a fourth strategy alongside mean and [CLS] in the toy
       implementation and compare their semantic search results for 3 queries.

    **Intermediate (analysis)**
    5. Implement **triplet loss** from scratch: given a batch of (anchor, positive,
       negative) triples, compute $\max(0, d(a,p) - d(a,n) + \text{margin})$ and show
       gradient updates that push $p$ closer and $n$ further from $a$.
    6. Simulate the effect of hard vs easy negatives: create "easy" negatives (random
       sentences from a different topic) and "hard" negatives (sentences from the same
       topic but different meaning). Measure MNR loss with each and explain the
       difference.

    **Senior (interview + production design)**
    7. *Design:* the embedding pipeline for the job-matching system in §9. Include:
       model selection, fine-tuning strategy (data, loss, epochs), FAISS index type
       (Notebook 25), update frequency, and monitoring KPIs.
    8. *Tradeoff:* you have a budget of 10 GPU-hours/day for re-encoding a 5M-document
       corpus that updates daily. MiniLM-L6 encodes at 14K sentences/GPU-second.
       Calculate whether daily full re-encoding is feasible, and propose an incremental
       update strategy if not.
    9. *Evaluate:* your bi-encoder Recall@50 is 85% on the general STS-B benchmark but
       only 60% on your domain (legal contracts). Propose a step-by-step improvement
       plan: data collection, training, evaluation, and production rollout.
    """),

    md(r"""
    ---
    ### Summary
    Sentence embeddings lift word vectors (Notebook 20) to the sentence level. Mean
    pooling over Transformer hidden states is simple but requires **contrastive fine-
    tuning** (SBERT, MNR loss) to make cosine similarity align with human semantic
    judgements. The **bi-encoder** (independent encoding, cosine similarity) enables
    scalable retrieval; the **cross-encoder** (joint encoding) provides high-accuracy
    reranking for top-K candidates. Both are production-grade tools used in every RAG
    pipeline (Phase 5).

    **Key lesson:** raw BERT embeddings are NOT sentence embeddings — fine-tuning on
    NLI/STS data with contrastive loss is what makes them geometrically meaningful.

    **Next:** `22 · LLM Training Pipeline` — how language models are pre-trained
    (next-token prediction at scale), continually pre-trained, instruction-tuned (SFT),
    and aligned (RLHF/DPO). This closes Phase 4's arc from bag-of-words (20) to
    trained dense embeddings (21) to full LLMs (22–24).
    """),
]

build("phase4_nlp_llms/21_sentence_embeddings.ipynb", cells)
