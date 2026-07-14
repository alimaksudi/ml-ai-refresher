"""Builder for Lesson NLP-01 — TF-IDF and Word Embeddings.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # NLP-01 · TF-IDF and Word Embeddings
    ### NLP representation foundations — taught before sequence architectures

    **Prerequisites:** FND-01, CML-02, and MLE-01. No RNN, attention, or Transformer
    knowledge is assumed. This module establishes tokenization and lexical/dense
    representation baselines before neural sequence models.

    > How do we turn words into numbers that machines can reason over? This notebook
    > traces the evolution from **count-based representations** (TF-IDF, BM25) to
    > **predictive embeddings** (word2vec, GloVe, FastText) — each a more powerful
    > theory of what "meaning" is. These methods are the direct precursors to the
    > contextual embeddings inside the Transformer (Lesson DL-08); Lesson NLP-02 shows how
    > to pool them into sentence-level vectors for downstream tasks.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - TF-IDF from first principles: why term frequency alone is misleading, what
      inverse document frequency corrects for, and why we log-scale it.
    - BM25 — the production-grade extension of TF-IDF used in Elasticsearch, Lucene,
      and every major search engine.
    - Why dense word embeddings beat sparse vectors: the distributional hypothesis
      ("a word is known by the company it keeps") and why *prediction* forces geometry.
    - **word2vec** (CBOW and Skip-gram) — the training task, the negative-sampling
      trick, and what the resulting vectors encode.
    - GloVe's global co-occurrence objective and FastText's subword trick.
    - Practical embedding space arithmetic: king − man + woman ≈ queen.
    - When to use sparse vs dense retrieval (links to Lesson RAG-01 — Similarity Search).

    **Why it matters**
    - TF-IDF/BM25 is still the baseline for text search in most production systems.
    - word2vec embeddings trained in 2013 are still embedded in many production
      pipelines. Understanding them is required to reason about their limitations and
      know when to upgrade to contextual embeddings (Lesson NLP-02).

    **Typical interview questions**
    - "What is TF-IDF and why do we use log in IDF?"
    - "Why do word2vec embeddings encode analogies?"
    - "When would you use BM25 over dense retrieval?"
    - "What's the difference between CBOW and Skip-gram?"
    """),

    md(r"""
    ## 2 · Historical Motivation

    **The bag-of-words era.** Before neural embeddings, text was represented as a
    sparse vector of word counts — a "bag of words" that discarded order and meaning.
    **TF-IDF** (Salton & Buckley, 1988) improved on raw counts by down-weighting words
    that appear in every document (e.g., "the") and up-weighting words that
    discriminate one document from others. For two decades it powered every web search
    engine.

    **The problem: sparsity and no semantics.** A TF-IDF vector for a 100K-word
    vocabulary is 99.99% zeros. Worse, "car" and "automobile" have zero cosine
    similarity because they share no characters or positions. There's no way for a
    bag-of-words model to know they're synonyms.

    **The distributional hypothesis (Harris, 1954; Firth, 1957).** "A word is
    characterized by the company it keeps." Words that appear in similar contexts have
    similar meanings. This insight predates neural NLP by 60 years, but the field
    didn't find an efficient way to exploit it until word2vec.

    **word2vec (Mikolov et al., 2013a, 2013b).** The key insight: don't try to count
    co-occurrences — train a shallow neural network to *predict* a word from its
    context (or vice versa). The weights of this network become dense, low-dimensional
    embeddings that encode meaning geometrically. Trained on ~100B words in a day,
    these 300-dimensional vectors showed that *linear analogies encode semantic
    relations*: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$.

    **GloVe (Pennington et al., 2014)** combined the efficiency of word2vec with the
    interpretability of co-occurrence counts using a global matrix factorisation.
    **FastText (Bojanowski et al., 2017)** extended word2vec with character n-grams,
    enabling embeddings for misspellings and morphologically rich languages.

    **Today.** Static embeddings (one vector per word regardless of context) have been
    mostly superseded by contextual embeddings from Transformers (DL-08 and NLP-02), but
    they remain the conceptual foundation: every LLM's first layer is still a token
    embedding lookup, and sparse retrieval (BM25) still beats dense retrieval on many
    keyword-heavy tasks.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **TF-IDF intuition.** A word's importance in a document should be high when it
    appears frequently *in that document* (TF) but rarely *across all documents* (IDF).
    "Machine" in a paper about robotics: high TF, high IDF → high score. "The" in that
    same paper: high TF, near-zero IDF → low score. The log in IDF compresses the
    range: a word appearing in 1 of 1000 documents isn't 1000× more distinctive than
    one appearing in 500 of 1000 — the log gives a ratio of 3 vs 1, which matches
    intuition.

    **BM25 intuition.** TF-IDF has two saturation failures: (1) a document with "cat"
    100 times shouldn't score 100× one with "cat" 5 times — there are diminishing
    returns; (2) long documents naturally accumulate more term occurrences — they should
    be penalised relative to short ones. BM25 adds a **TF saturation** parameter $k_1$
    and a **length normalisation** parameter $b$ to fix both.

    **Word embedding intuition.** Give a network the task: "given 'The ___ sat on the
    mat', predict 'cat'." To do this well, it must learn that 'cat', 'dog', 'rabbit'
    are all plausible — they all sit on mats. So it learns to give them similar vectors.
    By training on billions of sentences, the vectors organise into a geometry where
    directions encode semantic relationships (gender, tense, location, etc.).

    ```mermaid
    flowchart LR
        Raw["raw text"] --> Tok["tokenise"] --> BagOfWords["bag of words\n(sparse, no semantics)"]
        Tok --> TFIDF["TF-IDF\n(sparse, discriminative)"]
        Tok --> Pred["prediction task\n(CBOW / Skip-gram)"] --> Embed["dense embeddings\n(300d, semantic geometry)"]
        Embed --> Ctx["contextual embedding\n(Transformer, DL-08)"]
    ```
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    import math, re

    rng = np.random.default_rng(42)
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # Tiny corpus for all demonstrations.
    CORPUS = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat ate the rat",
        "the dog chased the cat",
        "the rat ran from the cat",
        "a fox ran over the mat",
        "the log was near the mat",
        "the cat is a small animal",
        "a dog is a loyal animal",
        "the rat is a tiny animal",
    ]
    print(f"Corpus: {len(CORPUS)} documents")
    for i, d in enumerate(CORPUS):
        print(f"  [{i}] {d}")
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 TF-IDF

    **Term Frequency (TF):** how often term $t$ appears in document $d$, normalised
    by document length:
    $$\text{TF}(t,d) = \frac{f(t,d)}{\sum_{t'} f(t',d)}$$

    **Inverse Document Frequency (IDF):** log-scaled rarity across $N$ documents:
    $$\text{IDF}(t) = \log\!\left(\frac{N}{1 + \text{df}(t)}\right) + 1$$
    where $\text{df}(t)$ is the number of documents containing $t$. The $+1$ in the
    denominator prevents division-by-zero; the outer $+1$ ensures even ubiquitous terms
    get a small positive weight (sklearn smooth variant).

    **TF-IDF score:**
    $$\text{TFIDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$$

    **Why log?** Without it, IDF is a ratio — a term in 1 of 1000 docs would get
    weight 1000, drowning all other terms. Log compresses this: $\log 1000 \approx 7$
    vs $\log 2 \approx 0.7$, a 10× range instead of 500×.

    ### 4.2 BM25

    BM25 saturates TF and normalises for document length:
    $$\text{BM25}(t,d) = \text{IDF}(t) \cdot
    \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\!\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}$$
    Typical defaults: $k_1=1.5$, $b=0.75$. As $f \to \infty$, the TF factor saturates
    at $k_1+1$ (diminishing returns). Length normalisation: docs longer than average
    are penalised by the $b \cdot |d|/\text{avgdl}$ term.

    ### 4.3 word2vec — Skip-gram with Negative Sampling (SGNS)

    Given centre word $w_c$ and context word $w_o$ from a window of size $m$, SGNS
    maximises:
    $$J = \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})
         + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n}\!\left[\log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_c})\right]$$
    where $\mathbf{v}$ are "input" embeddings, $\mathbf{u}$ are "output" embeddings,
    $\sigma$ is the sigmoid function, and $K$ noise words are drawn from the unigram
    distribution $P_n(w) \propto f(w)^{3/4}$.

    Each gradient step pushes $\mathbf{v}_{w_c}$ toward $\mathbf{u}_{w_o}$ (positive
    pair) and away from $K$ random words (negative samples). Over billions of pairs,
    words that co-occur frequently end up with high dot products — i.e., **close in
    embedding space**.

    ### 4.4 GloVe

    GloVe fits a weighted least-squares on the log co-occurrence matrix $X$:
    $$J = \sum_{i,j} f(X_{ij})\!\left(\mathbf{v}_i^\top \mathbf{u}_j + b_i + c_j - \log X_{ij}\right)^2$$
    where $f$ is a capping function that down-weights very frequent co-occurrences.
    GloVe is more interpretable (explicit matrix factorisation) but performs similarly
    to SGNS in practice.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    ### 5a TF-IDF from scratch
    """),

    code(r"""
    # 5a.1 Tokenise and build vocabulary.
    def tokenise(doc):
        return re.findall(r"[a-z]+", doc.lower())

    tokenised = [tokenise(d) for d in CORPUS]
    vocab = sorted(set(w for doc in tokenised for w in doc))
    w2i = {w: i for i, w in enumerate(vocab)}
    V, N = len(vocab), len(CORPUS)
    print(f"Vocabulary size: {V},  Documents: {N}")

    # 5a.2 Term frequencies (normalised).
    def tf(doc_tokens):
        c = Counter(doc_tokens)
        total = sum(c.values())
        return {w: v / total for w, v in c.items()}

    # 5a.3 Document frequencies and smooth IDF.
    df = Counter(w for doc in tokenised for w in set(doc))
    def idf(word, smooth=True):
        if smooth:
            return math.log((N + 1) / (df.get(word, 0) + 1)) + 1
        else:
            return math.log(N / df.get(word, 1))

    # 5a.4 TF-IDF matrix (N x V, dense for demo).
    tfidf_matrix = np.zeros((N, V))
    for d_idx, doc in enumerate(tokenised):
        tf_d = tf(doc)
        for w, tf_val in tf_d.items():
            tfidf_matrix[d_idx, w2i[w]] = tf_val * idf(w)

    print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
    # Show top words for doc 0.
    doc0_scores = {vocab[i]: tfidf_matrix[0, i] for i in range(V)}
    top5 = sorted(doc0_scores, key=doc0_scores.get, reverse=True)[:5]
    print(f"Top TF-IDF words in doc 0: {[(w, round(doc0_scores[w], 3)) for w in top5]}")
    """),

    code(r"""
    # 5a.5 Cosine similarity search (Lesson FND-01 reuse).
    def cosine_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def tfidf_search(query, top_k=3):
        q_tok = tokenise(query)
        q_tf = tf(q_tok)
        q_vec = np.array([q_tf.get(w, 0) * idf(w) for w in vocab])
        scores = [(i, cosine_sim(q_vec, tfidf_matrix[i])) for i in range(N)]
        return sorted(scores, key=lambda x: -x[1])[:top_k]

    for q in ["cat mat", "dog animal", "fox ran"]:
        results = tfidf_search(q)
        print(f"Query '{q}': {[(CORPUS[i][:30], round(s, 3)) for i, s in results]}")
    """),

    md(r"""
    ### 5b BM25 from scratch
    """),

    code(r"""
    # BM25 implementation from scratch.
    k1, b = 1.5, 0.75
    avgdl = np.mean([len(doc) for doc in tokenised])

    def bm25_idf(word):
        df_w = df.get(word, 0)
        return math.log((N - df_w + 0.5) / (df_w + 0.5) + 1)

    def bm25_score(query, doc_tokens):
        tf_d = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for w in tokenise(query):
            tf_w = tf_d.get(w, 0)
            score += bm25_idf(w) * (tf_w * (k1 + 1)) / (
                tf_w + k1 * (1 - b + b * dl / avgdl)
            )
        return score

    def bm25_search(query, top_k=3):
        scores = [(i, bm25_score(query, tokenised[i])) for i in range(N)]
        return sorted(scores, key=lambda x: -x[1])[:top_k]

    print("BM25 search results:")
    for q in ["cat mat", "dog animal", "rat ran"]:
        results = bm25_search(q)
        print(f"  Query '{q}': {[(CORPUS[i][:30], round(s, 3)) for i, s in results]}")
    """),

    md(r"""
    ### 5c word2vec — Skip-gram with Negative Sampling (from scratch)
    """),

    code(r"""
    # 5c.1 Build co-occurrence data from corpus with window=2.
    WINDOW = 2
    pairs = []                                          # (centre, context) positive pairs
    all_words = [w for doc in tokenised for w in doc]
    word_freq = Counter(all_words)
    # Use only words that appear at least twice.
    kept_vocab = [w for w, c in word_freq.items() if c >= 1]
    w2i_sg = {w: i for i, w in enumerate(kept_vocab)}
    i2w_sg = {i: w for w, i in w2i_sg.items()}
    V_sg = len(kept_vocab)

    for doc in tokenised:
        ids = [w2i_sg[w] for w in doc if w in w2i_sg]
        for ci, cw in enumerate(ids):
            lo = max(0, ci - WINDOW)
            hi = min(len(ids), ci + WINDOW + 1)
            for j in range(lo, hi):
                if j != ci:
                    pairs.append((cw, ids[j]))

    print(f"Vocabulary: {V_sg} words,  Training pairs: {len(pairs)}")

    # 5c.2 Noise distribution P(w) ~ freq^0.75 for negative sampling.
    freq_arr = np.array([word_freq.get(i2w_sg[i], 0) for i in range(V_sg)], dtype=float)
    noise_probs = freq_arr ** 0.75
    noise_probs /= noise_probs.sum()

    # 5c.3 Initialise embeddings.
    D = 10                                              # small dimension for demo
    W_in  = rng.uniform(-0.5 / D, 0.5 / D, (V_sg, D))  # input (centre) embeddings
    W_out = np.zeros((V_sg, D))                         # output (context) embeddings

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def sgns_step(centre, context, K=5, lr=0.025):
        # Positive pair.
        v_c = W_in[centre]
        u_o = W_out[context]
        sig_pos = sigmoid(u_o @ v_c)
        grad_v_c = (sig_pos - 1.0) * u_o          # push centre toward context
        grad_u_o = (sig_pos - 1.0) * v_c
        # Negative samples.
        negs = rng.choice(V_sg, size=K, p=noise_probs)
        for neg in negs:
            u_neg = W_out[neg]
            sig_neg = sigmoid(u_neg @ v_c)
            grad_v_c += sig_neg * u_neg            # push centre away from noise
            W_out[neg] -= lr * sig_neg * v_c
        W_in[centre]  -= lr * grad_v_c
        W_out[context] -= lr * grad_u_o

    # 5c.4 Training loop (few passes — tiny corpus).
    EPOCHS = 500
    pair_arr = np.array(pairs)
    for epoch in range(EPOCHS):
        idx = rng.permutation(len(pair_arr))
        for centre, context in pair_arr[idx]:
            sgns_step(centre, context)

    print(f"Training done ({EPOCHS} epochs x {len(pairs)} pairs)")
    """),

    code(r"""
    # 5c.5 Check nearest neighbours in embedding space using cosine similarity.
    def nn(word, k=4):
        if word not in w2i_sg:
            return []
        v = W_in[w2i_sg[word]]
        sims = [(i2w_sg[i], cosine_sim(v, W_in[i])) for i in range(V_sg) if i != w2i_sg[word]]
        return sorted(sims, key=lambda x: -x[1])[:k]

    for probe in ["cat", "dog", "rat", "mat"]:
        neighbours = nn(probe)
        print(f"  Nearest to '{probe}': {neighbours}")
    print("\n(With a larger corpus, 'cat'/'dog'/'rat' cluster as animals; on this tiny")
    print(" corpus, context overlap already pulls co-occurring words together.)")
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — TF-IDF heatmap: rows=docs, cols=vocab.
    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(tfidf_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(N)); ax.set_yticklabels([f"doc{i}" for i in range(N)], fontsize=8)
    ax.set_xticks(range(0, V, 2)); ax.set_xticklabels(vocab[::2], rotation=45, ha="right", fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Figure 1 — TF-IDF matrix: discriminative weights per (doc, term)")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** The TF-IDF matrix. Each cell is the weight of a term (column) in a
    document (row). Stop words ("the", "a", "on") have near-zero IDF and thus near-zero
    TF-IDF regardless of how often they appear. Content words ("cat", "mat", "animal")
    that distinguish documents get high weights — exactly the discriminative signal a
    retrieval system needs. This sparsity is the defining property of bag-of-words
    representations: most entries are zero, and there is no shared structure between
    similar words like "cat" and "animal."
    """),

    code(r"""
    # Figure 2 — TF saturation: BM25 vs raw TF.
    tf_raw = np.arange(0, 20, 0.1)
    bm25_tf = tf_raw * (k1 + 1) / (tf_raw + k1)      # length-normalised out (b=0, avgdl=dl)
    fig, ax = plt.subplots()
    ax.plot(tf_raw, tf_raw / tf_raw.max(), label="raw TF (linear)")
    ax.plot(tf_raw, bm25_tf / bm25_tf.max(), label=f"BM25 TF term (k1={k1})")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="saturation ceiling")
    ax.set_xlabel("raw term count"); ax.set_ylabel("normalised contribution")
    ax.set_title(f"Figure 2 — BM25 TF saturation (k1={k1}): diminishing returns")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 2.** The key BM25 improvement over raw TF-IDF: **saturation**. A document
    containing "cat" 20 times should not score 4× one containing "cat" 5 times — there
    are diminishing returns. The BM25 TF term (orange) saturates at $k_1+1$ (here 2.5),
    while raw TF (blue) grows linearly without bound. The parameter $k_1$ controls how
    fast saturation is reached: smaller $k_1$ saturates faster. This is why BM25
    consistently outperforms TF-IDF on standard IR benchmarks.
    """),

    code(r"""
    # Figure 3 — word2vec embeddings in 2D via PCA.
    from numpy.linalg import svd

    def pca2d(X):
        X = X - X.mean(0)
        U, S, Vt = svd(X, full_matrices=False)
        return X @ Vt[:2].T                       # project onto top 2 PCs

    emb2d = pca2d(W_in)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, word in enumerate(kept_vocab):
        x, y = emb2d[i]
        ax.scatter(x, y, s=20, color="steelblue")
        ax.annotate(word, (x, y), fontsize=8, alpha=0.8)
    ax.set_title("Figure 3 — word2vec Skip-gram embeddings (PCA 2D)\ntrained on tiny corpus")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** The learned Skip-gram embeddings projected onto their top two
    principal components. On this tiny 10-sentence corpus the geometry is limited, but
    you should see words that co-occur in similar contexts (animal names, location words)
    drift toward each other. On a billion-token corpus this 2D projection reveals clear
    clusters: animal names, country names, action verbs, etc. — and linear directions
    that encode gender, number, tense, and other semantic relations. This is the
    geometry that makes the famous "king − man + woman ≈ queen" analogy work.
    """),

    code(r"""
    # Figure 4 — IDF values across vocabulary: rarity vs ubiquity.
    idf_vals = np.array([idf(w) for w in vocab])
    order = np.argsort(idf_vals)[::-1]
    top20 = order[:20]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh([vocab[i] for i in top20[::-1]], idf_vals[top20[::-1]])
    ax.set_xlabel("IDF value (smooth)"); ax.set_title("Figure 4 — IDF scores: rare vs common words")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** IDF scores for the top-20 most distinctive words. Rare, content-
    specific words (appearing in only 1–2 documents) get the highest IDF and drive
    retrieval. Common words ("the", "a", "on") get low IDF and contribute almost
    nothing to cosine similarity even if they appear many times. This is why IDF is the
    *discriminative* component of TF-IDF: it automatically learns which words separate
    documents without any manual stopword list.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Vocabulary mismatch** | Query term not in index → zero matches | OOV word at query or index time | BM25: expand query synonyms; FastText subword; dense retrieval (RAG-01) |
    | **IDF over-smoothing** | Rare domain terms penalised | Corpus IDF trained on wrong domain | Fine-tune IDF on domain corpus; use raw count with domain-specific stoplist |
    | **TF-IDF ignores order** | "cat chased dog" == "dog chased cat" | Bag-of-words discards position | n-gram TF-IDF (bigrams); or use neural retrieval |
    | **word2vec polysemy** | "bank" has one vector for financial/river | Static embedding: one vector per word | Contextual embeddings (BERT/NLP-02); sense disambiguation |
    | **word2vec frequency bias** | Very frequent words dominate co-occurrence | Training signal dominated by stop words | Sub-sampling: discard tokens with prob $\propto \sqrt{f_{\text{thresh}}/f}$ |
    | **Embedding dimension too small** | Poor analogy arithmetic | Low capacity | Increase $D$; typical production: 100–300d |
    | **BM25 no semantic matching** | Query "automobile" misses docs about "car" | Keyword only, no synonym expansion | Hybrid search: BM25 + dense retrieval (RAG-06) |
    """),

    code(r"""
    # Demonstrate: TF-IDF query for "automobile" returns nothing even though docs contain "car".
    hits = tfidf_search("automobile")
    print("TF-IDF search for 'automobile':")
    print(f"  Results: {hits}")
    print("  -> Zero matches even though semantically similar to 'cat', 'dog'")
    print("  -> Dense/hybrid retrieval (RAG-01 and RAG-06) solves this.")
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 sklearn TfidfVectorizer (production-grade, sparse matrices, sublinear TF).
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectoriser = TfidfVectorizer(sublinear_tf=True, min_df=1, norm="l2")
    X_tfidf = vectoriser.fit_transform(CORPUS)        # returns sparse (N, V)
    print(f"sklearn TF-IDF matrix: {X_tfidf.shape}, dtype={X_tfidf.dtype}")
    print(f"Non-zero entries: {X_tfidf.nnz} / {X_tfidf.shape[0]*X_tfidf.shape[1]}"
          f"  ({100*X_tfidf.nnz/(X_tfidf.shape[0]*X_tfidf.shape[1]):.1f}% dense)")

    # Query
    q_vec = vectoriser.transform(["cat mat"])
    sims = cosine_similarity(q_vec, X_tfidf)[0]
    ranked = np.argsort(sims)[::-1][:3]
    print("\nsklearn search for 'cat mat':")
    for i in ranked:
        print(f"  [{i}] {CORPUS[i][:40]}  sim={sims[i]:.3f}")
    """),

    code(r"""
    # 8.2 Gensim word2vec (production-grade, C-optimised, windowed SGNS/CBOW).
    try:
        from gensim.models import Word2Vec
        model_w2v = Word2Vec(
            sentences=tokenised,
            vector_size=50,
            window=3,
            min_count=1,
            sg=1,                   # sg=1: Skip-gram; sg=0: CBOW
            workers=1,
            epochs=200,
            seed=42,
        )
        print("Gensim word2vec trained.")
        for probe in ["cat", "dog"]:
            nn_g = model_w2v.wv.most_similar(probe, topn=3)
            print(f"  Most similar to '{probe}': {nn_g}")
    except Exception as e:
        print(f"[gensim not available: {type(e).__name__}]")
        print("In production: gensim.models.Word2Vec(sentences, vector_size=300, window=5,")
        print("               min_count=5, sg=1, workers=4, epochs=5) trains ~hours on 1B tokens")
    """),

    code(r"""
    # 8.3 Pre-trained FastText embeddings via gensim (guarded — large download).
    # In production: gensim.downloader.load("fasttext-wiki-news-subwords-300")
    # or: from gensim.models.fasttext import load_facebook_model
    # FastText advantage: OOV words handled via character n-grams.
    print("FastText (gensim) — production pattern (skipped here to avoid download):")
    print("  import gensim.downloader as api")
    print("  ft = api.load('fasttext-wiki-news-subwords-300')")
    print("  ft['runnning']  # handles misspellings via subword decomposition")
    print()
    print("Key difference from word2vec:")
    print("  word2vec: OOV word -> KeyError")
    print("  FastText: OOV word -> mean of character n-gram vectors (graceful fallback)")
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Search Relevance at a Legal Document Platform

    **Scenario.** A legal SaaS platform needs to retrieve relevant case law and
    statutes given a lawyer's free-text query. The corpus is 10M documents; queries
    are highly specific ("breach of fiduciary duty Delaware 2018").

    **Why not just word2vec / dense retrieval?**
    - Legal queries use precise terminology. A query for "mens rea" should NOT be
      expanded to "intent" unless explicitly desired — over-expansion lowers precision.
    - BM25 is exact-match, auditable, and GDPR-compliant (no model inference).
    - Dense retrieval adds recall for paraphrase, but the cost of false positives
      (wrong case cited in court) is very high.

    **Chosen architecture:** BM25 first-pass retrieval (top-100 candidates) → dense
    cross-encoder reranker (Lesson RAG-07). This is the standard hybrid approach.

    **Cost of mistakes:**
    - False negative (missing the key case) → legal malpractice risk.
    - False positive (wrong jurisdiction retrieved) → billing time on irrelevant docs.

    **KPIs:** MRR@10, NDCG@20, P@1 on a lawyer-curated golden set; p95 query latency
    (<200ms); index freshness (daily ingestion of new decisions).

    **Constraints:** index must be rebuilt nightly; vocabulary updates as new legal
    jargon emerges (BM25 naturally handles — just re-index). IDF computed on the full
    corpus, not updated per-query.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Sparse index (inverted index).** Production BM25 uses an inverted index (term →
      sorted list of (doc_id, tf) pairs) for sub-millisecond retrieval over 100M docs.
      Elasticsearch/Lucene implement this; our dense matrix above is for learning only.
    - **Tokenisation matters.** Stemming (Porter, Snowball) and lemmatisation can
      collapse "running"/"ran"/"run" to the same term, improving recall but potentially
      losing specificity. Choose based on domain.
    - **Memory.** word2vec 300d × 500K-word vocabulary = 600MB. Use memory-mapped files
      (gensim's `save` / `load` with `mmap='r'`) for low-latency lookup.
    - **Static vs contextual.** word2vec gives one vector per word regardless of context;
      in "bank robbery" vs "river bank" the vectors are identical. For disambiguation,
      upgrade to contextual embeddings (Lesson NLP-02, BERT).
    - **Updating embeddings.** word2vec must be retrained when the vocabulary changes
      significantly. FastText handles new words via subword; still needs periodic
      retraining for semantic drift (Lesson PROD-05).
    - **Hybrid retrieval.** BM25 + dense retrieval is the production standard (Notebook
      27): BM25 handles keyword-exact matches; dense handles paraphrase and synonyms.
    - **BM25 parameters.** $k_1$ and $b$ should be tuned on a query-relevance dataset
      (grid search over held-out queries); defaults (1.5, 0.75) work well for English
      web text but not always for code or legal text.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Representation type:**

    | Method | Dim | Semantic | OOV | Speed | When to use |
    |---|---|---|---|---|---|
    | TF-IDF | $V$ (100K+) | None | Zero score | Very fast | Keyword search, small corpora, baseline |
    | BM25 | Inverted index | None | Zero score | Very fast | **Production text search default** |
    | word2vec | 100–300 | Static | No (KeyError) | Fast | Frozen semantic features, analogy tasks |
    | GloVe | 50–300 | Static | No | Fast | Same as word2vec; more interpretable |
    | FastText | 100–300 | Static | Yes (subword) | Fast | Morphologically rich languages, misspellings |
    | BERT/contextual | 768 | Contextual | Yes | Slow (inference) | Disambiguation, NLU tasks (NLP-02) |

    **BM25 parameters:**

    | Parameter | Effect | Typical range |
    |---|---|---|
    | $k_1$ | TF saturation speed | 1.2 – 2.0 |
    | $b$ | Length normalisation strength | 0 (none) – 1.0 (full); 0.75 default |
    | $b=0$ | No length normalisation | Short-query corpora |
    | $b=1$ | Full length normalisation | Very heterogeneous doc lengths |

    **word2vec variants:**

    | | Skip-gram | CBOW |
    |---|---|---|
    | Objective | Predict context from centre | Predict centre from context |
    | Rare words | Better (treats each pair individually) | Worse |
    | Training speed | Slower | Faster |
    | Typical use | Most NLP tasks | Fast training budget |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What is TF-IDF and why do we use log in IDF?"* → TF measures local frequency;
      IDF measures global rarity; log compresses the extreme ratios so common words
      (IDF~0) and rare words (IDF~3–7) stay in a workable range. Without log, a
      term in 1 of 1M docs gets IDF=1M, dominating all others.
    - *"Why does word2vec produce embeddings that encode analogies?"* → The SGD
      objective pushes words that appear in similar contexts toward each other. If
      "king" and "queen" appear in the same sentences as "rules", "kingdom", "crown",
      their vectors cluster. The residual direction "man→woman" is shared because
      gender difference is consistent across many paired contexts.

    **Deep-dive questions**
    - *"When would you use BM25 over dense retrieval?"* → When: (1) exact-match
      matters (legal/medical/code); (2) latency/cost prohibit inference; (3) the corpus
      is small (<100K docs); (4) you need an auditable, deterministic baseline.
    - *"What's negative sampling and why does it work?"* → Full softmax over 500K vocab
      is too expensive per step. NS approximates it by sampling $K$ random words as
      negatives — turns a multi-class problem into $K+1$ binary problems. Works because
      the objective still pushes the positive context up and random contexts down, which
      is enough to learn the right geometry.
    - *"GloVe vs word2vec — which is better?"* → Empirically similar; GloVe is more
      interpretable (factorises a log co-occurrence matrix), word2vec trains online and
      scales to larger corpora. Choose by engineering constraints, not expected quality.

    **Whiteboard questions**
    - "Write the BM25 formula and explain each term." (§4.2)
    - "Draw the word2vec Skip-gram architecture and loss function." (§4.3)

    **Strong vs weak answers**
    - *"Should we train word2vec from scratch or use pre-trained?"*
      - **Weak:** "Pre-trained is always better."
      - **Strong:** "Pre-trained (GloVe-840B, fastText-cc) is the default for general-
        domain text. For highly domain-specific vocabulary (medical, legal, code), fine-
        tune or retrain on domain data — pre-trained embeddings for 'pleading' (legal)
        vs 'pleading' (emotional) are trained on general web text and may be misleading.
        But for most tasks, start with pre-trained and measure."

    **Common mistakes:** forgetting that TF-IDF is permutation-invariant (order doesn't
    matter); not knowing that word2vec has two separate embedding matrices; claiming
    word2vec "understands" meaning (it encodes distributional similarity, not semantics).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Explain TF-IDF in one sentence. Then explain word2vec in one
       sentence.
    2. **Why invented?** What problem does IDF solve that raw TF does not?
    3. **How does it work?** Walk through the BM25 formula — TF saturation, length
       norm.
    4. **The distributional hypothesis.** State it. Why does it justify word2vec?
    5. **Skip-gram training.** What is the input? What is predicted? What is the loss?
    6. **Negative sampling.** Why is full softmax too expensive? How does NS fix it?
    7. **Tradeoffs.** BM25 vs dense retrieval — name two scenarios where each wins.
    8. **OOV words.** How does BM25 handle OOV? How does word2vec handle OOV? How does
       FastText?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Compute TF-IDF by hand for the word "cat" in document 0 of our corpus, using
       the smooth IDF formula from §4.1. Verify against our scratch implementation.
    2. Why does the $b$ parameter in BM25 exist? Give an example where $b=0$ would be
       preferred.

    **Beginner → Intermediate (coding)**
    3. Add bigram support to the TF-IDF vectoriser: extract `(word1_word2)` pairs and
       include them in the vocabulary. Re-run the search for "cat mat" and show which
       bigram has the highest IDF.
    4. Implement sub-sampling in the word2vec trainer: with probability $1 -
       \sqrt{t/f(w)}$ skip a word during training (where $t=10^{-4}$ is a threshold and
       $f(w)$ is its unigram frequency). Show the effect on neighbour quality.

    **Intermediate (analysis)**
    5. Tune BM25 parameters $k_1 \in \{0.5, 1.0, 1.5, 2.0\}$ and $b \in \{0.25, 0.5,
       0.75\}$ on a tiny annotated query set (manually label the most relevant doc per
       query) and report the best combination by MRR@3.
    6. Implement GloVe's weighted least-squares objective from scratch (§4.4): build the
       co-occurrence matrix $X$, apply the capping function $f(X_{ij}) = \min(1,
       (X_{ij}/100)^{0.75})$, and minimise $J$ with SGD. Compare embeddings to SGNS.

    **Senior (interview + production design)**
    7. *Design:* the search system for the legal document platform in §9. Include:
       ingestion pipeline (daily), BM25 index (Elasticsearch/Lucene), reranker
       (Lesson RAG-07), latency SLAs, and monitoring (query distribution drift, PROD-05).
    8. *Scaling:* you have 10B tokens of in-domain legal text and a budget for 1 GPU-
       week. Would you train word2vec, GloVe, or fine-tune a pre-trained contextual
       model (NLP-02)? Justify with expected quality, training time, and serving cost.
    9. *Interview:* "Your BM25-based search suddenly returns poor results after a legal
       vocabulary update (new acronyms). How do you diagnose and fix this?"
    """),

    md(r"""
    ---
    ### Summary
    **TF-IDF** weights terms by local frequency × global rarity — fast, sparse, no
    semantics but a strong baseline. **BM25** adds saturation and length normalisation
    and remains the production default for keyword retrieval. **word2vec** (SGNS)
    replaces sparse counting with a prediction task, learning dense 100–300d embeddings
    where geometric direction encodes semantic relations — the distributional hypothesis
    made computationally efficient. **GloVe** achieves the same via global matrix
    factorisation; **FastText** extends with subword character n-grams.

    **Key limitation of all methods here:** one vector per word regardless of context.
    Lesson NLP-02 shows how to pool or contextualise these vectors into sentence-level
    representations, and Lesson RAG-01 shows how to retrieve efficiently with them.

    **Related lesson:** `NLP-02 · Sentence Embeddings` — BERT [CLS] pooling, mean pooling,
    sentence-transformers (bi-encoders vs cross-encoders), and how to build a semantic
    search engine from scratch.
    """),
]

build("05_nlp_and_llms/01_tfidf_and_word_embeddings.ipynb", cells)
