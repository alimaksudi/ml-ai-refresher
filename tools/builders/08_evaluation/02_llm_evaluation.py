"""Builder for Lesson EVAL-02 — LLM Evaluation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # EVAL-02 · LLM Evaluation
    ### Section 08 — Evaluation · *ML/AI Senior Mastery Curriculum*

    > How do you know if your LLM is getting better or worse? This notebook teaches
    > the complete LLM evaluation toolkit: reference-based metrics (BLEU, ROUGE-L,
    > BERTScore), intrinsic metrics (perplexity), task-specific metrics (pass@k for
    > code, exact match + F1 for QA), and production regression testing patterns.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **BLEU** (from scratch): n-gram precision with brevity penalty — the standard for translation.
    - **ROUGE-L** (from scratch): longest common subsequence — the standard for summarisation.
    - **BERTScore** (intuition + cosine similarity): semantic similarity beyond n-gram overlap.
    - **Perplexity** (from scratch): intrinsic language model quality measure.
    - **Task-specific metrics**: exact match + token F1 for QA; pass@k for code generation.
    - **pass@k formula**: $1 - \binom{n-c}{k}/\binom{n}{k}$ — unbiased estimator.
    - **Production regression testing**: golden dataset, metric distributions, drift alerting.
    - **Benchmark suite**: MMLU, GSM8K, HumanEval, TruthfulQA — what each measures.

    **Why it matters**
    - Every model update or prompt change needs a measurement of impact. Without evaluation
      metrics, you are guessing. BLEU/ROUGE are easy to implement and fast; BERTScore is
      more semantically aware. Pass@k is the industry standard for code generation.
      Perplexity is used for language model selection and data quality filtering.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **BLEU (Papineni et al., 2002).** Bilingual Evaluation Understudy. Introduced at IBM
    for machine translation. Computes n-gram precision of the candidate translation against
    one or more reference translations, with a brevity penalty to prevent trivially short
    outputs. Still the most widely reported MT metric despite well-known limitations.

    **ROUGE (Lin, 2004).** Recall-Oriented Understudy for Gisting Evaluation. Designed
    for summarisation (where recall of key content matters more than precision). ROUGE-N
    (n-gram recall), ROUGE-L (longest common subsequence). Standard for summarisation tasks.

    **BERTScore (Zhang et al., 2020).** Uses pre-trained BERT to compute token-level
    cosine similarities between candidate and reference, then takes F1 of best-match scores.
    Overcomes BLEU/ROUGE's weakness: "The president declared war" ≈ "The head of state
    announced hostilities" but BLEU = 0 (no n-gram overlap). BERTScore captures semantics.

    **Perplexity (Shannon, 1948 / Brown et al., 1992).** A language model's perplexity
    on a test set measures how well it predicts unseen text. Lower perplexity = better model.
    Derived from cross-entropy: $PP = 2^{H}$ where $H$ is the per-token cross-entropy.

    **pass@k (Chen et al., 2021 — HumanEval paper).** Measures code generation quality:
    the probability that at least one of k generated samples passes all unit tests.
    Uses an unbiased estimator to avoid sampling all n candidates.

    **MMLU (Hendrycks et al., 2021).** 57-subject multiple-choice benchmark spanning STEM,
    humanities, and social science. Tests world knowledge and reasoning. Standard LLM capability metric.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **BLEU intuition:**
    ```
    Reference:  "The cat sat on the mat"
    Candidate:  "The cat sat on the mat"  → BLEU = 1.0 (perfect)
    Candidate:  "A cat on the mat"        → BLEU ≈ 0.5 (partial n-gram overlap)
    Candidate:  "The dog ate the mat"     → BLEU ≈ 0.2 (few n-gram matches)
    Candidate:  "The"                     → BLEU ≈ 0.0 (brevity penalty kills it)
    ```

    **ROUGE-L intuition:**
    ```
    Reference: "The cat sat on the mat"
    LCS:       "cat sat mat" → length 3
    ROUGE-L = LCS / len(reference) [recall] or LCS / len(candidate) [precision]
    ```

    **BERTScore intuition:**
    ```
    "The president declared war" ↔ "The head of state announced hostilities"
    BLEU: 0.0 (no n-gram match)
    BERTScore: ~0.85 (embeddings of "president" ≈ "head of state"; "declared" ≈ "announced")
    ```

    **Perplexity intuition:**
    - Low perplexity: model assigns high probability to the test text → "not surprised."
    - High perplexity: model is "confused" by the test text → poor fit.
    - PP = 1: perfect prediction. PP = V (vocabulary size): random model.

    **pass@k intuition:**
    ```
    n=10 samples generated, c=3 pass tests.
    pass@1:  3/10 = 30% (average 1 sample)
    pass@5:  P(at least 1 of 5 passes) = 1 - C(7,5)/C(10,5) ≈ 83%
    pass@10: P(at least 1 of 10 passes) = 1 - C(7,10)/C(10,10) = 1 (if c≥1)
    ```
    """),

    code(r"""
    import re
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    from itertools import combinations

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 BLEU

    Let $w$ = tokenised candidate, $r$ = tokenised reference.

    **Modified n-gram precision:**
    $p_n = \frac{\sum_{\text{n-gram} \in w} \min(\text{count}(\text{n-gram}, w), \text{count}(\text{n-gram}, r))}{\sum_{\text{n-gram} \in w} \text{count}(\text{n-gram}, w)}$

    **Brevity penalty:**
    $BP = \begin{cases} 1 & |w| > |r| \\ e^{1 - |r|/|w|} & |w| \le |r| \end{cases}$

    **BLEU-N:**
    $\text{BLEU-N} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$

    with uniform weights $w_n = 1/N$.

    ### 4.2 ROUGE-L

    Let $\text{LCS}(w, r)$ = length of longest common subsequence.

    $R_{\text{lcs}} = \frac{\text{LCS}(w,r)}{|r|}$ (recall), $P_{\text{lcs}} = \frac{\text{LCS}(w,r)}{|w|}$ (precision)

    $\text{ROUGE-L} = \frac{(1+\beta^2) P_{\text{lcs}} R_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}$ with $\beta \to \infty$ (recall-focused).

    ### 4.3 BERTScore

    Given contextual embeddings $\{\mathbf{x}_i\}$ (candidate) and $\{\mathbf{y}_j\}$ (reference):

    $P_{\text{BERT}} = \frac{1}{|w|}\sum_{i} \max_j \cos(\mathbf{x}_i, \mathbf{y}_j)$ (precision)

    $R_{\text{BERT}} = \frac{1}{|r|}\sum_{j} \max_i \cos(\mathbf{x}_i, \mathbf{y}_j)$ (recall)

    $F_{\text{BERT}} = \frac{2 P_{\text{BERT}} R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$

    ### 4.4 Perplexity

    For language model with per-token log-probability $\log p(w_t | w_{1:t-1})$:

    $PP = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log p(w_t | w_{1:t-1})\right) = 2^{H}$

    where $H = -\frac{1}{T}\sum_t \log_2 p(w_t | w_{1:t-1})$ is per-token cross-entropy.

    ### 4.5 pass@k (unbiased estimator)

    Given $n$ total samples and $c$ passing samples:

    $\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$

    This estimates P(at least one of k samples passes) without actually sampling $k$ times.
    Using this estimator is unbiased and avoids variance from choosing which k samples to use.
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a — BLEU from scratch
    """),

    code(r"""
    # 5a. BLEU from scratch: n-gram precision + brevity penalty.

    def tokenise(text):
        return text.lower().split()

    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def modified_ngram_precision(candidate_tokens, reference_tokens, n):
        cand_ngrams = Counter(ngrams(candidate_tokens, n))
        ref_ngrams  = Counter(ngrams(reference_tokens, n))
        if not cand_ngrams:
            return 0.0
        # Clipped count: min of candidate count and reference count.
        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        return clipped / total

    def brevity_penalty(candidate_tokens, reference_tokens):
        c = len(candidate_tokens)
        r = len(reference_tokens)
        if c >= r:
            return 1.0
        return math.exp(1 - r / c)

    def bleu_score(candidate, reference, max_n=4):
        cand_toks = tokenise(candidate)
        ref_toks  = tokenise(reference)
        bp = brevity_penalty(cand_toks, ref_toks)
        precisions = []
        for n in range(1, max_n + 1):
            pn = modified_ngram_precision(cand_toks, ref_toks, n)
            if pn == 0:
                return 0.0   # log(0) = -inf; standard BLEU returns 0.
            precisions.append(math.log(pn))
        avg_log_prec = sum(precisions) / max_n
        return bp * math.exp(avg_log_prec)

    test_pairs = [
        ('The cat sat on the mat',
         'The cat sat on the mat',
         'Perfect match'),
        ('A cat sat on the mat',
         'The cat sat on the mat',
         '1 word change'),
        ('The dog ate the hat',
         'The cat sat on the mat',
         '2 words match'),
        ('The cat',
         'The cat sat on the mat',
         'Too short (BP kicks in)'),
        ('Something completely different and unrelated to anything here',
         'The cat sat on the mat',
         'No match'),
    ]

    print('BLEU scores from scratch:')
    print(f'  {"Description":35s} BLEU-4')
    for cand, ref, desc in test_pairs:
        score = bleu_score(cand, ref)
        print(f'  {desc:35s} {score:.4f}')
    """),

    md(r"""
    ### 5b — ROUGE-L from scratch
    """),

    code(r"""
    # 5b. ROUGE-L: longest common subsequence based metric.

    def lcs_length(a, b):
        # Dynamic programming LCS length.
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def rouge_l(candidate, reference, beta=1.0):
        cand = tokenise(candidate)
        ref  = tokenise(reference)
        lcs  = lcs_length(cand, ref)
        if lcs == 0:
            return 0.0, 0.0, 0.0
        prec = lcs / len(cand) if cand else 0.0
        rec  = lcs / len(ref)  if ref  else 0.0
        if prec + rec == 0:
            return 0.0, 0.0, 0.0
        f1 = (1 + beta**2) * prec * rec / (rec + beta**2 * prec)
        return round(prec, 4), round(rec, 4), round(f1, 4)

    summaries = [
        ('The transformer model uses attention to capture long-range dependencies.',
         'The transformer architecture uses self-attention mechanisms for long-range context.',
         'Good summary'),
        ('The model is fast and efficient with good performance on all tasks.',
         'The transformer architecture uses self-attention mechanisms for long-range context.',
         'Vague summary'),
        ('Attention mechanism uses Q, K, V matrices and softmax normalisation.',
         'The transformer architecture uses self-attention mechanisms for long-range context.',
         'Different phrasing'),
    ]

    print('ROUGE-L scores from scratch:')
    print(f'  {"Description":25s} {"P_lcs":8s} {"R_lcs":8s} {"F1_lcs":8s}')
    for cand, ref, desc in summaries:
        p, r, f1 = rouge_l(cand, ref)
        print(f'  {desc:25s} {p:.4f}   {r:.4f}   {f1:.4f}')
    """),

    md(r"""
    ### 5c — BERTScore (cosine similarity approximation)
    """),

    code(r"""
    # 5c. BERTScore: token-level cosine similarity (simplified — uses hash embeddings).
    # In production: use transformers.BERTScore with a real BERT model.

    def word_embed(word, dim=32):
        rng_w = np.random.default_rng(abs(hash(word.lower())) % (2**31))
        v = rng_w.standard_normal(dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def bert_score_approx(candidate, reference, dim=32):
        cand_toks = tokenise(candidate)
        ref_toks  = tokenise(reference)
        cand_embs = np.array([word_embed(w, dim) for w in cand_toks])
        ref_embs  = np.array([word_embed(w, dim) for w in ref_toks])
        if len(cand_embs) == 0 or len(ref_embs) == 0:
            return 0.0, 0.0, 0.0
        # Similarity matrix: [len_cand, len_ref].
        sim_matrix = cand_embs @ ref_embs.T   # cosine (embeddings normalised)
        # Precision: for each candidate token, max similarity to any reference token.
        p_bert = float(sim_matrix.max(axis=1).mean())
        # Recall: for each reference token, max similarity to any candidate token.
        r_bert = float(sim_matrix.max(axis=0).mean())
        if p_bert + r_bert == 0:
            return 0.0, 0.0, 0.0
        f_bert = 2 * p_bert * r_bert / (p_bert + r_bert)
        return round(p_bert, 4), round(r_bert, 4), round(f_bert, 4)

    bert_test_pairs = [
        ('The president declared war on the nation.',
         'The head of state announced hostilities against the country.',
         'Semantic paraphrase'),
        ('The cat sat on the mat.',
         'The cat sat on the mat.',
         'Perfect match'),
        ('Neural networks learn from data via gradient descent.',
         'Deep learning models optimise using backpropagation.',
         'Related but different wording'),
        ('The sky is blue and birds sing.',
         'The cat sat on the mat.',
         'Unrelated'),
    ]

    print('BERTScore approximation (hash embeddings):')
    print(f'  {"Description":30s} {"P_BERT":8s} {"R_BERT":8s} {"F_BERT":8s}')
    for cand, ref, desc in bert_test_pairs:
        p, r, f = bert_score_approx(cand, ref)
        print(f'  {desc:30s} {p:.4f}   {r:.4f}   {f:.4f}')

    print('\nNote: hash embeddings are non-semantic — in production, use BERT/sentence-transformer embeddings.')
    print('BERTScore advantage: captures "president" ≈ "head of state" that BLEU misses.')
    """),

    md(r"""
    ### 5d — Perplexity from scratch
    """),

    code(r"""
    # 5d. Perplexity from scratch: bigram LM with Laplace smoothing.

    class BigramLM:
        # Simple bigram language model.
        def __init__(self):
            self.unigram_counts = Counter()
            self.bigram_counts  = Counter()
            self.vocab = set()

        def train(self, corpus):
            for sentence in corpus:
                tokens = ['<s>'] + tokenise(sentence) + ['</s>']
                self.vocab.update(tokens)
                for t in tokens:
                    self.unigram_counts[t] += 1
                for i in range(len(tokens)-1):
                    self.bigram_counts[(tokens[i], tokens[i+1])] += 1
            self.V = len(self.vocab)

        def log_prob(self, w1, w2, alpha=1.0):
            # Laplace-smoothed bigram probability.
            bigram_count = self.bigram_counts[(w1, w2)]
            unigram_count = self.unigram_counts[w1]
            return math.log((bigram_count + alpha) / (unigram_count + alpha * self.V))

        def perplexity(self, sentence):
            tokens = ['<s>'] + tokenise(sentence) + ['</s>']
            T = len(tokens) - 1
            if T == 0:
                return float('inf')
            log_prob_sum = sum(self.log_prob(tokens[i], tokens[i+1]) for i in range(T))
            return math.exp(-log_prob_sum / T)

    TRAIN_CORPUS = [
        'the cat sat on the mat',
        'the dog ran in the park',
        'the cat ran on the mat',
        'the dog sat in the park',
        'neural networks learn from data',
        'machine learning models train on data',
        'deep learning uses gradient descent',
        'attention mechanisms power transformers',
        'language models predict the next token',
        'perplexity measures language model quality',
    ]

    lm = BigramLM()
    lm.train(TRAIN_CORPUS)

    test_sentences = [
        ('the cat sat on the mat', 'In-domain (training-like)'),
        ('the dog ran on the mat', 'Slightly novel'),
        ('neural networks process data', 'Related domain'),
        ('quantum physics explains molecular behaviour', 'Out-of-domain'),
        ('the the the the the', 'Repetitive degenerate'),
    ]

    print('Perplexity evaluation:')
    print(f'  {"Sentence":45s} {"Perplexity":12s} {"Category"}')
    for sentence, category in test_sentences:
        pp = lm.perplexity(sentence)
        print(f'  {sentence[:45]:45s} {pp:12.2f}  {category}')
    print(f'\nVocabulary size: {lm.V}')
    print('Lower perplexity = model is less "surprised" by the text.')
    """),

    md(r"""
    ### 5e — QA evaluation: exact match and token F1
    """),

    code(r"""
    # 5e. Question answering metrics: exact match and token-level F1.

    def normalise_answer(text):
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def exact_match(prediction, ground_truth):
        return int(normalise_answer(prediction) == normalise_answer(ground_truth))

    def token_f1(prediction, ground_truth):
        pred_toks = normalise_answer(prediction).split()
        gt_toks   = normalise_answer(ground_truth).split()
        pred_counter = Counter(pred_toks)
        gt_counter   = Counter(gt_toks)
        common = sum((pred_counter & gt_counter).values())
        if common == 0:
            return 0.0
        precision = common / len(pred_toks)
        recall    = common / len(gt_toks)
        return 2 * precision * recall / (precision + recall)

    qa_examples = [
        ('The Eiffel Tower is in Paris, France.',        'Paris',               'Partial answer'),
        ('Paris',                                         'Paris',               'Exact match'),
        ('The tower is located in Paris',                 'Paris',               'Contains answer'),
        ('London',                                        'Paris',               'Wrong answer'),
        ('The French capital city, Paris, is in France.', 'Paris, France',       'Verbose but correct'),
    ]

    print('QA evaluation: Exact Match and Token F1')
    print(f'  {"Prediction":42s} {"Ground truth":20s} {"EM":4s} {"F1":8s}')
    for pred, gt, desc in qa_examples:
        em = exact_match(pred, gt)
        f1 = token_f1(pred, gt)
        print(f'  {pred[:42]:42s} {gt:20s} {em:4d} {f1:.3f}')
    """),

    md(r"""
    ### 5f — pass@k for code generation
    """),

    code(r"""
    # 5f. pass@k: unbiased estimator for code generation quality.

    from math import comb

    def pass_at_k(n, c, k):
        # P(at least 1 of k randomly drawn samples passes tests).
        # n = total samples generated, c = number that pass, k = samples drawn.
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)

    # Simulate HumanEval-style results for 5 coding problems.
    problems = [
        {'name': 'find_max',        'n': 10, 'c': 8},   # easy: 8/10 pass
        {'name': 'binary_search',   'n': 10, 'c': 6},   # medium: 6/10 pass
        {'name': 'lru_cache',       'n': 10, 'c': 3},   # hard: 3/10 pass
        {'name': 'regex_parser',    'n': 10, 'c': 1},   # very hard: 1/10 pass
        {'name': 'dp_knapsack',     'n': 10, 'c': 0},   # impossible: 0/10 pass
    ]

    print('pass@k for code generation (n=10 samples per problem):')
    print(f'  {"Problem":18s} {"c":4s} {"pass@1":8s} {"pass@3":8s} {"pass@5":8s} {"pass@10"}')
    for prob in problems:
        n, c = prob['n'], prob['c']
        p1  = pass_at_k(n, c, 1)
        p3  = pass_at_k(n, c, 3)
        p5  = pass_at_k(n, c, 5)
        p10 = pass_at_k(n, c, 10)
        print(f'  {prob["name"]:18s} {c:4d} {p1:.4f}   {p3:.4f}   {p5:.4f}   {p10:.4f}')

    # Overall pass@k across all problems.
    for k in [1, 3, 5]:
        mean_p = np.mean([pass_at_k(p['n'], p['c'], k) for p in problems])
        print(f'\nMean pass@{k}: {mean_p:.4f}')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — BLEU and ROUGE-L across output quality spectrum.
    quality_levels = {
        'Perfect': ('The cat sat on the mat', 'The cat sat on the mat'),
        'Near-perfect': ('The cat sat on the mat today', 'The cat sat on the mat'),
        'Good': ('A cat was sitting on a mat', 'The cat sat on the mat'),
        'Partial': ('Cat sitting mat', 'The cat sat on the mat'),
        'Poor': ('The dog played in the park', 'The cat sat on the mat'),
        'Unrelated': ('Quantum physics is fascinating', 'The cat sat on the mat'),
    }
    labels_q = list(quality_levels.keys())
    bleu_scores  = [bleu_score(c, r) for c, r in quality_levels.values()]
    rouge_scores = [rouge_l(c, r)[2] for c, r in quality_levels.values()]

    x_q = np.arange(len(labels_q))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_q, bleu_scores,  'o-', color='steelblue', label='BLEU-4', lw=2)
    ax.plot(x_q, rouge_scores, 's-', color='seagreen',  label='ROUGE-L F1', lw=2)
    ax.set_xticks(x_q); ax.set_xticklabels(labels_q, rotation=15)
    ax.set_ylabel('Score'); ax.set_ylim(-0.05, 1.1)
    ax.set_title('Figure 1 — BLEU and ROUGE-L across quality levels')
    ax.legend()
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 1.** BLEU-4 and ROUGE-L F1 across a quality spectrum from perfect to unrelated.
    Both metrics decline with quality, but at different rates. **BLEU** is more sensitive to
    n-gram overlap — even small word changes cause a bigger BLEU drop than ROUGE-L drop.
    **ROUGE-L** uses LCS (preserves word order but allows gaps) — more lenient than BLEU.
    Key limitation: both score 0.0 for semantically equivalent but differently-worded text
    ("A feline reclined upon the carpet" vs. "The cat sat on the mat"). This is why
    BERTScore was developed — it captures meaning beyond surface overlap.
    """),

    code(r"""
    # Figure 2 — pass@k curves for problems of varying difficulty.
    k_range = list(range(1, 11))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_p = ['seagreen', 'steelblue', 'orange', 'coral', 'gray']
    for prob, color in zip(problems, colors_p):
        n, c = prob['n'], prob['c']
        pk_vals = [pass_at_k(n, c, k) for k in k_range]
        ax.plot(k_range, pk_vals, 'o-', color=color,
                label=f'{prob["name"]} (c={c}/n={n})', lw=2)
    ax.set_xlabel('k (samples drawn)'); ax.set_ylabel('pass@k')
    ax.set_title('Figure 2 — pass@k curves by problem difficulty')
    ax.legend(fontsize=9); ax.set_ylim(-0.05, 1.1)
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 2.** pass@k curves for five problems of varying difficulty. **Easy problems**
    (c=8/10, green) reach pass@k ≈ 1.0 at k=3 — almost any 3 samples will include at
    least one passing solution. **Hard problems** (c=1/10, orange) only reach pass@k ≈ 0.65
    at k=10 — you need many samples to have a good chance. **Impossible problems** (c=0/10,
    gray) have pass@k = 0 for all k. This curve structure is why pass@k is reported for
    multiple k values: pass@1 measures "single-shot quality", pass@10 measures "best-of-10
    quality." The gap between pass@1 and pass@10 measures how much diversity helps.
    """),

    code(r"""
    # Figure 3 — Perplexity comparison: in-domain vs. out-of-domain.
    corpus_sizes = [10, 50, 200, 1000]
    domains = {
        'ML/AI text': [
            'neural networks learn from training data',
            'gradient descent optimises the loss function',
            'attention mechanism computes query key value matrices',
            'transformers use self-attention for sequence modelling',
            'language models are trained on large text corpora',
        ] * 10,
        'Legal text': [
            'the defendant shall pay damages to the plaintiff',
            'pursuant to section three of the contract hereinafter',
            'indemnification clauses protect parties from liability',
            'the court ruled in favour of the petitioner',
            'arbitration proceedings shall be conducted confidentially',
        ] * 10,
    }

    in_domain_test  = ['attention mechanism learns to focus on relevant tokens']
    out_domain_test = ['the defendant shall pay damages pursuant to the contract']

    pp_in, pp_out = [], []
    for size in corpus_sizes:
        for domain_corpus, test_sents, label_list in [
            (domains['ML/AI text'][:size], in_domain_test, pp_in),
            (domains['ML/AI text'][:size], out_domain_test, label_list := pp_out),
        ]:
            lm_tmp = BigramLM()
            lm_tmp.train(domain_corpus[:size])
            pp = lm_tmp.perplexity(test_sents[0])
            label_list.append(min(pp, 500))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(corpus_sizes, pp_in,  'o-', color='steelblue', label='In-domain (ML text)', lw=2)
    ax.plot(corpus_sizes, pp_out, 's-', color='coral',     label='Out-of-domain (legal text)', lw=2)
    ax.set_xlabel('Training corpus size (sentences)')
    ax.set_ylabel('Perplexity (log scale)'); ax.set_yscale('log')
    ax.set_title('Figure 3 — Perplexity vs. corpus size: in-domain vs. out-of-domain')
    ax.legend(); plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 3.** Perplexity vs. training corpus size for in-domain and out-of-domain
    text. As training data grows, the LM improves on in-domain text (perplexity drops).
    Out-of-domain text remains high perplexity — the model trained on ML/AI text is
    "confused" by legal language. **Production use of perplexity:**
    1. **Data quality filtering**: filter training data to keep only low-perplexity examples
       (per a reference LM) — removes noise, code snippets, non-text.
    2. **Domain shift detection**: if production text perplexity rises, the model is
       encountering distribution shift — time to retrain or add domain data.
    3. **Model selection**: compare two LMs on held-out text — lower perplexity wins
       (all else equal).
    """),

    code(r"""
    # Figure 4 — Metric comparison for summarisation task.
    summaries_eval = [
        {
            'label': 'Extractive (verbatim copy)',
            'candidate': 'The attention mechanism uses Q, K, V matrices and softmax normalisation. Multi-head attention uses H parallel heads concatenated and projected.',
            'reference':  'The attention mechanism uses Q, K, V matrices. Multi-head attention runs H parallel heads then concatenates and projects the outputs.',
        },
        {
            'label': 'Abstractive (paraphrase)',
            'candidate': 'Transformers compute attention via query, key, and value projections, with multiple heads working in parallel for richer representations.',
            'reference':  'The attention mechanism uses Q, K, V matrices. Multi-head attention runs H parallel heads then concatenates and projects the outputs.',
        },
        {
            'label': 'Too brief',
            'candidate': 'Attention uses Q, K, V.',
            'reference':  'The attention mechanism uses Q, K, V matrices. Multi-head attention runs H parallel heads then concatenates and projects the outputs.',
        },
        {
            'label': 'Off-topic',
            'candidate': 'Gradient descent minimises the loss function using the negative gradient direction.',
            'reference':  'The attention mechanism uses Q, K, V matrices. Multi-head attention runs H parallel heads then concatenates and projects the outputs.',
        },
    ]

    bleu_s = [bleu_score(s['candidate'], s['reference']) for s in summaries_eval]
    rouge_s = [rouge_l(s['candidate'], s['reference'])[2] for s in summaries_eval]
    bert_s  = [bert_score_approx(s['candidate'], s['reference'])[2] for s in summaries_eval]

    x_s = np.arange(len(summaries_eval))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x_s - w, bleu_s,  w, label='BLEU-4',    color='steelblue', alpha=0.8)
    ax.bar(x_s,     rouge_s, w, label='ROUGE-L',   color='seagreen',  alpha=0.8)
    ax.bar(x_s + w, bert_s,  w, label='BERTScore', color='coral',     alpha=0.8)
    ax.set_xticks(x_s); ax.set_xticklabels([s['label'] for s in summaries_eval], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.2)
    ax.set_title('Figure 4 — BLEU vs. ROUGE-L vs. BERTScore for summarisation')
    ax.legend(); plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 4.** Metric comparison for summarisation across four candidate types.
    **Extractive summary** (verbatim copy): high BLEU and ROUGE-L (exact n-gram overlap),
    moderate BERTScore. **Abstractive summary** (good paraphrase): BLEU and ROUGE-L drop
    significantly (different words), but BERTScore stays high — it captures semantic
    equivalence. **Too brief**: ROUGE-L drops (low recall); BLEU drops (brevity penalty).
    **Off-topic**: all metrics near 0. The key takeaway: for tasks where paraphrasing
    is expected (summarisation, dialogue), **BERTScore is more appropriate** than BLEU.
    For translation (where fidelity to wording matters), **BLEU remains standard**.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **BLEU gaming** | Model generates short, high-precision outputs | Brevity penalty not strong enough; metric optimised directly | Use ROUGE-L + BERTScore alongside; human evaluation |
    | **ROUGE-L doesn't penalise repetition** | Repetitive output scores high | LCS ignores repeated subsequences | Use ROUGE-L + factual consistency check |
    | **BERTScore model sensitivity** | Scores change when BERT model updated | Contextual embeddings version-dependent | Pin BERT model version; recalibrate when updating |
    | **Perplexity doesn't correlate with task quality** | Low-perplexity model generates bad outputs | Perplexity measures fluency, not factual accuracy | Use perplexity as auxiliary metric; not sole metric |
    | **pass@k sampling variance** | pass@k varies widely across runs | Small n; high variance in c | Use large n (≥50); report 95% CI; use unbiased estimator |
    | **Reference quality problem** | High BLEU/ROUGE with poor references | Reference answers are themselves poor | Curate references carefully; use multiple references |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 Evaluation libraries (guarded).
    try:
        from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
        print('NLTK BLEU available.')
    except ImportError:
        print('[nltk not installed] pip install nltk → from nltk.translate.bleu_score import corpus_bleu')

    try:
        from rouge_score import rouge_scorer  # noqa: F401
        print('rouge_score available.')
    except ImportError:
        print('[rouge-score not installed] pip install rouge-score → from rouge_score import rouge_scorer')

    try:
        import bert_score  # noqa: F401
        print('bert_score available.')
    except ImportError:
        print('[bert_score not installed] pip install bert-score → from bert_score import score as bert_score_fn')

    lines = [
        '',
        'Production BERTScore usage:',
        '  from bert_score import score',
        '  P, R, F1 = score(candidates, references, lang="en", model_type="microsoft/deberta-xlarge-mnli")',
        '',
        'Production ROUGE:',
        '  from rouge_score import rouge_scorer',
        '  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])',
        '  scores = scorer.score(reference, candidate)',
        '  # scores["rougeL"].fmeasure',
    ]
    print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 Production regression testing pattern.

    GOLDEN_DATASET = [
        {'input': 'Explain attention mechanism', 'ref': 'Attention uses Q, K, V matrices and softmax to weight values.'},
        {'input': 'What is gradient descent?', 'ref': 'Gradient descent minimises loss by moving in the negative gradient direction.'},
        {'input': 'Define NDCG', 'ref': 'NDCG normalises DCG by the ideal DCG. DCG weights relevance by log of position.'},
    ]

    def evaluate_model_on_golden(model_outputs, golden_dataset):
        scores = {'bleu': [], 'rouge_l': [], 'em': []}
        for output, example in zip(model_outputs, golden_dataset):
            scores['bleu'].append(bleu_score(output, example['ref']))
            scores['rouge_l'].append(rouge_l(output, example['ref'])[2])
            scores['em'].append(exact_match(output, example['ref']))
        return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}

    # Simulate model A (good) vs. model B (regressed).
    model_a_outputs = [
        'Attention computes Q, K, V dot products with softmax normalisation to weight the values.',
        'Gradient descent minimises the loss function using the negative gradient of the loss.',
        'NDCG is normalised discounted cumulative gain, computed as DCG divided by ideal DCG.',
    ]
    model_b_outputs = [   # regressed model — worse outputs.
        'The model processes inputs.',
        'Optimisation is the goal.',
        'Ranking quality can be measured.',
    ]

    scores_a = evaluate_model_on_golden(model_a_outputs, GOLDEN_DATASET)
    scores_b = evaluate_model_on_golden(model_b_outputs, GOLDEN_DATASET)

    print('Production regression test:')
    print(f'  {"Metric":10s} {"Model A":10s} {"Model B":10s} {"Delta":10s} {"Alert?"}')
    sla_thresholds = {'bleu': 0.10, 'rouge_l': 0.30, 'em': 0.0}
    for metric in ['bleu', 'rouge_l', 'em']:
        a, b = scores_a[metric], scores_b[metric]
        delta = b - a
        alert = 'ALERT' if abs(delta) > sla_thresholds[metric] and delta < 0 else 'OK'
        print(f'  {metric:10s} {a:10.4f} {b:10.4f} {delta:+10.4f} {alert}')
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — LLM Regression Testing

    **Scenario.** An AI product team deploys Claude for customer-facing writing assistance.
    They update the system prompt monthly to improve tone and task coverage. Each update
    needs a regression test to ensure it doesn't degrade performance on existing use cases.

    **Golden dataset:** 300 queries with reference outputs, curated by the team and
    validated by human raters. Covers: email drafting (100), summarisation (100), Q&A (100).

    **Evaluation pipeline (runs every deployment):**
    1. Run all 300 golden queries through the new model version.
    2. Compute BLEU, ROUGE-L, BERTScore, exact match per category.
    3. Alert if any metric drops > 5% from previous version baseline.
    4. Auto-block deployment if any metric drops > 10%.
    5. Human review sample (20 queries) for any blocked deployment.

    **Incident caught:** System prompt update in month 3 changed tone instructions,
    which caused the model to produce shorter, bullet-point-heavy outputs. ROUGE-L
    dropped 12% on email drafting tasks (users expect full paragraphs). Deployment blocked.
    Root cause: tone instruction conflicted with length expectations. Fixed by adding
    explicit length guidance. Estimated value: prevented rollout to 50,000 daily active users.

    **Cost:** Full evaluation run = 300 API calls × $0.01 = $3.00. Scheduled daily.
    Monthly cost: $90. Value of one prevented regression: user churn prevention worth
    estimated $50,000+.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Multiple references.** Single reference BLEU/ROUGE is noisy — there are many valid
      outputs for any task. Use multiple references (3–5) and compute max BLEU across references.
      NLTK's `corpus_bleu` supports multiple references natively.
    - **Domain-specific calibration.** BLEU scores mean different things for different tasks.
      Translation BLEU > 0.40 = near-human quality. Summarisation BLEU > 0.15 = competitive.
      Raw numbers without context are misleading — always compare to a baseline.
    - **Statistical significance.** Metric differences of < 1 BLEU point are rarely significant.
      Use bootstrap resampling to compute confidence intervals before declaring one model better.
    - **pass@k sample size.** For reliable pass@k estimates, use n ≥ 50 (not just n=5).
      Small n causes high variance. Report 95% CI via bootstrap or Wilson interval.
    - **Evaluation regression testing in CI.** Add golden dataset evaluation to your CI/CD
      pipeline. Pull request must not regress metrics beyond threshold before merging.
      Fail the PR if ROUGE-L drops > 5% on the summarisation golden set.
    - **Human-in-the-loop.** Automatic metrics miss fluency, factual accuracy, and instruction
      following nuances. 10% of automatic evaluation should be verified by human raters.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Metric selection guide:**

    | Metric | Task | Strength | Weakness |
    |---|---|---|---|
    | BLEU | Translation | Standard; fast | n-gram overlap only; no semantics |
    | ROUGE-L | Summarisation | LCS captures order | Doesn't penalise repetition; no semantics |
    | BERTScore | Any text generation | Semantic similarity | Model-dependent; slower |
    | Perplexity | LM quality / data quality | Intrinsic; no references needed | Doesn't measure usefulness |
    | Exact Match | QA (structured) | Binary, unambiguous | Requires exact string match |
    | Token F1 | QA | Partial credit | Doesn't penalise fluency issues |
    | pass@k | Code generation | Tests actual correctness | Requires unit tests; n must be large |

    **When to use what:**
    - Translation: BLEU (+ human BLEU critique for high-stakes)
    - Summarisation: ROUGE-L + BERTScore + Faithfulness
    - QA: Exact Match + Token F1 + answer correctness
    - Code generation: pass@k (k=1 for deployment quality, k=10 for capability ceiling)
    - General instruction following: BERTScore + LLM-as-judge
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"Explain BLEU score. What are its limitations?"* → BLEU computes modified n-gram
      precision (clipped to reference count) with a brevity penalty to prevent trivially short
      outputs. It's fast and reproducible. Limitations: (1) no semantic understanding — synonyms
      score 0; (2) single reference is noisy; (3) BLEU optimisation creates pathological outputs
      (repetitive, short); (4) different tasks have different BLEU scales — 0.30 is good for
      translation, mediocre for abstractive summarisation.
    - *"What is pass@k and how is the unbiased estimator computed?"* → pass@k = P(at least one
      of k random samples passes unit tests). The unbiased estimator: $1 - \binom{n-c}{k}/\binom{n}{k}$
      where n = total samples generated, c = passing samples. Unbiased because it doesn't require
      actually drawing k samples — it estimates the expected probability over all possible k-subsets.

    **Deep-dive questions**
    - *"Why is BERTScore better than ROUGE for abstractive summarisation?"* → ROUGE is n-gram
      based — it penalises any paraphrase. Abstractive summarisation correctly replaces long phrases
      with shorter equivalents (e.g. "the head of state announced" → "the president said"). BERTScore
      uses contextual embeddings to measure semantic similarity — "president" and "head of state" have
      high cosine similarity in BERT space, so BERTScore rewards the paraphrase. ROUGE would score it low.
    - *"How would you set up regression testing for an LLM update?"* → (1) Curate a golden dataset
      of 200–500 representative queries with reference outputs; (2) run on every PR/deployment;
      (3) compute BLEU, ROUGE-L, BERTScore per task category; (4) alert on > 5% regression in any
      metric on any category; (5) block on > 10% regression; (6) human review for blocked
      deployments; (7) track metric trends over time to detect slow regression.

    **Common mistakes:** using BLEU alone (misses semantics); not using a baseline (raw numbers
    meaningless); small n for pass@k (high variance); no statistical significance testing.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **BLEU formula.** What is "modified n-gram precision"? Why is clipping necessary?
    2. **ROUGE-L.** What is LCS? Give an example where ROUGE-L > BLEU for the same pair.
    3. **BERTScore advantage.** Give a concrete example where BLEU = 0 but BERTScore = 0.9.
    4. **Perplexity interpretation.** A model has perplexity 50 on in-domain text and 300 on
       out-of-domain text. What does this tell you?
    5. **pass@k formula.** n=10, c=4, k=5. Calculate pass@5 using the unbiased estimator.
    6. **Metric for code.** Why is BLEU inappropriate for code generation? What should you use?
    7. **Regression testing.** Your ROUGE-L drops from 0.42 to 0.37 after a model update.
       Is this significant? How would you check?
    8. **Reference quality.** Your BLEU score is 0.45 on a test set. A colleague's BLEU is
       0.38 on a different test set. Can you conclude your model is better? Why/why not?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Compute BLEU-2 by hand for: Candidate="the cat sat", Reference="the cat sat on the mat."
       Show n-gram counts, clipped counts, precision, brevity penalty, and final score.
    2. n=20 samples, c=12 pass. Calculate pass@1, pass@5, pass@10 using the unbiased estimator.

    **Beginner → Intermediate (coding)**
    3. Extend `bleu_score` to support multiple references: take a list of references, compute
       BLEU for each, return the maximum. Test: a candidate that matches reference B better than
       reference A should score higher with multi-reference BLEU.
    4. Implement `rouge_n` (ROUGE-N with recall, precision, F1). Verify ROUGE-1 and ROUGE-2
       on 5 summarisation examples.

    **Intermediate (analysis)**
    5. **Metric correlation study**: generate 50 (candidate, reference) pairs at varying quality.
       Compute BLEU, ROUGE-L, and BERTScore for each. Plot pairwise correlations. Which two
       metrics agree most? Which diverge most? When do they diverge?
    6. **pass@k sensitivity**: for a fixed n=20, vary c from 0 to 20. Plot pass@1, pass@5,
       pass@10 as functions of c/n (success rate). At what success rate does pass@5 cross 0.90?

    **Senior (design)**
    7. *System design:* design an automated evaluation pipeline for a code generation assistant
       (generates Python functions from natural language descriptions). 500 queries/day. Design:
       metrics (BLEU, ROUGE, pass@k), golden dataset composition, CI/CD integration, alert
       thresholds, human review workflow, evaluation cadence.
    8. *Interview:* "We're considering switching from GPT-4 to Claude for our summarisation
       product. How would you rigorously decide which model is better for our use case?"
       (Expected: curate task-specific golden dataset; evaluate ROUGE-L + BERTScore + human
       evaluation on same inputs; compute bootstrap CI for differences; A/B test on 5% of live
       traffic; measure downstream business metrics like user edit rate and satisfaction.)
    """),

    md(r"""
    ---
    ### Summary
    LLM evaluation combines reference-based metrics (BLEU for translation, ROUGE-L for
    summarisation, BERTScore for semantic tasks), intrinsic metrics (perplexity for LM
    quality), and task-specific metrics (exact match + F1 for QA, pass@k for code).
    In production: build a golden dataset, run evaluation on every model update, alert on
    regression. Never use a single metric — build a dashboard with multiple metrics per
    task category. Statistical significance testing before declaring model improvements.

    **Related lesson:** `EVAL-04 · Human Evaluation` — when automated metrics are not enough: annotation
    guidelines, inter-annotator agreement (Krippendorff's alpha, Cohen's kappa), Likert scales,
    pairwise preference, and how to run a reliable human evaluation study.
    """),
]

build("08_evaluation/02_llm_evaluation.ipynb", cells)
