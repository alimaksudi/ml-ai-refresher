"""Builder for Notebook 29 — Chunking Strategies."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 29 · Chunking Strategies
    ### Phase 5 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    > Every RAG pipeline requires splitting documents into chunks before indexing.
    > This deceptively simple step has an enormous impact on retrieval quality:
    > the wrong chunking strategy causes the LLM to answer from incomplete context,
    > miss relevant information, or hallucinate because the retrieved passage lacks
    > surrounding context. This notebook teaches every major chunking strategy from
    > scratch and the theory behind choosing the right one.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Why chunking**: context window limits, embedding model limits, retrieval
      precision vs. generation context tradeoff.
    - **Fixed-size chunking with overlap**: implementation, tuning chunk_size and
      overlap, when it works and when it fails.
    - **Sentence-boundary chunking**: respect natural language boundaries; avoid
      splitting mid-sentence.
    - **Recursive character splitting**: try separators in order (paragraph → sentence
      → word) to preserve structure while hitting a size target.
    - **Semantic chunking**: use embedding cosine distance between consecutive
      sentences to detect topic shifts; split at peaks.
    - **Parent-child chunking**: index small child chunks for precise retrieval;
      return large parent chunks for generation context.
    - **Chunk size tradeoffs**: quantitative analysis of how chunk size affects
      retrieval precision and generation quality.

    **Why it matters**
    - In production RAG, chunking is often the most impactful tuning parameter.
      A 2023 study found chunk size alone could change answer accuracy by 15–25%.
      Chunking must be co-designed with the embedding model (max sequence length)
      and the LLM context window.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Pre-RAG: document retrieval.** Classic IR systems stored full documents and
    returned them wholesale. This worked for short web pages but broke for long
    documents — a 50-page PDF contains hundreds of distinct facts; returning all 50
    pages for every query wastes the LLM's context window.

    **Early RAG (2020).** DPR and RAG (Lewis et al., 2020) chunked Wikipedia into
    100-word passages. Fixed-size chunking was the only strategy discussed. It worked
    for Wikipedia (fairly uniform paragraph structure) but failed for heterogeneous
    documents.

    **LangChain popularises chunking strategies (2022–2023).** LangChain's
    `RecursiveCharacterTextSplitter` introduced the "try separators in priority order"
    approach, becoming the de facto standard for production RAG.

    **Semantic chunking (Greg Kamradt, 2023).** The idea of splitting on embedding
    distance between consecutive sentences emerged from the RAG community in 2023.
    It automatically detects topic transitions without rule-based heuristics.

    **Parent-child chunking (2023).** As RAG practitioners noticed that small chunks
    improved retrieval precision but reduced generation context, the parent-child
    pattern emerged: index fine-grained child chunks, retrieve them, then expand to
    the parent document or section for generation.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The fundamental tension.**

    | Dimension | Small chunks | Large chunks |
    |---|---|---|
    | Retrieval precision | High (specific match) | Low (many irrelevant sentences) |
    | Generation context | Low (missing context) | High (full background) |
    | Embedding quality | Good (fits model window) | Bad if > 512 tokens |
    | Storage | High (more chunks) | Low (fewer chunks) |

    **Chunking strategies on a spectrum.**
    ```
    Fixed-size ──► Sentence ──► Recursive ──► Semantic ──► Parent-child
    Simplest                                               Most sophisticated
    Fast                                                   Slower (needs embeddings)
    Ignores structure                                      Respects meaning
    ```

    **Recursive splitting intuition.** A document has natural hierarchy: paragraphs
    → sentences → words. The algorithm tries to split on `\n\n` (paragraphs) first.
    If any resulting chunk is still too large, split it on `\n` (lines), then `.` (sentences),
    then ` ` (words). Each level is a fallback. This preserves structure while
    guaranteeing a maximum chunk size.

    **Semantic chunking intuition.** Embed each sentence. Compute cosine similarity
    between consecutive sentence embeddings. Where similarity *drops* (two consecutive
    sentences are about different topics), insert a chunk boundary. The threshold
    controls sensitivity: low threshold = many small topic-aware chunks; high
    threshold = fewer, coarser chunks.

    **Parent-child intuition.** Think of a textbook: each paragraph is a child chunk
    (indexed for retrieval); each section is a parent chunk (sent to the LLM for
    generation). The retrieval needle-in-a-haystack is done at child granularity;
    the generation is done with full section context.
    """),

    code(r"""
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (9, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Sample technical document: Python documentation excerpt (simulated).
    DOCUMENT = (
        'Python is a high-level, general-purpose programming language. '
        'Its design philosophy emphasises code readability with the use of significant indentation. '
        'Python is dynamically typed and garbage-collected. '
        'It supports multiple programming paradigms, including structured, object-oriented and functional programming.\n\n'
        'Python was created by Guido van Rossum and first released in 1991. '
        'It is consistently ranked as one of the most popular programming languages. '
        'The language\'s core philosophy is summarised in the document The Zen of Python.\n\n'
        'Python\'s standard library is very extensive. '
        'The library contains built-in modules that provide access to system functionality. '
        'There are also many third-party libraries available through the Python Package Index. '
        'Popular libraries include NumPy for numerical computing, Pandas for data analysis, and Flask for web development.\n\n'
        'List comprehensions provide a concise way to create lists. '
        'For example, squares = [x**2 for x in range(10)] creates a list of squares. '
        'Generator expressions are similar but produce values lazily. '
        'Dictionary comprehensions work in an analogous way for creating dictionaries.\n\n'
        'Exception handling uses try, except, else, and finally blocks. '
        'The raise statement is used to throw an exception. '
        'Custom exceptions are defined by subclassing the Exception class. '
        'Context managers use the with statement to ensure proper resource cleanup.'
    )

    print(f'Document length: {len(DOCUMENT)} characters, '
          f'~{len(DOCUMENT.split())} words')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Chunk size in tokens

    Most embedding models have a **maximum sequence length** $L_{\max}$ (e.g. 512
    tokens for BERT-based models). The rule:
    $$\text{chunk\_size\_chars} \approx \text{chunk\_size\_tokens} \times 4$$
    (rough approximation: 1 token $\approx$ 4 characters in English).

    Practical constraints:
    - chunk\_size $\leq L_{\max}$ of the embedding model (else truncation)
    - chunk\_size $\leq$ LLM context window / $k$ (to fit $k$ chunks in context)
    - Overlap $\in [10\%, 20\%]$ of chunk\_size

    ### 4.2 Semantic chunking threshold

    Let $s_i$ be the sentence embedding of sentence $i$. Define the breakpoint score:
    $$b_i = 1 - \cos(\hat{s}_i, \hat{s}_{i+1}) \in [0, 2]$$

    Split at position $i$ if $b_i > \theta$. Typical $\theta = 0.3$ (90th percentile
    of $b$ values in the document).

    ### 4.3 Retrieval precision vs. chunk size

    For a query $q$ and relevant passage $p$ of length $L_p$ words, if we use chunks
    of size $C$:
    - If $C \ll L_p$: relevant content is split across multiple chunks; each chunk
      has only partial context → low recall per chunk, many chunks needed.
    - If $C \gg L_p$: relevant passage is diluted by surrounding irrelevant content
      → lower cosine similarity to query → lower retrieval score.
    - **Optimal $C \approx L_p$**: chunk size matches passage length.

    Since $L_p$ varies across queries, a fixed $C$ is always a compromise.
    Parent-child chunking provides multiple granularities simultaneously.

    ### 4.4 Overlap effect on recall

    With chunk size $C$ and overlap $O$, a passage of length $L_p > C$ is covered
    by at least $\lceil (L_p - O) / (C - O) \rceil$ chunks. At least one chunk
    contains $\geq C - O$ tokens of the passage — ensuring that critical content is
    not split at the boundary.
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a Fixed-size chunking with overlap
    """),

    code(r"""
    # 5a. Fixed-size chunking: split at character boundaries with overlap.
    def fixed_size_chunks(text, chunk_size=200, overlap=40):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({'text': chunk, 'start': start, 'end': end})
            if end == text_len:
                break
            start += chunk_size - overlap
        return chunks

    fixed_chunks = fixed_size_chunks(DOCUMENT, chunk_size=200, overlap=40)
    print(f'Fixed-size chunking (size=200, overlap=40): {len(fixed_chunks)} chunks')
    for i, c in enumerate(fixed_chunks[:3]):
        print(f'\n  [Chunk {i}] chars {c["start"]}–{c["end"]}:')
        print(f'  {c["text"][:100]}...')
    """),

    md(r"""
    ### 5b Sentence-boundary chunking
    """),

    code(r"""
    # 5b. Sentence-boundary chunking: split at sentence boundaries, accumulate
    # sentences until chunk_size is exceeded.
    def sentence_chunks(text, max_chars=300, overlap_sentences=1):
        # Split into sentences using punctuation.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_sents = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > max_chars and current_sents:
                chunks.append(' '.join(current_sents))
                # Keep overlap sentences.
                current_sents = current_sents[-overlap_sentences:]
                current_len = sum(len(s) for s in current_sents)
            current_sents.append(sent)
            current_len += len(sent) + 1

        if current_sents:
            chunks.append(' '.join(current_sents))

        return [{'text': c, 'n_chars': len(c)} for c in chunks]

    sent_chunks = sentence_chunks(DOCUMENT, max_chars=300, overlap_sentences=1)
    print(f'Sentence chunking (max_chars=300, overlap=1 sent): {len(sent_chunks)} chunks')
    for i, c in enumerate(sent_chunks[:3]):
        print(f'\n  [Chunk {i}] {c["n_chars"]} chars:')
        print(f'  {c["text"][:120]}...')
    """),

    md(r"""
    ### 5c Recursive character splitting
    """),

    code(r"""
    # 5c. Recursive character splitter: try separators in priority order.
    def recursive_split(text, chunk_size=250, chunk_overlap=30,
                        separators=None):
        if separators is None:
            separators = ['\n\n', '\n', '. ', '! ', '? ', ' ', '']

        def _split(text, seps):
            if len(text) <= chunk_size:
                return [text] if text.strip() else []
            sep = seps[0]
            rest = seps[1:]
            if sep == '':
                # Character-level fallback.
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
            parts = text.split(sep)
            chunks = []
            current = ''
            for part in parts:
                candidate = (current + sep + part).lstrip(sep) if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # Part itself may be too large — recurse with next separator.
                    if len(part) > chunk_size and rest:
                        chunks.extend(_split(part, rest))
                        current = ''
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

        raw_chunks = _split(text, separators)
        return [{'text': c.strip(), 'n_chars': len(c.strip())} for c in raw_chunks if c.strip()]

    rec_chunks = recursive_split(DOCUMENT, chunk_size=250, chunk_overlap=30)
    print(f'Recursive splitting (size=250, overlap=30): {len(rec_chunks)} chunks')
    for i, c in enumerate(rec_chunks[:3]):
        print(f'\n  [Chunk {i}] {c["n_chars"]} chars:')
        print(f'  {c["text"][:120]}...')
    """),

    md(r"""
    ### 5d Semantic chunking
    """),

    code(r"""
    # 5d. Semantic chunking: split where consecutive sentence embeddings diverge.
    def simple_embed(text, dim=32):
        # Deterministic word-hash embedding (stand-in for sentence-transformers).
        words = text.lower().split()
        vec = np.zeros(dim)
        for w in words:
            h = hash(w) % (10**9 + 7)
            vec[h % dim] += 1.0
            vec[(h * 31) % dim] += 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    def semantic_chunks(text, threshold_percentile=75, min_chars=100):
        # Split into sentences first.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if len(sentences) < 2:
            return [{'text': text, 'n_chars': len(text)}]

        # Embed sentences.
        embs = np.stack([simple_embed(s) for s in sentences])

        # Compute cosine distance between consecutive sentences.
        breakpoint_scores = []
        for i in range(len(sentences) - 1):
            cos_sim = float(embs[i] @ embs[i + 1])
            breakpoint_scores.append(1.0 - cos_sim)   # distance = 1 - similarity

        # Threshold: split where distance exceeds the given percentile.
        threshold = float(np.percentile(breakpoint_scores, threshold_percentile))

        # Build chunks.
        chunks = []
        current_sents = [sentences[0]]
        for i, score in enumerate(breakpoint_scores):
            if score > threshold:
                text_chunk = ' '.join(current_sents)
                if len(text_chunk) >= min_chars:
                    chunks.append({'text': text_chunk, 'n_chars': len(text_chunk),
                                   'breakpoint_score': score})
                    current_sents = [sentences[i + 1]]
                else:
                    current_sents.append(sentences[i + 1])
            else:
                current_sents.append(sentences[i + 1])
        if current_sents:
            chunks.append({'text': ' '.join(current_sents),
                           'n_chars': len(' '.join(current_sents)),
                           'breakpoint_score': 0.0})
        return chunks, breakpoint_scores, threshold

    sem_chunks, bp_scores, bp_threshold = semantic_chunks(DOCUMENT, threshold_percentile=75)
    print(f'Semantic chunking (p75 threshold={bp_threshold:.3f}): {len(sem_chunks)} chunks')
    for i, c in enumerate(sem_chunks[:3]):
        print(f'\n  [Chunk {i}] {c["n_chars"]} chars:')
        print(f'  {c["text"][:120]}...')
    """),

    md(r"""
    ### 5e Parent-child chunking
    """),

    code(r"""
    # 5e. Parent-child chunking: large parent chunks for context, small child chunks for retrieval.
    def parent_child_chunks(text, parent_size=500, child_size=100, child_overlap=20):
        parents = fixed_size_chunks(text, chunk_size=parent_size, overlap=50)
        result = []
        for pid, parent in enumerate(parents):
            children = fixed_size_chunks(parent['text'], chunk_size=child_size, overlap=child_overlap)
            for cid, child in enumerate(children):
                result.append({
                    'parent_id':   pid,
                    'child_id':    cid,
                    'child_text':  child['text'],
                    'parent_text': parent['text'],
                    'child_chars': len(child['text']),
                    'parent_chars': len(parent['text']),
                })
        return result

    pc_chunks = parent_child_chunks(DOCUMENT, parent_size=500, child_size=100, child_overlap=20)
    n_parents = len(set(c['parent_id'] for c in pc_chunks))
    print(f'Parent-child: {n_parents} parents, {len(pc_chunks)} child chunks')
    print(f'Avg children per parent: {len(pc_chunks)/n_parents:.1f}')
    print('\nExample: child retrieved, parent returned to LLM:')
    ex = pc_chunks[3]
    print(f'  Child  ({ex["child_chars"]} chars): {ex["child_text"][:80]}...')
    print(f'  Parent ({ex["parent_chars"]} chars): {ex["parent_text"][:120]}...')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Chunk size distributions for all four strategies.
    strategies = {
        'Fixed-size': [len(c['text']) for c in fixed_chunks],
        'Sentence':   [len(c['text']) for c in sent_chunks],
        'Recursive':  [len(c['text']) for c in rec_chunks],
        'Semantic':   [len(c['text']) for c in sem_chunks],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple']
    for ax, (name, sizes), color in zip(axes.flat, strategies.items(), colors):
        ax.hist(sizes, bins=min(20, len(sizes)), color=color, alpha=0.8, edgecolor='white')
        ax.axvline(np.mean(sizes), color='red', ls='--', label=f'mean={np.mean(sizes):.0f}')
        ax.set_title(f'{name} ({len(sizes)} chunks)')
        ax.set_xlabel('Chunk size (chars)'); ax.set_ylabel('Count')
        ax.legend(fontsize=9)
    plt.suptitle('Figure 1 — Chunk size distributions by strategy')
    plt.tight_layout()
    plt.show()

    print('Strategy summary:')
    for name, sizes in strategies.items():
        print(f'  {name:12s}: n={len(sizes):3d}  '
              f'mean={np.mean(sizes):.0f}  '
              f'std={np.std(sizes):.0f}  '
              f'min={np.min(sizes):.0f}  max={np.max(sizes):.0f}')
    """),

    md(r"""
    **Figure 1.** Chunk size distributions reveal each strategy's character.
    **Fixed-size** (top-left): tight distribution — all chunks are exactly
    chunk\_size except the last. **Sentence** (top-right): more variance — some
    sentences are long, some short, but boundaries are at natural language breaks.
    **Recursive** (bottom-left): attempts to hit a size target while respecting
    structure — distribution is narrower than sentence chunking. **Semantic**
    (bottom-right): high variance — topic-coherent sections vary greatly in length.
    The right strategy depends on your corpus: uniform prose → fixed; prose with
    varying density → recursive; heterogeneous topics → semantic; need both retrieval
    precision and generation context → parent-child.
    """),

    code(r"""
    # Figure 2 — Semantic breakpoint scores along the document.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(bp_scores)), bp_scores, 'o-', ms=5, color='steelblue', alpha=0.7)
    ax.axhline(bp_threshold, color='red', ls='--', label=f'Split threshold (p75={bp_threshold:.3f})')
    split_positions = [i for i, s in enumerate(bp_scores) if s > bp_threshold]
    ax.scatter(split_positions, [bp_scores[i] for i in split_positions],
               s=100, color='red', zorder=5, label=f'Split points ({len(split_positions)})')
    ax.set_xlabel('Sentence pair index')
    ax.set_ylabel('Cosine distance between consecutive sentences')
    ax.set_title('Figure 2 — Semantic chunking: breakpoint scores along the document')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** Cosine distance between consecutive sentence embeddings (higher =
    more different topics). Red dashed line: the 75th-percentile threshold — splits
    are inserted where the score exceeds it. Peaks correspond to paragraph transitions
    in the source document (Python overview → history → standard library → comprehensions
    → exception handling). The threshold controls granularity: lower threshold → more
    splits (finer chunks); higher threshold → fewer splits (coarser chunks). In
    practice, tuning the threshold percentile (p70–p90) on a sample of your corpus
    and measuring retrieval Recall@k is the best approach.
    """),

    code(r"""
    # Figure 3 — Retrieval precision vs. chunk size (simulated).
    # Simulate: a query matches a 150-char passage. Measure retrieval score as
    # a function of chunk size (smaller = more focused; larger = diluted).
    target_passage = 'List comprehensions provide a concise way to create lists'
    target_pos = DOCUMENT.find(target_passage)

    def retrieval_score_for_chunk_size(chunk_size, overlap=30):
        chunks = fixed_size_chunks(DOCUMENT, chunk_size=chunk_size, overlap=overlap)
        best_score = 0.0
        for c in chunks:
            passage_overlap = min(c['end'], target_pos + len(target_passage)) - max(c['start'], target_pos)
            if passage_overlap > 0:
                # Fraction of chunk that is the target passage.
                precision = passage_overlap / len(c['text'])
                # Fraction of target passage covered.
                recall_chunk = passage_overlap / len(target_passage)
                # Score = harmonic mean.
                f1 = 2 * precision * recall_chunk / (precision + recall_chunk + 1e-9)
                best_score = max(best_score, f1)
        return best_score

    chunk_sizes = [50, 80, 100, 150, 200, 300, 400, 500]
    scores = [retrieval_score_for_chunk_size(cs) for cs in chunk_sizes]

    fig, ax = plt.subplots()
    ax.plot(chunk_sizes, scores, 'D-', ms=8, color='seagreen')
    ax.axvline(len(target_passage), color='red', ls=':', label=f'Passage length={len(target_passage)} chars')
    ax.set_xlabel('Chunk size (chars)')
    ax.set_ylabel('Retrieval F1 (precision × recall of target passage)')
    ax.set_title('Figure 3 — Retrieval precision vs. chunk size')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** Retrieval precision (measured as F1 between the chunk and the
    target passage) peaks when chunk size ≈ passage length (red dotted line). Below
    this: the passage is split across chunks — each chunk captures only part of the
    relevant content (low recall per chunk). Above this: the target passage is diluted
    by surrounding irrelevant content, lowering the cosine similarity to the query.
    The **key design principle**: choose chunk size to match the expected length of
    relevant answers in your corpus. For FAQ-style knowledge bases (short answers):
    100–200 chars. For technical documentation (paragraph-length explanations):
    300–500 chars. For scientific papers (section-length): 500–1000 chars.
    """),

    code(r"""
    # Figure 4 — Parent-child: retrieval at child granularity, generation at parent.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.barh(['Parent\n(generation)', 'Child\n(retrieval)'],
            [np.mean([c['parent_chars'] for c in pc_chunks]),
             np.mean([c['child_chars']  for c in pc_chunks])],
            color=['steelblue', 'darkorange'], alpha=0.8)
    ax.set_xlabel('Average chunk size (chars)')
    ax.set_title(f'Figure 4 — Parent-child chunking: '
                 f'{n_parents} parents, {len(pc_chunks)} children')
    for i, (label, val) in enumerate(zip(
        ['Parent', 'Child'],
        [np.mean([c['parent_chars'] for c in pc_chunks]),
         np.mean([c['child_chars']  for c in pc_chunks])]
    )):
        ax.text(val + 5, i, f'{val:.0f} chars', va='center', fontsize=10)
    plt.tight_layout()
    plt.show()

    print('Parent-child chunk stats:')
    print(f'  Parent size: {np.mean([c["parent_chars"] for c in pc_chunks]):.0f} chars avg')
    print(f'  Child size:  {np.mean([c["child_chars"]  for c in pc_chunks]):.0f} chars avg')
    print(f'  Children per parent: {len(pc_chunks)/n_parents:.1f} avg')
    """),

    md(r"""
    **Figure 4.** Parent-child chunking maintains two granularities simultaneously.
    Child chunks (orange) are indexed for vector search — their small size means
    the embedding captures a specific, focused concept with minimal noise. Parent
    chunks (blue) are returned to the LLM — they provide enough surrounding context
    for coherent answer generation. In production, the mapping `child_id → parent_id`
    is stored in the vector database as metadata. When a child chunk is retrieved,
    a second lookup fetches the parent document. This adds ~1ms to retrieval latency
    but dramatically improves generation quality.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Mid-sentence splits** | Truncated context confuses LLM | Fixed-size splits ignore sentence boundaries | Use sentence or recursive chunking |
    | **Chunk too large for encoder** | Embedding truncated at 512 tokens | chunk_size > model max_seq_len | Set chunk_size ≤ (max_seq_len × ~4 chars) |
    | **No overlap** | Query at chunk boundary retrieves neither chunk | Fixed-size with overlap=0 | Set overlap 10–20% of chunk_size |
    | **Code split mid-function** | Code context lost | Sentence splitter sees '.' inside code | Use language-aware splitter or preserve code blocks |
    | **Semantic threshold too low** | Hundreds of 1-sentence chunks | High sensitivity splits every sentence | Increase threshold percentile (try p85, p90) |
    | **Parent too large for LLM** | Context window overflow | Parent chunk > LLM context / k | Cap parent size; reduce k_retrieve |
    | **Homogeneous document** | Semantic chunking = 1 chunk | No topic transitions in document | Fall back to recursive chunking |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 LangChain text splitters (guarded).
    try:
        from langchain.text_splitter import (  # noqa: F401
            RecursiveCharacterTextSplitter,
            SentenceTransformersTokenTextSplitter,
        )
        lines = [
            'from langchain.text_splitter import RecursiveCharacterTextSplitter',
            '',
            '# Recursive splitter — production default.',
            'splitter = RecursiveCharacterTextSplitter(',
            '    chunk_size=500,',
            '    chunk_overlap=50,',
            '    separators=["\\n\\n", "\\n", ". ", " ", ""]',
            ')',
            'chunks = splitter.split_text(document)',
            '',
            '# Token-aware splitter (respects embedding model limits).',
            'from langchain.text_splitter import SentenceTransformersTokenTextSplitter',
            'token_splitter = SentenceTransformersTokenTextSplitter(',
            '    model_name="sentence-transformers/all-MiniLM-L6-v2",',
            '    chunk_overlap=25',
            ')',
            'token_chunks = token_splitter.split_text(document)',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[langchain not installed — production pattern]:',
            '  from langchain.text_splitter import RecursiveCharacterTextSplitter',
            '  splitter = RecursiveCharacterTextSplitter(',
            '      chunk_size=500, chunk_overlap=50,',
            '      separators=["\\n\\n", "\\n", ". ", " ", ""])',
            '  chunks = splitter.split_text(document)',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Technical Documentation Search

    **Scenario.** A software company's support team uses RAG to answer developer
    questions over 10,000 pages of API documentation, tutorials, and code examples.

    **Challenge.** The documentation mixes prose (explanations) with code blocks.
    Fixed-size chunking splits mid-function. Sentence splitting fails because code
    contains many `.` characters that are not sentence boundaries.

    **Solution: Content-aware recursive chunking + parent-child.**
    - Pre-processing: identify code blocks (` ``` ` delimiters) and keep them intact
      as atomic units. Replace with placeholders before chunking.
    - Chunking: recursive splitter with separators: `\n\n` (paragraph) → `\n` (line)
      → `. ` (sentence, but only outside code) → ` ` (word).
    - Parent size: 600 chars (one full code example + explanation).
    - Child size: 150 chars (function signature or key sentence).
    - Re-insert code blocks after chunking.

    **Results:** Precision@5 improved from 0.58 (fixed-size) to 0.79 (recursive +
    parent-child). The most impactful change was keeping code blocks intact — previously
    50% of code-containing queries retrieved partial function signatures.

    **KPIs:** Precision@5 on 100 golden queries, answer correctness rated by senior
    engineers (4-point scale), p99 retrieval latency.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Chunk size ≤ embedding model max_seq_len.** Most sentence-transformers cap
      at 256 or 512 tokens (~1000–2000 chars). Exceeding this causes silent truncation
      with no error — the embedding represents only the first N tokens.
    - **Token-based splitting.** Char-based splitting is an approximation. For tight
      control, use a token-aware splitter (HuggingFace tokenizer) to count tokens
      exactly per chunk.
    - **Chunk size and LLM context.** If the LLM has a 4K token context window and
      you retrieve k=5 chunks: max\_chunk\_size ≤ 4000/5 = 800 tokens ≈ 3200 chars,
      leaving room for the system prompt and query.
    - **Re-chunking on model change.** If you update the embedding model, chunks
      indexed with the old model may produce poor embeddings with the new one. Always
      re-chunk and re-embed when changing models.
    - **Metadata preservation.** Each chunk should carry: source document ID, page
      number, section title, creation date. These enable metadata filtering (Notebook 26)
      and source attribution in the LLM response.
    - **Overlap strategy.** Overlap ensures that queries hitting a chunk boundary
      retrieve relevant context. Too much overlap (>30%) wastes storage and creates
      near-duplicate embeddings. Typical: 10–15% of chunk size.
    - **Deduplication.** After parent-child retrieval, de-duplicate parent chunks
      before sending to the LLM (multiple children may share the same parent).
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Chunking strategy comparison:**

    | Strategy | Respects structure | Size control | Needs embeddings | Speed | Best for |
    |---|---|---|---|---|---|
    | Fixed-size | No | Exact | No | Very fast | Uniform prose, quick prototypes |
    | Sentence | Partial | Approximate | No | Fast | General prose, FAQs |
    | Recursive | Yes (hierarchy) | Good | No | Fast | Most production use cases |
    | Semantic | Yes (topics) | Varies | Yes (online) | Slow | Heterogeneous long documents |
    | Parent-child | Yes | Both levels | No (chunking) | Fast | **Production default** |

    **Chunk size tradeoffs:**

    | Chunk size | Retrieval precision | Generation context | Storage | Recommended for |
    |---|---|---|---|---|
    | Tiny (50–100 chars) | Very high | Very low | Very high | Fine-grained Q&A |
    | Small (100–300 chars) | High | Low | High | FAQ, short answers |
    | Medium (300–600 chars) | Good | Good | Medium | **General production default** |
    | Large (600–1200 chars) | Moderate | High | Low | Long-context generation |
    | Page-level (1000+) | Low | Excellent | Very low | Summarisation tasks |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What chunk size should you use?"* → It depends on the corpus and use case.
      Start with 300–500 chars (recursive splitter). Measure Recall@5 on golden queries
      at chunk sizes {100, 200, 300, 500, 800}. Choose the size that maximises Recall@5.
      Use parent-child if both fine-grained retrieval and full context are needed.
    - *"What is semantic chunking and when is it better than recursive splitting?"*
      → Semantic chunking splits on cosine distance drops between consecutive sentence
      embeddings. Better than recursive when: (1) document has variable-length topics;
      (2) you need each chunk to cover exactly one topic; (3) no consistent paragraph
      structure. Slower (requires embeddings) and not necessary for well-structured prose.

    **Deep-dive questions**
    - *"How does overlap prevent boundary misses?"* → If chunk_size=500 and overlap=50,
      a passage starting at position 480 (near a boundary) is included in both chunk 1
      (positions 0–500) and chunk 2 (positions 450–950). At least one of the two chunks
      contains the full passage. The fraction of the document covered twice = overlap /
      chunk_size. Balance: more overlap = better recall, more storage, more duplicate
      embeddings.
    - *"What breaks when you change the embedding model?"* → Old embeddings are in the
      old model's vector space. New query embeddings are in the new space. Cosine
      similarity between old and new is undefined (the spaces may not align). You must
      re-embed all chunks. This is why embedding model updates require full pipeline
      re-indexing (Notebook 26).

    **Whiteboard question**
    - "A legal document search system has contracts ranging from 1 page to 500 pages.
      Queries range from specific clause lookups ('indemnification clause') to
      high-level summaries ('what are the payment terms?'). Design the chunking strategy."

    **Common mistakes:** setting chunk_size larger than the embedding model's max_seq_len
    (silent truncation); not measuring retrieval quality at different chunk sizes; using
    fixed-size chunking for code or structured documents; forgetting to store source
    metadata with each chunk.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Why chunk at all?** Name two reasons. Which is a hard constraint?
    2. **Overlap.** What problem does overlap solve? What is the cost?
    3. **Recursive splitter.** Explain the separator priority order. What happens
       when a chunk is still too large after applying the first separator?
    4. **Semantic chunking.** What signal triggers a split? How do you tune sensitivity?
    5. **Parent-child.** Which chunk is indexed for retrieval? Which is sent to the LLM?
       What extra data must be stored?
    6. **Chunk size optimum.** If the relevant passages in your corpus are typically
       150 chars long, what chunk size should you target and why?
    7. **Code blocks.** Why does sentence chunking fail on technical documentation
       containing code? How do you fix it?
    8. **Model change.** Your team upgrades the embedding model. What must happen to
       all existing chunks in the vector database?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. A document is 10,000 characters. chunk_size=500, overlap=50. How many chunks
       are produced? Show the formula.
    2. Why is character-based chunk size an approximation for token counts? Give an
       example where 500 chars ≠ 125 tokens.

    **Beginner → Intermediate (coding)**
    3. Implement a **markdown-aware splitter**: split at `## ` headers first, then
       within each section use recursive splitting. Test on a markdown document with
       5 sections of varying length.
    4. Implement **code-block preservation**: before chunking, extract all ` ``` `
       code blocks, replace with `[CODE_BLOCK_0]` etc., chunk the remaining prose,
       then re-insert the code blocks into the chunks that contain the placeholders.

    **Intermediate (analysis)**
    5. Grid-search chunk_size × overlap (try sizes [100, 200, 300, 500] × overlaps
       [0, 10%, 20%]). For a query "list comprehensions in Python", define a ground-
       truth relevant passage, and measure the fraction of configurations that retrieve
       it at rank 1.
    6. Implement **token-based chunking**: use HuggingFace
       `AutoTokenizer.from_pretrained("bert-base-uncased")` to count tokens per chunk
       and split at exactly 128 tokens. Compare chunk count and size distribution to
       char-based chunking with chunk_size=512.

    **Senior (design)**
    7. *System design:* design the chunking pipeline for a medical knowledge base
       (PubMed, 35M abstracts + 2M full-text PDFs). Abstracts are 200–300 words;
       full-text PDFs are 5,000–20,000 words with figures, tables, and references.
       Specify: chunking strategy per content type, chunk size, overlap, metadata fields,
       and how you handle tables and figures.
    8. *Tradeoff:* your RAG system has latency = 150ms (retrieval 20ms, generation 130ms).
       The team proposes adding semantic chunking to improve Recall@5 from 0.82 to 0.89.
       Semantic chunking requires re-embedding 50M chunks (1 hour on 8 GPUs) and adds
       3ms to ingestion latency per document. Evaluate the tradeoff: is it worth it?
       What additional information would you need?
    """),

    md(r"""
    ---
    ### Summary
    Chunking is the foundational preprocessing step of every RAG pipeline. **Fixed-size**
    is simple but ignores language structure. **Sentence** and **recursive** splitting
    respect natural boundaries and are the production default. **Semantic chunking**
    detects topic shifts using embeddings — best for long heterogeneous documents.
    **Parent-child** chunking provides both fine-grained retrieval and full-context
    generation. The right chunk size matches the expected length of relevant passages
    in your corpus; always measure Recall@k across chunk sizes before deploying.

    **Next:** `30 · Advanced RAG Architectures` — query transformation techniques
    (HyDE, multi-query, step-back) that improve retrieval before the first stage,
    and hierarchical indexing (RAPTOR) that improves recall at multiple granularities.
    """),
]

build("phase5_rag/29_chunking_strategies.ipynb", cells)
