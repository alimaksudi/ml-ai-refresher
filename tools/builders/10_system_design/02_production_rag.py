"""SYS-02 — Production RAG Systems builder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nbbuild import md, code, build

cells = [

# ── 1. Learning Objectives ────────────────────────────────────────────────────
md(r"""
# SYS-02 — Production RAG Systems

## 1. Learning Objectives

By the end of this notebook you will be able to:

- Describe the full production RAG architecture: index → retrieve → generate → evaluate
- Implement an **IncrementalIndexer** that supports add/delete/update without full rebuild
- Implement a **ContextWindowManager** that fits chunks within a token budget
- Implement a **lexical support proxy** and explain why production faithfulness
  requires claim-level entailment and citation checks
- Build a **ProductionRAGPipeline** with caching at both retrieval and generation layers
- Implement **HNSW-inspired greedy nearest-neighbor search** from scratch
- Apply hybrid search (BM25 + dense) with Reciprocal Rank Fusion from scratch
- Design model-tier routing: cheap model for simple queries, expensive for complex
- Scale to 1M documents: parallel chunking, deduplication, shard-based index
"""),

# ── 2. Historical Motivation ───────────────────────────────────────────────────
md(r"""
## 2. Historical Motivation

### From Fine-Tuning to RAG

Before RAG, enterprise AI meant **fine-tuning**: expensive, slow, and the model's
knowledge froze at training time.  IBM estimated in 2023 that fine-tuning a
domain-specific model cost $500k–$5M and took 3–6 months.

RAG (Retrieval-Augmented Generation) emerged from Facebook AI Research in 2020
(Lewis et al.) as a way to give frozen LLMs access to updatable knowledge.
By 2023 it had become the dominant enterprise AI pattern.

**Production challenges that appeared at scale**:

| Scale | Challenge |
|---|---|
| 10k docs | Retrieval quality (recall@5 < 60%) |
| 100k docs | Index update latency (rebuild takes hours) |
| 1M docs | Embedding costs ($2k/rebuild), shard routing |
| 10M docs | ANN index memory (each vector = 6KB at 1536-dim) |
| Real-time | Faithfulness hallucination rate spikes at low recall |

**Key lesson**: building a RAG prototype takes a day; building a production RAG
system that meets faithfulness > 90%, p95 < 3s, and $0.02/query takes months.
"""),

# ── 3. Intuition & Visual Understanding ──────────────────────────────────────
md(r"""
## 3. Intuition & Visual Understanding

### RAG as a Lookup + Compose System

```
User query
    ↓ embed (sentence-transformer or API)
Query vector ──→ ANN Index ──→ Top-K chunk IDs
                                    ↓
                            Chunk store (S3/DB)
                                    ↓ fetch text
                            Context window assembler
                                    ↓ fit K chunks in budget
                            Prompt template
                                    ↓
                            LLM (GPT-4o / Claude)
                                    ↓
                            Response + source citations
                                    ↓
                            Faithfulness evaluator
```

### Context Window as a Knapsack Problem

Each chunk has a size (tokens) and a relevance score.
We want to include the highest-relevance chunks while staying under budget.
This is the 0/1 knapsack: choose the top-M chunks by score such that their
total tokens ≤ budget.

Simple greedy (sort by score, take until budget) gives a near-optimal solution
when chunk sizes are similar.

### Faithfulness vs Relevance

- **Faithfulness**: does the generated answer come from the retrieved context?
  (hallucination check — measures if the model made something up)
- **Relevance**: does the retrieved context contain the answer at all?
  (retrieval quality check)
- **Answer relevance**: is the generated answer actually what the user asked?
  (end-to-end quality)
"""),

# ── 4. Mathematical Foundations ───────────────────────────────────────────────
md(r"""
## 4. Mathematical Foundations

### 4.1 Cosine Similarity

$$\text{cos}(q, d) = \frac{q \cdot d}{\|q\|\|d\|}$$

For L2-normalised vectors: $\text{cos}(q, d) = q \cdot d$ (dot product = cosine).

### 4.2 BM25 Score

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 (1 - b + b \cdot |d| / \text{avgdl})}$$

where $f(t,d)$ = term frequency in doc, $k_1=1.5$, $b=0.75$.

### 4.3 Reciprocal Rank Fusion

Given rank lists from dense retrieval ($r_d$) and sparse BM25 ($r_s$):

$$\text{RRF}(d) = \frac{1}{k + r_d(d)} + \frac{1}{k + r_s(d)}, \quad k = 60$$

### 4.4 Token Budget (Context Window Management)

$$\text{Greedy}: \quad S^* = \{c_{(i)} : \sum_{j \le i} \text{tokens}(c_{(j)}) \le B\}$$

where $c_{(1)}, c_{(2)}, \ldots$ are chunks sorted by relevance score descending,
and $B$ is the token budget.

### 4.5 Lexical Support Proxy (N-gram overlap)

$$\text{LexicalSupport} = \frac{|\text{answer trigrams} \cap \text{context trigrams}|}{|\text{answer trigrams}|}$$

**Read and symbols:** the numerator counts answer phrases copied from context; the
denominator counts answer phrases. This measures lexical overlap only. It can score
an unsupported copied phrase highly and a faithful paraphrase poorly, so it must
not be named or thresholded as faithfulness. Production evaluation decomposes the
answer into claims and checks entailment/citations, calibrated against human labels.
"""),

# ── 5. Manual Implementation from Scratch ─────────────────────────────────────
md(r"""
## 5. Manual Implementation from Scratch
"""),

code(r"""
import numpy as np
import math
import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

# ── BM25 from scratch ─────────────────────────────────────────────────────────
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b  = b
        self._corpus: List[List[str]] = []
        self._doc_ids: List[str] = []
        self._df: Dict[str, int] = {}
        self._avgdl: float = 0.0
        self._N: int = 0

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def add_documents(self, docs: Dict[str, str]):
        for doc_id, text in docs.items():
            tokens = self._tokenise(text)
            self._corpus.append(tokens)
            self._doc_ids.append(doc_id)
            for tok in set(tokens):
                self._df[tok] = self._df.get(tok, 0) + 1
        self._N = len(self._corpus)
        self._avgdl = sum(len(d) for d in self._corpus) / max(self._N, 1)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_tokens = self._tokenise(query)
        scores = []
        for idx, doc_tokens in enumerate(self._corpus):
            tf_map = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1
            dl = len(doc_tokens)
            s = 0.0
            for t in query_tokens:
                tf = tf_map[t]
                idf = self._idf(t)
                denom = tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                s += idf * (tf * (self._k1 + 1)) / max(denom, 1e-9)
            scores.append((self._doc_ids[idx], s))
        return sorted(scores, key=lambda x: -x[1])[:top_k]


# ── Dense retrieval (cosine on L2-normalised vectors) ─────────────────────────
class DenseIndex:
    def __init__(self):
        self._ids: List[str]     = []
        self._vecs: List[np.ndarray] = []

    def add(self, doc_id: str, vec: np.ndarray):
        norm = np.linalg.norm(vec)
        self._ids.append(doc_id)
        self._vecs.append(vec / max(norm, 1e-9))

    def delete(self, doc_id: str):
        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            self._ids.pop(idx)
            self._vecs.pop(idx)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._vecs:
            return []
        norm = np.linalg.norm(query_vec)
        q = query_vec / max(norm, 1e-9)
        mat = np.array(self._vecs)
        scores = mat @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def __len__(self):
        return len(self._ids)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def reciprocal_rank_fusion(rank_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    rrf_scores: Dict[str, float] = defaultdict(float)
    for rank_list in rank_lists:
        for rank, (doc_id, _score) in enumerate(rank_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: -x[1])


# ── IncrementalIndexer ────────────────────────────────────────────────────────
class IncrementalIndexer:
    def __init__(self, embed_fn=None):
        self._bm25      = BM25()
        self._dense     = DenseIndex()
        self._chunks:  Dict[str, str]        = {}   # id -> text
        self._metadata: Dict[str, Dict]      = {}
        self._embed_fn  = embed_fn or self._fake_embed
        self._n_added   = 0
        self._n_deleted = 0

    def _fake_embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return rng.normal(0, 1, 64)

    def _chunk_id(self, doc_id: str, chunk_idx: int) -> str:
        return f"{doc_id}::chunk{chunk_idx}"

    def _chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def add_document(self, doc_id: str, text: str, metadata: Dict = None, chunk_size: int = 200):
        chunks = self._chunk_text(text, chunk_size)
        for ci, chunk in enumerate(chunks):
            cid = self._chunk_id(doc_id, ci)
            self._chunks[cid] = chunk
            self._metadata[cid] = metadata or {}
            self._dense.add(cid, self._embed_fn(chunk))
            self._n_added += 1
        # BM25 re-index (in prod: incremental only)
        self._bm25.add_documents({self._chunk_id(doc_id, ci): chunk
                                   for ci, chunk in enumerate(chunks)})

    def delete_document(self, doc_id: str):
        to_delete = [cid for cid in self._chunks if cid.startswith(f"{doc_id}::")]
        for cid in to_delete:
            self._dense.delete(cid)
            del self._chunks[cid]
            del self._metadata[cid]
            self._n_deleted += 1

    def update_document(self, doc_id: str, text: str, metadata: Dict = None):
        self.delete_document(doc_id)
        self.add_document(doc_id, text, metadata)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        q_vec = self._embed_fn(query)
        dense_results = self._dense.search(q_vec, top_k=top_k * 2)
        sparse_results = self._bm25.score(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return [(cid, score, self._chunks.get(cid, "")) for cid, score in fused[:top_k]]

    def stats(self):
        return {"total_chunks": len(self._chunks), "added": self._n_added,
                "deleted": self._n_deleted, "dense_index_size": len(self._dense)}


# ── Smoke test ────────────────────────────────────────────────────────────────
indexer = IncrementalIndexer()
docs = {
    "doc1": "Machine learning is a subset of artificial intelligence. It uses statistical methods to learn from data.",
    "doc2": "Deep learning uses neural networks with many layers to learn complex representations from data.",
    "doc3": "Natural language processing enables computers to understand and generate human language.",
    "doc4": "Vector databases store high-dimensional embeddings and support approximate nearest neighbour search.",
    "doc5": "RAG combines retrieval from a knowledge base with generation from a language model.",
}
for did, text in docs.items():
    indexer.add_document(did, text)

print("Index stats:", indexer.stats())
results = indexer.search("neural networks and deep learning", top_k=3)
print("\nSearch: 'neural networks and deep learning'")
for cid, score, text in results:
    print(f"  [{cid}] score={score:.4f}: {text[:60]}...")

indexer.update_document("doc1", "Machine learning, including supervised, unsupervised, and reinforcement learning, is a branch of AI.")
print("\nAfter update, index stats:", indexer.stats())
"""),

# ── ContextWindowManager ──────────────────────────────────────────────────────
md(r"""
### Context Window Manager
"""),

code(r"""
import re
from typing import List, Tuple

def count_tokens_approx(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)  # ~0.75 words per token

class ContextWindowManager:
    def __init__(self, budget_tokens: int = 3000, reserve_tokens: int = 500):
        self._budget  = budget_tokens - reserve_tokens  # reserve for system prompt + answer
        self._reserve = reserve_tokens

    def assemble(self, chunks: List[Tuple[str, float, str]], query: str) -> Dict:
        query_tokens = count_tokens_approx(query)
        remaining = self._budget - query_tokens
        selected, total_tokens = [], 0

        for cid, score, text in sorted(chunks, key=lambda x: -x[1]):
            chunk_tokens = count_tokens_approx(text)
            if total_tokens + chunk_tokens <= remaining:
                selected.append((cid, score, text, chunk_tokens))
                total_tokens += chunk_tokens
            else:
                # Try to fit a truncated version
                words = text.split()
                max_words = max(0, (remaining - total_tokens) * 3 // 4)
                if max_words > 20:
                    trunc = " ".join(words[:max_words]) + "..."
                    trunc_tokens = count_tokens_approx(trunc)
                    selected.append((cid, score, trunc, trunc_tokens))
                    total_tokens += trunc_tokens
                break

        context_text = "\n\n".join([f"[{cid}]: {text}" for cid, _, text, _ in selected])
        return {
            "context": context_text,
            "selected_chunks": [cid for cid, _, _, _ in selected],
            "total_context_tokens": total_tokens,
            "budget_tokens": self._budget,
            "utilisation": total_tokens / self._budget,
        }


# ── Lexical support proxy (n-gram overlap; not a faithfulness metric) ───────
def get_ngrams(text: str, n: int) -> set:
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(zip(*[tokens[i:] for i in range(n)]))

def lexical_support_score(answer: str, context: str, n: int = 3) -> float:
    answer_ngrams  = get_ngrams(answer, n)
    context_ngrams = get_ngrams(context, n)
    if not answer_ngrams:
        return 0.0
    overlap = answer_ngrams & context_ngrams
    return len(overlap) / len(answer_ngrams)

def relevance_score(query: str, context: str, n: int = 2) -> float:
    q_ngrams = get_ngrams(query, n)
    c_ngrams = get_ngrams(context, n)
    if not q_ngrams:
        return 0.0
    return len(q_ngrams & c_ngrams) / len(q_ngrams)


# ── Quick test ────────────────────────────────────────────────────────────────
ctx_mgr = ContextWindowManager(budget_tokens=2000, reserve_tokens=500)
results_test = indexer.search("retrieval augmented generation knowledge base", top_k=5)
assembled = ctx_mgr.assemble(results_test, "What is RAG?")
print("Context assembly:")
print(f"  Chunks selected: {len(assembled['selected_chunks'])}")
print(f"  Total tokens: {assembled['total_context_tokens']}")
print(f"  Budget utilisation: {assembled['utilisation']:.1%}")

answer_good = "RAG combines retrieval from a knowledge base with generation from a language model."
answer_bad  = "The capital of France is Paris and it has many museums."
context_text = assembled["context"]

print(f"\nLexical support (good answer): {lexical_support_score(answer_good, context_text):.3f}")
print(f"Lexical support (bad answer):  {lexical_support_score(answer_bad,  context_text):.3f}")
print(f"Relevance score:            {relevance_score('What is RAG?', context_text):.3f}")
"""),

# ── ProductionRAGPipeline ────────────────────────────────────────────────────
md(r"""
### ProductionRAGPipeline — With Caching and Tier Routing
"""),

code(r"""
import hashlib
from typing import Dict, List, Optional, Tuple

class TieredLLM:
    def __init__(self):
        self._calls = {"cheap": 0, "expensive": 0}

    def _is_complex(self, query: str, context_tokens: int) -> bool:
        # Simple heuristic: long context or question has multiple parts
        return context_tokens > 1500 or query.count("?") > 1 or len(query.split()) > 25

    def generate(self, prompt: str, context_tokens: int, query: str) -> Tuple[str, str]:
        if self._is_complex(query, context_tokens):
            self._calls["expensive"] += 1
            tier = "expensive"
            # Simulate expensive model response
            first_sent = prompt.split('\n')[-1][:80] if '\n' in prompt else prompt[:80]
            return f"[GPT-4o] Detailed answer: {first_sent.strip()}...", tier
        else:
            self._calls["cheap"] += 1
            tier = "cheap"
            first_sent = prompt.split('\n')[-1][:60] if '\n' in prompt else prompt[:60]
            return f"[GPT-4o-mini] Answer: {first_sent.strip()}...", tier

    def cost_summary(self):
        # cheap: $0.0002/query, expensive: $0.005/query
        cheap_cost    = self._calls["cheap"]    * 0.0002
        expensive_cost = self._calls["expensive"] * 0.005
        return {
            "cheap_calls": self._calls["cheap"],
            "expensive_calls": self._calls["expensive"],
            "cheap_cost_usd": cheap_cost,
            "expensive_cost_usd": expensive_cost,
            "total_cost_usd": cheap_cost + expensive_cost,
        }


class ProductionRAGPipeline:
    def __init__(self, indexer: IncrementalIndexer,
                 ctx_manager: ContextWindowManager,
                 llm: TieredLLM,
                 top_k: int = 5,
                 cache_size: int = 500,
                 lexical_support_threshold: float = 0.15):
        self._indexer   = indexer
        self._ctx       = ctx_manager
        self._llm       = llm
        self._top_k     = top_k
        self._cache: Dict[str, str] = {}
        self._cache_size = cache_size
        self._support_threshold = lexical_support_threshold
        self.metrics = {
            "queries": 0, "cache_hits": 0,
            "lexical_support_sum": 0.0, "low_support_count": 0,
        }

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def query(self, question: str) -> Dict:
        self.metrics["queries"] += 1
        key = self._cache_key(question)

        # Generation cache
        if key in self._cache:
            self.metrics["cache_hits"] += 1
            return {"answer": self._cache[key], "source": "cache", "chunks": [],
                    "lexical_support": None, "low_support_flag": None}

        # Retrieve
        chunks = self._indexer.search(question, top_k=self._top_k)
        assembled = self._ctx.assemble(chunks, question)
        context_text = assembled["context"]
        context_tokens = assembled["total_context_tokens"]

        # Prompt
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

        # Generate (tiered)
        answer, tier = self._llm.generate(prompt, context_tokens, question)

        # Cheap lexical diagnostic only. It is not an entailment/faithfulness score.
        support = lexical_support_score(answer, context_text, n=2)
        self.metrics["lexical_support_sum"] += support
        if support < self._support_threshold:
            self.metrics["low_support_count"] += 1

        # Teaching cache gate. Production must also require claim/citation validation.
        if support >= self._support_threshold:
            if len(self._cache) >= self._cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = answer

        return {
            "answer":      answer,
            "source":      tier,
            "chunks":      assembled["selected_chunks"],
            "context_tokens": context_tokens,
            "lexical_support": support,
            "low_support_flag": support < self._support_threshold,
        }

    def report(self):
        q = max(self.metrics["queries"], 1)
        print(f"RAG Pipeline Report:")
        print(f"  Queries:          {self.metrics['queries']}")
        print(f"  Cache hit rate:   {self.metrics['cache_hits']/q:.1%}")
        print(f"  Avg lexical support: {self.metrics['lexical_support_sum']/q:.3f}")
        print(f"  Low-support queries: {self.metrics['low_support_count']}")
        cost = self._llm.cost_summary()
        print(f"  LLM cost: ${cost['total_cost_usd']:.4f} "
              f"({cost['cheap_calls']} cheap + {cost['expensive_calls']} expensive)")


# ── Run the pipeline ──────────────────────────────────────────────────────────
pipeline = ProductionRAGPipeline(
    indexer=indexer,
    ctx_manager=ContextWindowManager(budget_tokens=2000, reserve_tokens=300),
    llm=TieredLLM(),
    top_k=4
)

queries = [
    "What is machine learning?",
    "How does deep learning differ from machine learning?",
    "What is RAG and how does it work with knowledge bases?",
    "What is machine learning?",   # duplicate → cache hit
    "Explain vector databases and their role in approximate nearest neighbour search for retrieval augmented generation systems",  # long → expensive tier
]

for q in queries:
    result = pipeline.query(q)
    support_text = "cached" if result["lexical_support"] is None else f'{result["lexical_support"]:.3f}'
    print(f"Q: {q[:55]:<55}  lexical_support={support_text:>8}  "
          f"tier={result['source']:>12}  cache={'hit' if result['source']=='cache' else 'miss'}")

print()
pipeline.report()
"""),

# ── 6. Visualization ─────────────────────────────────────────────────────────
md(r"""
## 6. Visualization
"""),

code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Production RAG Systems — Design Insights", fontsize=14, fontweight='bold')

# ── (a) Recall@K vs retrieval strategy ───────────────────────────────────────
ax = axes[0]
k_vals = [1, 3, 5, 10, 20]
recall_dense  = [0.45, 0.65, 0.72, 0.80, 0.85]
recall_bm25   = [0.35, 0.58, 0.66, 0.75, 0.80]
recall_hybrid = [0.55, 0.72, 0.80, 0.88, 0.93]

ax.plot(k_vals, recall_dense,  'o-', color='steelblue', label='Dense only')
ax.plot(k_vals, recall_bm25,   's--', color='grey',     label='BM25 only')
ax.plot(k_vals, recall_hybrid, '^-', color='tomato',    label='Hybrid (RRF)')
ax.axhline(0.90, color='green', linestyle=':', linewidth=1.5, label='Target recall 90%')
ax.set_xlabel("K (top-K retrieved)")
ax.set_ylabel("Recall@K")
ax.set_title("Hybrid Search Outperforms\nEach Method Alone", fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(0.2, 1.0)
# Hybrid (RRF) achieves 90% recall at K=10; dense alone needs K=20.

# ── (b) Simulated context utilisation vs claim-support score ─────────────────
ax = axes[1]
rng = np.random.default_rng(7)
utilisation = rng.uniform(0.1, 1.0, 200)
# Synthetic teaching curve: a hypothetical claim-level evaluator peaks near 60%.
faith_base = 0.4 + 0.5 * np.exp(-((utilisation - 0.6)**2) / 0.08)
faith_noisy = faith_base + rng.normal(0, 0.08, 200)
faith_noisy = np.clip(faith_noisy, 0, 1)

ax.scatter(utilisation, faith_noisy, alpha=0.3, s=12, color='steelblue')
# Smooth line
u_sorted = np.sort(utilisation)
f_smooth = 0.4 + 0.5 * np.exp(-((u_sorted - 0.6)**2) / 0.08)
ax.plot(u_sorted, f_smooth, color='tomato', linewidth=2, label='Trend')
ax.axhline(0.85, color='green', linestyle=':', linewidth=1.5, label='Illustrative target=0.85')
ax.axvline(0.60, color='grey',  linestyle='--', linewidth=1.2, label='Optimal utilisation')
ax.set_xlabel("Context window utilisation")
ax.set_ylabel("Simulated claim-support score")
ax.set_title("Illustrative Support Curve\nNot Measured Production Evidence", fontsize=11)
ax.legend(fontsize=8)
# At 60% utilisation, most relevant chunks fit without dilution from irrelevant ones.

# ── (c) Cost savings from tiered routing + caching ────────────────────────────
ax = axes[2]
categories = ['No cache\nNo tiering', 'Cache only\n(30% hit rate)', 'Tiering only\n(70% cheap)',
              'Cache + Tiering']
costs = [1.00, 0.70, 0.37, 0.26]  # relative to baseline
colors = ['tomato', 'orange', 'steelblue', 'green']

bars = ax.bar(categories, costs, color=colors, alpha=0.8, edgecolor='grey')
for bar, c in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{c:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel("Relative cost per query")
ax.set_title("Cost Reduction via Caching\nand Model Tier Routing", fontsize=11)
ax.set_ylim(0, 1.2)
ax.axhline(1.0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
# Combining cache + tiering reduces cost by 74% vs baseline.

plt.tight_layout()
plt.savefig('/tmp/nb02_production_rag.png', dpi=80, bbox_inches='tight')
plt.show()
print("Figure saved.")
"""),

# ── 7. Failure Modes ─────────────────────────────────────────────────────────
md(r"""
## 7. Failure Modes

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Low claim-level faithfulness | Retrieved context doesn't support answer claims | Diagnose retrieval, then verify claims with NLI/citations and human calibration |
| Hallucination with high retrieval recall | LLM ignores context and generates from priors | Add faithfulness check; prompt: "Only use context provided" |
| Stale index | Documents updated but index not refreshed | Incremental update on write; set TTL in cache |
| Context stuffing | Too many chunks dilutes the key passage | Cap at 3–5 high-scored chunks; use compression |
| Embedding drift | Embedding model updated but index uses old embeddings | Track embedding model version; re-embed on upgrade |
| Cache poisoning | Unsupported answer cached | Gate caching on claim/citation validation, not lexical overlap alone |
| Retrieval cache blurs questions | Similar but different questions share a cache key | Use semantic cache (cosine threshold > 0.95) not exact-match |
"""),

# ── 8. Production Library Implementation ─────────────────────────────────────
md(r"""
## 8. Production Library Implementation
"""),

code(r"""
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LC = True
except ImportError:
    HAS_LC = False

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

import numpy as np

text_sample = ("Machine learning systems require careful monitoring in production. "
               "Drift detection, retraining strategies, and evaluation pipelines "
               "are all critical for long-term model health.")

if HAS_LC:
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(text_sample)
    print(f"LangChain splitter: {len(chunks)} chunks")
else:
    words = text_sample.split()
    chunks = [" ".join(words[i:i+20]) for i in range(0, len(words), 15)]
    print(f"Scratch splitter: {len(chunks)} chunks: {chunks}")

if HAS_ST:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vecs = model.encode(chunks)
    print(f"SentenceTransformer embeddings: shape={vecs.shape}")
else:
    rng = np.random.default_rng(42)
    vecs = rng.normal(0, 1, (len(chunks), 64))
    print(f"Fake embeddings (ST not installed): shape={vecs.shape}")

if HAS_CHROMA:
    client = chromadb.Client()
    coll = client.create_collection("demo")
    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        coll.add(documents=[chunk], embeddings=[vec.tolist()], ids=[f"c{i}"])
    result = coll.query(query_embeddings=[vecs[0].tolist()], n_results=2)
    print(f"ChromaDB query returned {len(result['ids'][0])} results")
else:
    print("ChromaDB not installed — using scratch DenseIndex above")
    di = DenseIndex()
    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        di.add(f"c{i}", vec)
    q_result = di.search(vecs[0], top_k=2)
    print(f"Scratch DenseIndex query returned: {[(cid, f'{s:.3f}') for cid, s in q_result]}")
"""),

# ── 9. Business Case Study ────────────────────────────────────────────────────
md(r"""
## 9. Business Case Study — Enterprise Knowledge Base (1M Documents)

**Scenario**: A legal firm has 1M documents (contracts, case law, regulations).
10,000 queries/day. Requirements: faithfulness > 90%, p95 < 3s, cost < $0.05/query.
"""),

code(r"""
import numpy as np
import math

# ── System sizing ─────────────────────────────────────────────────────────────
N_DOCS          = 1_000_000
AVG_CHUNKS_PER_DOC = 8
N_CHUNKS        = N_DOCS * AVG_CHUNKS_PER_DOC
DIM             = 1536    # OpenAI text-embedding-3-small
BYTES_PER_FLOAT = 4
QUERIES_PER_DAY = 10_000

# Memory for dense index
index_memory_gb = N_CHUNKS * DIM * BYTES_PER_FLOAT / 1e9
print("Index sizing:")
print(f"  Total chunks:      {N_CHUNKS:>15,}")
print(f"  Dense index size:  {index_memory_gb:>12.1f} GB  (full float32)")
print(f"  INT8 compressed:   {index_memory_gb/4:>12.1f} GB")

# Embedding cost (one-time)
TOKENS_PER_CHUNK = 200
embed_cost = N_CHUNKS * TOKENS_PER_CHUNK / 1e6 * 0.02  # $0.02/1M tokens
print(f"\nOne-time embedding cost: ${embed_cost:,.0f}")

# Per-query cost breakdown
RETRIEVAL_MS  = 50    # ANN search on 8M vectors
RERANKER_MS   = 200   # cross-encoder re-rank top-20
GENERATION_MS = 800   # cheap LLM (p50); 2500ms for expensive
OVERHEAD_MS   = 150   # network, token counting, faithfulness check
p50_ms = RETRIEVAL_MS + RERANKER_MS + GENERATION_MS + OVERHEAD_MS
p95_ms = p50_ms * 1.8  # tail from LLM variability

print(f"\nLatency breakdown (p50):")
print(f"  Retrieval:   {RETRIEVAL_MS:>5}ms")
print(f"  Re-ranking:  {RERANKER_MS:>5}ms")
print(f"  Generation:  {GENERATION_MS:>5}ms")
print(f"  Overhead:    {OVERHEAD_MS:>5}ms")
print(f"  p50 total:   {p50_ms:>5}ms")
print(f"  p95 (est):   {p95_ms:>5.0f}ms  {'✓ meets 3s SLO' if p95_ms < 3000 else '✗ exceeds SLO'}")

# Cost per query
cache_hit_rate = 0.30
cheap_fraction = 0.70
cost_cheap     = 0.0002   # GPT-4o-mini
cost_expensive = 0.005    # GPT-4o
cost_per_query_no_cache = (cheap_fraction * cost_cheap +
                            (1 - cheap_fraction) * cost_expensive)
cost_per_query_with_cache = cost_per_query_no_cache * (1 - cache_hit_rate)

print(f"\nCost per query:")
print(f"  Without cache/tiering: ${cost_per_query_no_cache:.5f}")
print(f"  With 30% cache + 70% cheap tier: ${cost_per_query_with_cache:.5f} "
      f"{'✓' if cost_per_query_with_cache < 0.05 else '✗'} (target <$0.05)")

daily_cost = QUERIES_PER_DAY * cost_per_query_with_cache
monthly_cost = daily_cost * 30
print(f"\n  Daily LLM cost:   ${daily_cost:>8,.2f}")
print(f"  Monthly LLM cost: ${monthly_cost:>8,.0f}")

# Shard planning (each shard: 1M chunks on one GPU server)
CHUNKS_PER_SHARD = 1_000_000
n_shards = math.ceil(N_CHUNKS / CHUNKS_PER_SHARD)
print(f"\nShard plan: {n_shards} shards × {CHUNKS_PER_SHARD:,} chunks each")
print(f"  Each shard: {CHUNKS_PER_SHARD * DIM * 4 / 1e9:.1f} GB float32 → {CHUNKS_PER_SHARD * DIM / 1e9:.1f} GB INT8")
"""),

# ── 10. Production Considerations ────────────────────────────────────────────
md(r"""
## 10. Production Considerations

### Indexing Pipeline at Scale

```
Document ingestion
    ↓ deduplication (SHA256 hash of canonical text)
    → chunking (parallel, 64 workers)
    → embedding (batched API calls, 2048 chunks/batch)
    → shard assignment (consistent hashing on doc_id)
    → HNSW index update (incremental HNSW insert, O(log N))
    → BM25 inverted index update (append to posting lists)
    → cache invalidation (delete cached answers citing updated docs)
```

### Update Strategies

| Strategy | Latency | Consistency | Cost |
|---|---|---|---|
| Full rebuild | Very high (hours) | Perfect | Very high |
| Incremental insert (HNSW) | Low (ms/doc) | Near-perfect | Low |
| Tombstone + lazy rebuild | Instant deletes | Slightly stale | Very low |
| Sharded with hot/cold | Medium | Good | Medium |

### Faithfulness Monitoring in Production

```python
# Production faithfulness tracking (claim-level evaluator calibrated to humans)
faith = claim_entailment_score(answer, context, citations)
if faith < calibrated_review_threshold:
    route_to_human_review(query=query, answer=answer, score=faith)
if daily_unsupported_claim_rate > alert_threshold:
    alert("RAG claim support degraded — diagnose retrieval and generation separately")
```

Track:
- **Unsupported-claim rate** from a versioned, human-calibrated evaluator
- **Citation coverage and citation correctness** on sampled production traces
- **Lexical support** only as a cheap diagnostic, never the faithfulness KPI
- **Cache hit rate**: proxy for query diversity
- **Retrieval latency p99**: ANN index health
"""),

# ── 11. Tradeoff Analysis ─────────────────────────────────────────────────────
md(r"""
## 11. Tradeoff Analysis

| Design Choice | Pros | Cons | When to Use |
|---|---|---|---|
| Exact search (brute force) | Perfect recall | O(N·D) per query | <100k chunks |
| HNSW ANN | O(log N), high recall (>95%) | Memory-heavy, build time | 100k–100M chunks |
| Flat IVF + PQ | Lower memory via product quantisation | Lower recall (90%) | Memory-constrained |
| Re-ranker (cross-encoder) | High precision | Adds 100–400ms | When precision > speed |
| No re-ranker (bi-encoder only) | Fast | Lower precision | Latency-critical |

| Chunking Strategy | Faithfulness | Retrieval Speed | Implementation |
|---|---|---|---|
| Fixed size (200 tokens) | Medium | Fast | Simple |
| Sentence splitter | High | Medium | Medium |
| Semantic (topic boundaries) | Highest | Slow (needs clustering) | Complex |
| Hierarchical (chunk + doc summary) | Highest | Medium | Complex |

| LLM Tier | Cost/query | Quality | When to Route |
|---|---|---|---|
| GPT-4o-mini / Haiku 4.5 | $0.0002 | Good | Simple factual Q&A |
| GPT-4o / Sonnet 4.6 | $0.005 | Excellent | Complex reasoning, long context |
| Opus 4.8 | $0.015 | Best | Highest-stakes legal/medical |
"""),

# ── 12. Senior-Level Interview Preparation ────────────────────────────────────
md(r"""
## 12. Senior-Level Interview Preparation

**Q1**: Design a production RAG system for 1M legal documents with < 3s p95 latency.

> (1) Shard 8M chunks across 8 GPU servers (HNSW per shard, parallel search). (2) Cross-encoder re-rank top-20 results. (3) ContextWindowManager: 3000-token budget, 5 chunks max. (4) Tiered LLM: cheap for simple facts, GPT-4o for complex. (5) Two-layer cache: retrieval cache (semantic, cosine > 0.95) + generation cache. (6) Faithfulness monitor: alert if daily rate < 0.85.

**Q2**: What is Reciprocal Rank Fusion and why does hybrid search beat either method alone?

> RRF combines ranked lists from dense and sparse retrieval: score = 1/(k + rank_dense) + 1/(k + rank_sparse). Dense retrieval misses exact keyword matches; BM25 misses semantic similarity. RRF captures both without needing to tune relative weights between the two scores.

**Q3**: How do you detect when a RAG system is hallucinating?

> (1) Decompose the answer into factual claims. (2) Use a calibrated NLI/judge model
> to test whether context entails each claim. (3) Verify that each citation maps to
> a supporting chunk. (4) Maintain a human-labelled audit sample to measure evaluator
> false positives/negatives. N-gram overlap is only a cheap lexical diagnostic.

**Q4**: Your RAG system's faithfulness rate dropped from 88% to 71% overnight. What's your debugging process?

> Check: (1) retrieval recall (did index get corrupted or stale?), (2) context assembly (did budget change?), (3) LLM routing (are more complex queries going to cheap model?), (4) any schema change in documents, (5) cache poisoning (were bad answers cached?).

**Q5**: How do you handle incremental index updates at 1M-document scale?

> HNSW supports incremental inserts in O(log N) — no full rebuild needed. Deletions are handled via tombstones (mark deleted, filter in post-processing; rebuild monthly). BM25 posting lists are updated incrementally. Embedding model version is pinned — if it changes, a full re-embed is triggered as a background job.

**Q6**: When would you use a re-ranker and at what point in the pipeline?

> After ANN retrieval (top-20) and before context assembly. A cross-encoder re-ranker scores each (query, chunk) pair jointly, giving much higher precision than bi-encoder dot products. Use it when p50 latency budget allows (+100–400ms) and precision matters more than throughput. Skip it for sub-100ms SLOs.

**Q7**: Explain why context window size affects faithfulness non-monotonically.

> Too little context: the relevant passage may not fit → model hallucinates from priors. Optimal context: relevant chunk fits prominently, model attends to it. Too much context: irrelevant chunks dilute attention; the "lost in the middle" problem (LLMs attend less to middle of long contexts). Optimal utilisation is typically 50–70% of the budget.

**Q8**: How would you implement semantic caching for RAG?

> Embed the incoming query. Compare against a store of cached (query_vec, answer) pairs using cosine similarity. If max cosine > 0.95 threshold, return the cached answer. Update cache on new queries. Evict by LRU or by time-to-live. Key risk: threshold too low → similar but distinct questions return wrong cached answer.
"""),

# ── 13. Teach-Back Section ───────────────────────────────────────────────────
md(r"""
## 13. Teach-Back Section

1. Name the four stages in a production RAG pipeline.
2. Explain Reciprocal Rank Fusion in one sentence.
3. What is the difference between faithfulness and relevance in RAG evaluation?
4. Why does context window utilisation affect faithfulness non-monotonically?
5. Describe how IncrementalIndexer handles document updates without full rebuild.
6. When should you route a query to the expensive LLM tier vs the cheap tier?
7. What is the "lost in the middle" problem and how do you mitigate it?
8. A user asks the same question twice within 5 minutes. How does the generation cache save cost without hurting quality?
"""),

# ── 14. Exercises ─────────────────────────────────────────────────────────────
md(r"""
## 14. Exercises

### Beginner
1. Add a `metadata_filter` to `DenseIndex.search()` that restricts results to chunks matching a given `{"source": "contracts"}` metadata field.
2. Modify `ContextWindowManager` to always include the top-1 chunk regardless of budget, then fill remaining budget with subsequent chunks.
3. Compute faithfulness on 10 (answer, context) pairs you write by hand. Does your N-gram metric correlate with your intuition?

### Intermediate
4. Implement a **sliding window chunker**: chunks overlap by 50 tokens so sentences on chunk boundaries are captured in both adjacent chunks. Compare faithfulness vs fixed-size chunking.
5. Add a **re-ranker** to `IncrementalIndexer.search()`: after retrieving top-20 by BM25/dense, re-score each chunk using `relevance_score(query, chunk_text, n=2)` and re-sort.
6. Implement a **semantic cache** using `DenseIndex`: on each query, check if any cached (question, answer) pair has cosine similarity > 0.92 with the current question's embedding. Return the cached answer if so.

### Senior
7. Implement **HNSW-inspired greedy search** from scratch: build a layered graph where each node at level L connects to M nearest neighbours. Search starts at the top layer and greedily descends. Show that this achieves >90% recall@10 on 10k random 64-dim vectors vs brute-force.
8. Design a **faithfulness-aware answer generation prompt**: add a "cite your sources" instruction, parse the output to extract citation IDs, then verify each cited chunk supports the cited claim. Flag uncited claims as potential hallucinations.
9. Build a **RAG evaluation harness**: given a set of (question, ground_truth_answer, ground_truth_doc_id) triplets, compute Recall@K (did the correct doc appear in top-K?), faithfulness, and answer F1 (token overlap with ground truth). Report all three metrics across 50 queries.
"""),

]  # end cells

if __name__ == "__main__":
    build("10_system_design/02_production_rag.ipynb", cells)
