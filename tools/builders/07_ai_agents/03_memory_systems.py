"""Builder for Lesson AGT-03 — Memory Systems."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # AGT-03 · Memory Systems
    ### Section 07 — Agentic AI · *ML/AI Senior Mastery Curriculum*

    > Lesson AGT-01 built a ReAct agent; Lesson AGT-02 added planning and tool validation.
    > Both agents were **stateless** — each turn started fresh with no memory of the past.
    > Real assistants remember preferences, past interactions, and domain knowledge.
    > This notebook teaches the four types of agent memory, implements each from scratch,
    > and shows how to integrate them into a coherent memory-augmented agent.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Four memory types**: working, episodic, semantic, procedural — what each stores
      and when the agent retrieves from each.
    - **Working memory from scratch**: token-limited context deque; eviction policy.
    - **Episodic memory from scratch**: trajectory log with recency and cosine retrieval.
    - **Semantic memory from scratch**: mini vector store (same as RAG-01/RAG-05 RAG).
    - **Procedural memory from scratch**: few-shot example cache with retrieval.
    - **Memory routing**: how an agent decides which memory type to query.
    - **MemGPT / Letta architecture**: hierarchical context management
      (in-context = working, out-of-context = episodic + semantic archival).
    - **Production considerations**: memory staleness, privacy, cost of retrieval.

    **Why it matters**
    - Without memory, every session is a blank slate — the agent cannot improve, cannot
      personalise, and cannot build on prior work. Memory is what turns a chatbot into an
      assistant. Understanding the four memory types lets you design the right persistence
      layer for each use case instead of stuffing everything into the context window.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Cognitive science origin.** Atkinson & Shiffrin (1968) proposed the multi-store
    model of human memory: sensory → short-term (working) → long-term. Long-term memory
    is further divided into episodic (autobiographical events), semantic (facts about the
    world), and procedural (how to do things). AI memory architectures mirror this.

    **RAG as semantic memory.** The RAG pipeline (RAG-01–RAG-08) is effectively an external
    semantic memory — the model retrieves factual knowledge that doesn't fit in its weights
    or context. The innovation: retrieval at inference time rather than baking knowledge
    into weights.

    **MemGPT (Packer et al., 2023).** Extended the idea to all four memory types:
    in-context memory (working), archival memory (episodic + semantic, stored in a vector
    DB), and recursive summarisation to manage context window limits. Renamed Letta in 2024.

    **LangMem, Zep, Mem0 (2024).** Production memory services that layer on top of LLMs:
    extract memories from conversations, store them in vector DBs, inject relevant memories
    into future prompts. Same four-type architecture with production infrastructure.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Four memory types — the filing cabinet analogy:**
    ```
    Working memory   — your desk (limited space, immediate access, cleared each session)
    Episodic memory  — your diary (dated events, "last Tuesday we discussed X")
    Semantic memory  — your encyclopedia (facts, "the capital of France is Paris")
    Procedural memory — your muscle memory (how-to, "to write SQL: SELECT col FROM tbl")
    ```

    **When the agent uses each:**

    | Memory type | Query trigger | Example |
    |---|---|---|
    | Working | Always | Current conversation turn |
    | Episodic | "Do you remember when...?" / recent history | "Last session you were analysing EU data" |
    | Semantic | Factual question needing knowledge | "What is the formula for NDCG?" |
    | Procedural | Task requiring known pattern | "Write a SQL query" → retrieve SQL template |

    **MemGPT hierarchy:**
    ```
    In-context  [working: recent msgs + retrieved memories]   ← fast, expensive per-token
    Out-of-context [episodic: all past turns in vector DB  ]  ← cheap storage, retrieval cost
               [semantic: knowledge base in vector DB     ]  ← cheap storage, retrieval cost
    ```

    When context fills: summarise + evict oldest → archive to episodic memory → space freed.
    """),

    code(r"""
    import re
    import math
    import json
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import deque, defaultdict
    from typing import List, Optional, Dict, Any

    rng_global = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Working memory as bounded queue

    Let $C$ = token budget (e.g. 4096 tokens). Working memory holds messages
    $m_1, m_2, \ldots, m_k$ where $\sum_{i=1}^{k} |m_i| \le C$.

    **FIFO eviction**: when $\sum |m_i| + |m_{new}| > C$, evict $m_1$ (oldest).
    In practice: evict oldest non-system messages first; preserve system prompt.

    ### 4.2 Cosine similarity for episodic/semantic retrieval

    $\text{sim}(\mathbf{q}, \mathbf{m}) = \frac{\mathbf{q} \cdot \mathbf{m}}{\|\mathbf{q}\| \|\mathbf{m}\|}$

    For episodic memory, blend recency with semantic similarity:

    $\text{score}(m, t) = \alpha \cdot \text{sim}(\mathbf{q}, \mathbf{m}) + (1-\alpha) \cdot \text{recency}(t)$

    $\text{recency}(t) = e^{-\lambda (t_{now} - t_m)}$ — exponential decay; $\lambda$ controls
    how fast old memories lose priority.

    ### 4.3 Procedural memory — nearest-neighbour retrieval

    Procedural memory maps task patterns to procedures. Given query $\mathbf{q}$:

    $p^* = \arg\max_{p \in P} \text{sim}(\mathbf{q}, \mathbf{e}_p)$

    where $\mathbf{e}_p$ is the embedding of procedure $p$'s description.

    Return $p^*$ as a few-shot example prepended to the prompt.

    ### 4.4 Token counting approximation

    Exact token counts require the model's tokeniser (BPE). Approximation:
    $|m|_{\text{tokens}} \approx \lceil |m|_{\text{chars}} / 4 \rceil$

    (empirically valid for English text with GPT-family tokenisers).
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a — Working memory (token-limited deque)
    """),

    code(r"""
    # 5a. Working memory: bounded context window with FIFO eviction.

    def approx_tokens(text):
        # Approximate tokenisation: 4 chars per token (valid for English).
        return max(1, math.ceil(len(str(text)) / 4))

    class WorkingMemory:
        # Bounded context window. Evicts oldest non-system messages when full.
        def __init__(self, token_budget=2000):
            self.token_budget = token_budget
            self.messages = deque()   # (role, content, token_count)
            self.system_prompt = None
            self.system_tokens = 0

        def set_system(self, content):
            self.system_prompt = content
            self.system_tokens = approx_tokens(content)

        def add(self, role, content):
            tokens = approx_tokens(content)
            # Evict until there's room.
            while self.used_tokens() + tokens > self.token_budget and self.messages:
                self.messages.popleft()
            self.messages.append((role, content, tokens))

        def used_tokens(self):
            return self.system_tokens + sum(t for _, _, t in self.messages)

        def context(self):
            msgs = []
            if self.system_prompt:
                msgs.append({'role': 'system', 'content': self.system_prompt})
            for role, content, _ in self.messages:
                msgs.append({'role': role, 'content': content})
            return msgs

        def summary(self):
            return {
                'messages': len(self.messages),
                'used_tokens': self.used_tokens(),
                'budget': self.token_budget,
                'utilisation': self.used_tokens() / self.token_budget,
            }

    wm = WorkingMemory(token_budget=500)
    wm.set_system('You are a helpful research assistant.')

    conversations = [
        ('user', 'What is gradient descent?'),
        ('assistant', 'Gradient descent is an iterative optimisation algorithm that minimises a loss function by moving in the direction of steepest descent.'),
        ('user', 'Can you show me the update rule?'),
        ('assistant', 'theta = theta - learning_rate * gradient(loss, theta)'),
        ('user', 'What learning rate should I use?'),
        ('assistant', 'Typical values: 0.1 for SGD, 3e-4 for Adam (Karpathy constant). Use learning rate finder in practice.'),
        ('user', 'Tell me about Adam optimizer'),
        ('assistant', 'Adam combines momentum (first moment) and RMSProp (second moment) for adaptive per-parameter learning rates.'),
    ]

    print('Working memory — filling up:')
    for role, content in conversations:
        wm.add(role, content)
        s = wm.summary()
        print(f'  [{role:9s}] msgs={s["messages"]} tokens={s["used_tokens"]}/{s["budget"]} ({100*s["utilisation"]:.0f}%)')

    print(f'\nFinal context ({len(wm.context())} messages):')
    for m in wm.context():
        print(f'  {m["role"]:9s}: {m["content"][:60]}...')
    """),

    md(r"""
    ### 5b — Episodic memory (trajectory log with cosine + recency retrieval)
    """),

    code(r"""
    # 5b. Episodic memory: dated interaction log with hybrid retrieval.

    def simple_embed(text, dim=32):
        # Deterministic character-hash embedding (stand-in for sentence transformer).
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.standard_normal(dim)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    class EpisodicMemory:
        # Autobiographical log: stores (timestamp, summary, embedding).
        def __init__(self, decay_rate=0.05, embed_dim=32):
            self.episodes = []   # {t, summary, embedding}
            self.decay = decay_rate
            self.dim = embed_dim
            self._clock = 0

        def store(self, summary):
            self._clock += 1
            self.episodes.append({
                't': self._clock,
                'summary': summary,
                'embedding': simple_embed(summary, self.dim),
            })

        def retrieve(self, query, k=3, alpha=0.6):
            # alpha=0.6: 60% semantic, 40% recency.
            if not self.episodes:
                return []
            q_emb = simple_embed(query, self.dim)
            t_max = max(e['t'] for e in self.episodes)
            scored = []
            for ep in self.episodes:
                sem = cosine(q_emb, ep['embedding'])
                rec = math.exp(-self.decay * (t_max - ep['t']))
                score = alpha * sem + (1 - alpha) * rec
                scored.append((score, ep))
            scored.sort(key=lambda x: -x[0])
            return [(ep['summary'], round(score, 3)) for score, ep in scored[:k]]

    em = EpisodicMemory()

    past_sessions = [
        'Session 1: User asked about linear regression. Discussed OLS, R-squared, assumptions.',
        'Session 2: Reviewed gradient boosting and XGBoost. User confused about learning rate vs n_estimators.',
        'Session 3: Built a RAG pipeline. User had FAISS indexing issue — fixed by normalising vectors.',
        'Session 4: Discussed transformer attention. User wanted intuition for Q/K/V matrices.',
        'Session 5: Deployed model to FastAPI. User encountered CORS issues with the prediction endpoint.',
        'Session 6: User asked about chunking strategies for long PDFs. Chose recursive splitting.',
    ]

    for s in past_sessions:
        em.store(s)

    print('Episodic memory retrieval:')
    queries = ['embedding and vector search', 'deployment and API issues', 'tree-based models']
    for q in queries:
        results = em.retrieve(q, k=2)
        print(f'\n  Query: "{q}"')
        for summary, score in results:
            print(f'    [{score:.3f}] {summary[:70]}...')
    """),

    md(r"""
    ### 5c — Semantic memory (mini vector store — factual knowledge base)
    """),

    code(r"""
    # 5c. Semantic memory: factual knowledge base with cosine retrieval.
    # This is the RAG pattern from Section 06, used as memory.

    class SemanticMemory:
        # Stores facts/documents and retrieves by cosine similarity.
        def __init__(self, embed_dim=32):
            self.store = []   # {id, text, embedding, category}
            self.dim = embed_dim

        def add(self, text, category='general'):
            self.store.append({
                'id': len(self.store),
                'text': text,
                'embedding': simple_embed(text, self.dim),
                'category': category,
            })

        def retrieve(self, query, k=3, filter_category=None):
            if not self.store:
                return []
            q_emb = simple_embed(query, self.dim)
            candidates = [d for d in self.store
                          if filter_category is None or d['category'] == filter_category]
            scored = [(cosine(q_emb, d['embedding']), d) for d in candidates]
            scored.sort(key=lambda x: -x[0])
            return [(d['text'], round(s, 3)) for s, d in scored[:k]]

    sm = SemanticMemory()

    # Knowledge base — facts the agent knows about the user's domain.
    knowledge = [
        ('NDCG@k measures ranking quality: sum of discounted gains / ideal DCG.', 'metrics'),
        ('BM25 uses TF saturation (k1) and length normalisation (b) parameters.', 'retrieval'),
        ('Cosine similarity is scale-invariant; dot product is not.', 'embeddings'),
        ('HyDE improves recall for abstract queries by embedding hypothetical documents.', 'retrieval'),
        ('RRF formula: score(d) = sum(1 / (k + rank_r(d))) with k=60.', 'retrieval'),
        ('Adam optimizer: combines momentum (m) and RMSProp (v); lr=3e-4 default.', 'optimization'),
        ('Gradient descent: theta = theta - lr * grad(loss, theta).', 'optimization'),
        ('Transformer attention: Q,K,V = XW_Q, XW_K, XW_V; attn = softmax(QK/sqrt(d))V.', 'architecture'),
        ('Cross-encoder encodes query+doc jointly; cannot pre-compute doc embeddings.', 'retrieval'),
        ('HNSW builds a multi-layer proximity graph; recall vs speed tradeoff via ef.', 'indexing'),
    ]

    for text, cat in knowledge:
        sm.add(text, category=cat)

    print('Semantic memory retrieval:')
    for q in ['ranking metrics', 'how does Adam work', 'vector search algorithm']:
        results = sm.retrieve(q, k=2)
        print(f'\n  Query: "{q}"')
        for text, score in results:
            print(f'    [{score:.3f}] {text[:75]}')
    """),

    md(r"""
    ### 5d — Procedural memory (few-shot example cache)
    """),

    code(r"""
    # 5d. Procedural memory: maps task patterns to how-to procedures / few-shot examples.

    class ProceduralMemory:
        # Stores task→procedure pairs; retrieves best-matching procedure for a task.
        def __init__(self, embed_dim=32):
            self.procedures = []   # {task_desc, procedure, embedding}
            self.dim = embed_dim

        def store(self, task_description, procedure):
            self.procedures.append({
                'task': task_description,
                'procedure': procedure,
                'embedding': simple_embed(task_description, self.dim),
            })

        def retrieve(self, query, k=1):
            if not self.procedures:
                return []
            q_emb = simple_embed(query, self.dim)
            scored = [(cosine(q_emb, p['embedding']), p) for p in self.procedures]
            scored.sort(key=lambda x: -x[0])
            return [(p['task'], p['procedure'], round(s, 3)) for s, p in scored[:k]]

    pm = ProceduralMemory()

    # Stored procedures — reusable how-to patterns.
    procedures = [
        ('write a SQL aggregation query',
         'SELECT group_col, AGG(value_col) FROM table GROUP BY group_col ORDER BY 2 DESC LIMIT N;'),
        ('evaluate a ranking model',
         '1. Compute NDCG@10. 2. Compute MRR. 3. Compare to BM25 baseline. 4. A/B test top-K.'),
        ('debug a slow API endpoint',
         '1. Profile with cProfile. 2. Check DB query count (N+1?). 3. Add caching. 4. Load test.'),
        ('chunk a long document for RAG',
         '1. Try recursive splitting (chunk=256, overlap=32). 2. Evaluate retrieval accuracy. 3. Try semantic chunking if accuracy < 0.7.'),
        ('train a gradient boosting model',
         '1. Start with n_estimators=100, max_depth=4, lr=0.1. 2. Tune via cross-val. 3. Add early stopping.'),
    ]

    for task, proc in procedures:
        pm.store(task, proc)

    print('Procedural memory retrieval:')
    for q in ['SQL query for totals', 'measure ranking quality', 'make inference faster']:
        results = pm.retrieve(q, k=1)
        print(f'\n  Task: "{q}"')
        for task, proc, score in results:
            print(f'    [{score:.3f}] Matched: {task}')
            print(f'    Procedure: {proc[:80]}')
    """),

    md(r"""
    ### 5e — Memory-augmented agent with routing
    """),

    code(r"""
    # 5e. Memory-augmented agent: routes queries to the right memory type.

    class MemoryRouter:
        # Decides which memory types to query for a given user message.
        def route(self, message):
            msg_lower = message.lower()
            types = []
            # Episodic: references to past sessions/history.
            if any(t in msg_lower for t in ['remember', 'last time', 'earlier', 'session', 'before', 'yesterday']):
                types.append('episodic')
            # Semantic: factual questions.
            if any(t in msg_lower for t in ['what is', 'how does', 'explain', 'formula', 'define', 'what are']):
                types.append('semantic')
            # Procedural: how-to / task execution.
            if any(t in msg_lower for t in ['how to', 'write', 'implement', 'build', 'create', 'show me how', 'steps']):
                types.append('procedural')
            # Working memory: always.
            types.append('working')
            return list(dict.fromkeys(types))   # deduplicate, preserve order

    class MemoryAugmentedAgent:
        def __init__(self, working_budget=1000):
            self.working = WorkingMemory(token_budget=working_budget)
            self.episodic = EpisodicMemory()
            self.semantic = SemanticMemory()
            self.procedural = ProceduralMemory()
            self.router = MemoryRouter()
            self.working.set_system('You are a helpful ML research assistant with memory.')

        def load_knowledge(self, facts, category='general'):
            for fact in facts:
                self.semantic.add(fact, category)

        def load_procedures(self, procs):
            for task, proc in procs:
                self.procedural.store(task, proc)

        def archive_session(self, summary):
            self.episodic.store(summary)

        def respond(self, user_message, verbose=True):
            memory_types = self.router.route(user_message)
            retrieved = {}

            if 'episodic' in memory_types:
                retrieved['episodic'] = self.episodic.retrieve(user_message, k=2)
            if 'semantic' in memory_types:
                retrieved['semantic'] = self.semantic.retrieve(user_message, k=2)
            if 'procedural' in memory_types:
                retrieved['procedural'] = self.procedural.retrieve(user_message, k=1)

            # Build augmented context.
            augmented = user_message
            for mtype, results in retrieved.items():
                if results:
                    snippets = [r[0] if mtype == 'episodic' else r[0] for r in results]
                    augmented += f'\n[{mtype.upper()} MEMORY]: ' + ' | '.join(snippets[:1])

            self.working.add('user', augmented)

            # Simulate LLM response (in prod: call API with self.working.context()).
            response = f'[SIMULATED RESPONSE to: "{user_message[:50]}..." | memory types used: {memory_types}]'
            self.working.add('assistant', response)

            if verbose:
                print(f'\nUser: {user_message}')
                print(f'  Memory types queried: {memory_types}')
                for mtype, results in retrieved.items():
                    if results:
                        print(f'  [{mtype}] retrieved: {results[0][0][:60]}...')
                print(f'  Working memory: {self.working.summary()["used_tokens"]} tokens used')

            return response, retrieved

    agent = MemoryAugmentedAgent(working_budget=800)
    agent.load_knowledge(
        ['NDCG@k measures ranking quality.', 'BM25 uses TF saturation and length normalisation.'],
        category='retrieval'
    )
    agent.load_procedures([
        ('evaluate a ranking model', '1. NDCG@10. 2. MRR. 3. A/B test.'),
        ('write SQL query', 'SELECT col, AGG(val) FROM tbl GROUP BY col;'),
    ])
    agent.archive_session('Session 1: User studied RAG pipelines and vector databases.')
    agent.archive_session('Session 2: Debugged slow FAISS index — fixed by normalising vectors.')

    print('Memory-augmented agent demo:')
    agent.respond('What is NDCG?')
    agent.respond('Do you remember what we worked on last session?')
    agent.respond('How do I write a SQL aggregation query?')
    agent.respond('Hi, what can you help me with today?')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Memory types: capacity vs access speed.
    fig, ax = plt.subplots(figsize=(10, 5))
    memory_types = ['Working\n(context)', 'Episodic\n(event log)', 'Semantic\n(knowledge)', 'Procedural\n(how-to)']
    capacity    = [1,   8,   9,  6]   # relative capacity (1=small, 10=large)
    speed       = [10,  5,   5,  6]   # access speed (10=fast)
    persistence = [1,  9,  10,  9]   # persistence across sessions

    x = np.arange(len(memory_types))
    w = 0.25
    ax.bar(x - w,   capacity,    w, label='Capacity',    color='steelblue', alpha=0.8)
    ax.bar(x,       speed,       w, label='Access speed', color='seagreen',  alpha=0.8)
    ax.bar(x + w,   persistence, w, label='Persistence',  color='coral',     alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(memory_types, fontsize=10)
    ax.set_ylabel('Relative score (1–10)'); ax.set_ylim(0, 12)
    ax.set_title('Figure 1 — Memory types: capacity vs speed vs persistence')
    ax.legend()
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 1.** Comparison of the four agent memory types across three dimensions.
    **Working memory** (context window) has the smallest capacity (4K–128K tokens) but
    the fastest access (already in-context, zero retrieval latency). It is not persistent
    across sessions. **Episodic and semantic memory** have high capacity (millions of entries
    in a vector DB) and are persistent, but require retrieval latency (embedding lookup +
    ANN search, typically 5–50ms). **Procedural memory** is mid-capacity and persistent —
    few-shot examples retrieved by task similarity. The key design decision: what goes in
    working memory (expensive per-token) vs. out-of-context memory (cheap storage, retrieval
    cost on demand).
    """),

    code(r"""
    # Figure 2 — Episodic memory: recency vs semantic score tradeoff.
    queries_fig = ['vector search and FAISS', 'deployment issues', 'training models']
    episode_labels = [f'S{i+1}' for i in range(len(past_sessions))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, q in zip(axes, queries_fig):
        q_emb = simple_embed(q, 32)
        t_max = len(past_sessions)
        sem_scores = [cosine(q_emb, simple_embed(s, 32)) for s in past_sessions]
        rec_scores = [math.exp(-0.05 * (t_max - (i+1))) for i in range(len(past_sessions))]
        hybrid = [0.6*s + 0.4*r for s, r in zip(sem_scores, rec_scores)]

        ax.plot(episode_labels, sem_scores, 'o-', label='Semantic', color='steelblue')
        ax.plot(episode_labels, rec_scores, 's--', label='Recency', color='coral')
        ax.plot(episode_labels, hybrid, 'd-', label='Hybrid (α=0.6)', color='seagreen', lw=2)
        ax.set_title(f'Query: "{q[:20]}..."', fontsize=9)
        ax.set_ylabel('Score'); ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
    plt.suptitle('Figure 2 — Episodic retrieval: semantic vs recency vs hybrid score')
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 2.** Episodic memory retrieval scores for three queries across six past sessions.
    The **semantic score** (blue) picks the most topically relevant episode regardless of age.
    The **recency score** (red) decays exponentially — always highest for the most recent
    session. The **hybrid score** (green, α=0.6) balances both: it retrieves recent episodes
    slightly higher than old ones, but can still surface an old episode when it is highly
    relevant. Tune α based on your use case: higher α = more semantic, lower α = more
    recency-biased. A personal assistant tends to want recency-biased retrieval; a knowledge
    base tends to want pure semantic retrieval.
    """),

    code(r"""
    # Figure 3 — Working memory token utilisation over conversation.
    wm2 = WorkingMemory(token_budget=600)
    wm2.set_system('You are an expert ML assistant.')
    turns = [
        ('user', 'Explain transformer attention in detail with the full mathematical derivation of the attention scores and how multi-head attention extends single-head attention.'),
        ('assistant', 'Transformer attention computes Q, K, V = XW_Q, XW_K, XW_V. Attention = softmax(QK^T / sqrt(d_k))V. Multi-head: run H attention heads in parallel, concatenate, project.'),
        ('user', 'What are the computational complexity implications of this?'),
        ('assistant', 'Self-attention is O(n^2 * d) in time and O(n^2) in memory where n is sequence length. This is the bottleneck for long documents — hence FlashAttention, sparse attention, etc.'),
        ('user', 'How does FlashAttention address this?'),
        ('assistant', 'FlashAttention tiles the QKV matrices to fit in SRAM, avoiding materialising the full O(n^2) attention matrix. Same result, IO-optimal. Reduces memory from O(n^2) to O(n).'),
        ('user', 'What about for very long contexts like 100K tokens?'),
        ('assistant', 'At 100K tokens: O(n^2) = 10B operations — too slow. Use sparse attention (Longformer, BigBird: local windows + global tokens), linear attention approximations (Performer, RWKV), or state-space models (Mamba, S4).'),
    ]

    utilisation = []
    evictions = []
    prev_len = len(wm2.messages)
    for role, content in turns:
        wm2.add(role, content)
        s = wm2.summary()
        utilisation.append(s['utilisation'])
        evicted = max(0, prev_len + 1 - len(wm2.messages))
        evictions.append(evicted)
        prev_len = len(wm2.messages)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x_turns = list(range(1, len(turns) + 1))
    ax1.plot(x_turns, [u * 100 for u in utilisation], 'o-', color='steelblue', lw=2, label='Token utilisation %')
    ax1.axhline(100, color='red', ls='--', alpha=0.5, label='Budget limit')
    ax1.set_xlabel('Conversation turn'); ax1.set_ylabel('Token utilisation %')
    ax1.set_ylim(0, 120)

    ax2 = ax1.twinx()
    ax2.bar(x_turns, evictions, alpha=0.4, color='coral', label='Messages evicted')
    ax2.set_ylabel('Messages evicted per turn')

    ax1.set_title('Figure 3 — Working memory token utilisation and eviction over turns')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 3.** Working memory utilisation over conversation turns. The blue line rises
    as messages accumulate. When utilisation exceeds the budget, the oldest messages are
    evicted (red bars). After eviction, utilisation drops then rises again. The information
    lost via eviction — important context from early in the conversation — is why production
    systems combine working memory with episodic memory: evicted messages are archived rather
    than discarded, and can be retrieved if the user references them later.
    """),

    code(r"""
    # Figure 4 — Memory routing heatmap: which queries trigger which memory types.
    test_queries = [
        'What is the formula for cosine similarity?',
        'Do you remember what we discussed last week?',
        'How do I implement a vector database?',
        'Hi, how are you doing today?',
        'What was that RAG issue we fixed before?',
        'Explain what NDCG measures',
        'Write me a SQL query to count orders by region',
    ]
    router = MemoryRouter()
    memory_labels = ['episodic', 'semantic', 'procedural']

    matrix = np.zeros((len(test_queries), len(memory_labels)))
    for i, q in enumerate(test_queries):
        types = router.route(q)
        for j, mtype in enumerate(memory_labels):
            if mtype in types:
                matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(memory_labels))); ax.set_xticklabels(memory_labels)
    ax.set_yticks(range(len(test_queries)))
    ax.set_yticklabels([q[:45] + '...' if len(q) > 45 else q for q in test_queries], fontsize=8)
    ax.set_title('Figure 4 — Memory routing: which query types activate which memory stores')
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 4.** Memory routing heatmap. Blue cells indicate which memory types are
    queried for each user message. Factual questions ("What is...", "Explain...") trigger
    **semantic memory**. Reference to past sessions ("remember", "last week") trigger
    **episodic memory**. How-to requests ("How do I", "Write me") trigger **procedural
    memory**. Simple greetings ("Hi") trigger only working memory (always active).
    The router is implemented as keyword matching here; in production, route with a
    small classification model or by asking the LLM to classify the query type.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Context overflow** | Agent loses early instructions | Working memory full; eviction removes key facts | Archive key context to episodic; keep system prompt pinned |
    | **Stale memory** | Agent references outdated information | Semantic memory not updated; old facts retrieved | Add metadata timestamps; filter by age; implement memory update/delete |
    | **Memory hallucination** | Agent confabulates past events | Retrieval noise; agent fills gaps with invention | Require high-confidence threshold for episodic retrieval; cite sources |
    | **Memory poisoning** | Agent corrupted by false stored memory | Malicious or erroneous episode stored | Validate before storing; use human review for high-stakes memories |
    | **Retrieval latency** | Agent response slow for factual Q&A | Embedding + ANN search on every turn | Cache recent retrievals; pre-embed common queries; async retrieval |
    | **Privacy leakage** | Agent reveals another user's memories | Multi-tenant memory store without namespace isolation | Namespace all stores by user_id; encrypt at rest |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 Letta / MemGPT architecture pattern (guarded import).
    try:
        import letta  # noqa: F401
        print('Letta available — use letta.create_client() for persistent memory agents.')
    except ImportError:
        lines = [
            '[letta not installed — MemGPT/Letta architecture pattern]:',
            '  from letta import create_client',
            '  client = create_client()',
            '  agent = client.create_agent()',
            '  # In-context: main_memory (core facts, user preferences)',
            '  # Out-of-context: archival_memory (vector DB), recall_memory (conversation log)',
            '  # Agent manages context automatically via memory tool calls:',
            '  #   memory_search(query) — retrieve from archival/recall',
            '  #   memory_insert(text) — add to archival',
            '  #   core_memory_append(field, value) — update working memory',
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 LangMem / Mem0 pattern (guarded import).
    try:
        from mem0 import Memory   # noqa: F401
        print('Mem0 available.')
    except ImportError:
        lines = [
            '[mem0 not installed — Mem0 pattern]:',
            '  from mem0 import Memory',
            '  m = Memory()',
            '  # Add: m.add("User prefers concise answers", user_id="alice")',
            '  # Search: results = m.search("communication style", user_id="alice")',
            '  # Update: m.update(memory_id, "User now prefers detailed answers")',
            '  # Integrates with OpenAI/Anthropic; auto-extracts memories from conversations',
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.3 LangChain conversation memory patterns (guarded import).
    try:
        from langchain.memory import ConversationSummaryBufferMemory  # noqa: F401
        print('LangChain memory available.')
    except ImportError:
        lines = [
            '[langchain not installed — LangChain memory patterns]:',
            '  ConversationBufferMemory — stores full history (like our WorkingMemory)',
            '  ConversationSummaryMemory — summarises old turns, keeps recent verbatim',
            '  ConversationSummaryBufferMemory — hybrid: buffer recent, summarise old',
            '  VectorStoreRetrieverMemory — retrieves by semantic similarity (episodic)',
            '',
            '  Example:',
            '  from langchain.memory import ConversationSummaryBufferMemory',
            '  memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)',
            '  chain = LLMChain(llm=llm, prompt=prompt, memory=memory)',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Personal Research Assistant

    **Scenario.** A research firm builds a personal AI assistant for each analyst.
    The assistant helps with literature review, data analysis, and writing. Each
    analyst uses it daily across multiple sessions.

    **Memory architecture:**
    - **Working memory** (8K tokens): current session messages + top retrieved memories.
    - **Episodic memory** (vector DB, unlimited): every past session summarised and stored.
      Retrieved when analyst references past work ("that analysis from last month").
    - **Semantic memory** (vector DB): uploaded papers, company reports, domain knowledge.
      Retrieved on factual questions ("what did that paper say about...").
    - **Procedural memory** (SQL DB, small): analyst preferences and standard workflows
      ("always use APA citation format", "my preferred Python charting style is seaborn").

    **Session lifecycle:**
    1. Session start → inject top-3 episodic memories + user preferences into working memory.
    2. Turn → route query → retrieve relevant memories → augment prompt → LLM response.
    3. Session end → summarise session → store to episodic memory → clear working memory.

    **Metrics:**
    - Memory retrieval accuracy: 80%+ of retrieved memories are rated relevant by analysts.
    - Context window savings: 40% reduction in tokens/session via episodic injection vs full history.
    - User retention: analysts who enabled memory features had 3× higher weekly active usage.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Memory TTL and decay.** Episodic memories older than 90 days should decay in
      retrieval weight (increase λ). Delete after 1 year unless explicitly pinned. Stale
      memories actively harm retrieval quality by competing with fresh ones.
    - **Memory consolidation.** For high-volume users (100+ sessions), run nightly
      consolidation: cluster similar episodic memories → merge into a single summary →
      delete originals. Reduces storage and retrieval noise.
    - **Cross-user isolation.** Every memory store must be namespaced by `user_id`.
      Failure to isolate is a GDPR and trust violation. Use row-level security in the DB
      or separate vector DB collections per user.
    - **Memory update vs. insert.** When a fact changes ("user now prefers verbose answers"),
      don't just insert the new preference — also delete or downweight the old one, or the
      agent retrieves conflicting memories.
    - **Retrieval budget.** Retrieval adds 5–50ms and uses tokens (the retrieved text is
      injected into the prompt). Profile retrieval latency; set k=3–5 (not k=20). Cache
      the most recently retrieved memories for the session.
    - **Privacy by design.** Users should be able to inspect, edit, and delete their
      memories. Build a `/memory` UI. Memory export/portability (GDPR Art. 20).
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Memory type selection:**

    | Memory type | Latency | Cost/token | Persistence | Capacity | When to use |
    |---|---|---|---|---|---|
    | Working (context) | 0ms | High | No | 4K–128K tokens | Current turn context |
    | Episodic (log) | 10–50ms | Low | Yes | Millions | Past interactions |
    | Semantic (KB) | 10–50ms | Low | Yes | Millions | Factual knowledge |
    | Procedural (cache) | 1–10ms | Low | Yes | Thousands | Task templates |

    **In-context vs. out-of-context tradeoff:**
    - More in-context → richer prompt → better response quality → higher per-call cost.
    - Less in-context (more out-of-context) → cheaper per call → retrieval latency added.
    - Sweet spot: 60–70% utilisation of context window; use retrieval for the rest.

    **Memory freshness tradeoff:**
    - Update memory frequently → always accurate → more memory writes, consolidation cost.
    - Update memory lazily → stale data risk → cheaper.
    - Recommendation: update episodic after every session; update semantic on document ingestion.
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What are the four types of agent memory?"* → Working (context window — current
      session, token-limited); Episodic (past events — session logs with timestamps, retrieved
      by recency/similarity); Semantic (factual knowledge — retrieved by embedding similarity,
      this is RAG); Procedural (how-to patterns — few-shot examples, retrieved by task match).
    - *"How does MemGPT / Letta manage context windows?"* → Hierarchical: in-context
      (main memory — limited) and out-of-context (archival memory — unlimited vector DB).
      When context fills, old messages are compressed/summarised and moved to archival.
      Agent can call `memory_search()` to retrieve from archival when needed.

    **Deep-dive questions**
    - *"How do you decide what to store in episodic vs. semantic memory?"* → Episodic =
      events with temporal context ("session 3: user debugged X"). Semantic = timeless
      facts ("NDCG formula is..."). Heuristic: if the memory is tied to "when it happened",
      it's episodic; if it's a standalone fact, it's semantic. In production, auto-classify
      with an LLM classifier or by source (user conversations → episodic; uploaded docs → semantic).
    - *"What is the privacy risk in multi-tenant memory systems?"* → Cross-user leakage
      if namespace isolation fails. User-A's private conversation could be retrieved for User-B
      if they share a vector collection without row-level filtering. Mitigation: namespace by
      user_id + verify at retrieval; encrypt at rest; audit memory accesses.

    **Common mistakes:** putting everything in working memory (blows context budget);
    never evicting (old context pollutes retrieval); not isolating by user (privacy violation);
    not updating stale memories (agent gives outdated advice).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Four memory types.** Name them, describe what each stores, and give one example
       of a user query that triggers each.
    2. **Working memory eviction.** What happens when the context window is full and a new
       message arrives? What should happen to the evicted message?
    3. **Hybrid episodic retrieval.** Write the formula that blends semantic similarity with
       recency. What does the α parameter control? When would you set α close to 0?
    4. **Memory routing.** A user asks "How do I write a SQL JOIN?" Which memory type(s)
       does the router query and why?
    5. **MemGPT hierarchy.** What is in-context memory? What is archival memory? When does
       MemGPT move something from in-context to archival?
    6. **Privacy isolation.** You have 1000 users sharing a single Pinecone index. What
       is the critical configuration that prevents User A's memories from leaking to User B?
    7. **Memory consolidation.** An analyst has 500 episodic memories after 2 years.
       Why does this hurt retrieval quality, and what do you do about it?
    8. **Cost tradeoff.** Working memory costs $0.01 per 1K tokens; retrieval costs $0.002
       per call. You have 3K tokens of relevant history for a 5K-token session. Calculate
       the cost difference between: (a) injecting all 3K tokens vs. (b) retrieving the top-2
       memories (100 tokens each).
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Classify each of these into working / episodic / semantic / procedural memory:
       (a) "The current conversation." (b) "The user prefers bullet points." (c) "Last month we debugged a FAISS issue." (d) "The formula for precision@k."
    2. Working memory has 4K tokens. System prompt uses 200 tokens. Current conversation
       uses 3900 tokens. A new 300-token message arrives. How many tokens must be evicted?

    **Beginner → Intermediate (coding)**
    3. Extend `WorkingMemory` to implement **summarisation-based eviction**: instead of
       discarding the oldest messages, call a mock `summarise(messages)` function and store
       the summary as a single message. Compare total information retained vs. FIFO eviction.
    4. Extend `EpisodicMemory` with **memory consolidation**: if two episodes have cosine
       similarity > 0.85, merge them into a single combined summary. Run after every 10 stored
       episodes.

    **Intermediate (analysis)**
    5. Implement a **memory benchmark**: create 20 episodes (10 relevant to topic A, 10 to
       topic B). For queries about topic A, measure: how many of top-3 retrieved episodes are
       actually topic A (precision@3)? Vary α from 0 to 1 and plot precision vs. α.
    6. Implement multi-user isolation: extend `SemanticMemory` with a `user_id` field. Add
       an assertion that retrieval for user_id="alice" never returns results stored for user_id="bob".

    **Senior (design)**
    7. *System design:* design a memory system for a coding assistant used by 10K engineers.
       Engineers use it daily; the assistant should remember their preferred languages, past
       debugging sessions, and project context. Design: which memory types, storage backend,
       eviction strategy, consolidation cadence, privacy isolation, and memory update triggers.
    8. *Interview:* "Our agent has a 128K context window. Why do we still need episodic and
       semantic memory?" (Expected: cost — 128K tokens/call is expensive at $0.01/1K tokens
       = $1.28/call vs. $0.02 for 2K tokens + retrieval; latency — 128K tokens is slower to
       process; relevance — 128K of irrelevant history hurts performance more than helps.)
    """),

    md(r"""
    ---
    ### Summary
    Agent memory has four types: **working** (in-context, fast, bounded), **episodic**
    (past events, retrieved by recency+similarity), **semantic** (factual knowledge,
    retrieved by embedding similarity — this is RAG), and **procedural** (how-to patterns,
    retrieved by task match). Production architectures like MemGPT/Letta treat these as
    a hierarchy: in-context for immediate needs, out-of-context archival for everything
    else. Key pitfalls: stale memories, cross-user privacy leakage, and working memory
    overflow without archival.

    **Related lesson:** `AGT-04 · Reflection and Self-Correction` — agents that critique their own
    output and improve iteratively: self-critique, Reflexion, self-RAG, and Constitutional AI.
    """),
]

build("07_ai_agents/03_memory_systems.ipynb", cells)
