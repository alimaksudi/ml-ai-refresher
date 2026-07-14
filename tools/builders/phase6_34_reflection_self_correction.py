"""Builder for Notebook 34 — Reflection and Self-Correction."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 34 · Reflection and Self-Correction
    ### Phase 6 — Agentic AI · *ML/AI Senior Mastery Curriculum*

    > Previous notebooks built agents that act but never question their own outputs.
    > **Reflection** closes the loop: the agent evaluates its draft, identifies weaknesses,
    > generates verbal feedback, and produces an improved version. This transforms
    > agents from one-shot generators into iterative refiners — the key to production-quality
    > autonomous outputs.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Self-critique**: the agent critiques its own output against task requirements.
    - **Reflexion** (Shinn et al., 2023): generate → evaluate → reflect → retry loop
      with verbal reinforcement learning (no gradient updates).
    - **Self-RAG**: retrieve only when needed; critique retrieval relevance; critique
      generation groundedness.
    - **Constitutional AI critique pattern**: check output against a fixed list of
      principles; revise until compliant.
    - **Reflexion from scratch**: score function, verbal critic, trajectory logging,
      early-stop on threshold.
    - **Score improvement curves**: visualise quality rising over iterations.
    - **LangGraph self-correction graph**: conditional edges for retry/stop.

    **Why it matters**
    - One-shot generation quality: 65–75% pass rate on complex tasks. With 2–3 reflection
      iterations: 80–90%. Reflection is the cheapest quality improvement available —
      it costs 1–3 extra LLM calls per task versus hours of human review.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Human analogy.** Expert writers, engineers, and scientists revise. A first draft
    is not the final product. The process: write → review → critique → revise is
    millennia old. Reflection gives agents the same loop.

    **Self-consistency (Wang et al., 2022).** Sample multiple CoT solutions, take the
    majority vote. This is breadth-first reflection — diversity via sampling, selection
    by consensus. It doesn't generate verbal feedback; it aggregates.

    **Constitutional AI (Bai et al., 2022 — Anthropic).** The model critiques its own
    response against a written "constitution" (a list of principles: "Is this helpful?
    Is it harmless? Is it honest?"). It revises based on each principle violation.
    Iterative critique + revision with principles is the core CAI loop.

    **Reflexion (Shinn et al., 2023).** The agent stores verbal reflections from past
    trials as episodic memory. On the next trial for the same task, the reflection
    is injected as context: "Last time I failed because X; this time I will Y."
    This is verbal reinforcement learning — improving without gradient updates.

    **Self-RAG (Asai et al., 2023).** The model generates special reflection tokens
    inline: `[Retrieve]` (should I retrieve?), `[Relevant]` (is this doc relevant?),
    `[Supported]` (is my claim supported by the doc?), `[Utility]` (is this answer
    useful?). Reflection on retrieval quality is embedded in generation.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The reflection loop:**
    ```
    Task → [Generator] → Draft → [Critic] → Score + Verbal feedback
               ↑                                         |
               └──── [Revise prompt + feedback] ←────────┘
                          repeat until score ≥ threshold or max_iter
    ```

    **Reflexion memory injection:**
    ```
    Trial 1: Task → Generate → Evaluate → FAIL
             Reflect: "I forgot to handle the edge case where X is empty."
    Trial 2: Task + Reflection → Generate (with prior learning) → Evaluate → PASS
    ```

    **Self-RAG decision tokens:**
    ```
    User: "What is the NDCG formula?"
    Agent: [Retrieve=Yes]
    → Retrieves doc about NDCG
    → [Relevant=Yes] [Supported=Yes]
    → Generates answer grounded in doc
    → [Utility=High]
    ```

    **Constitutional AI:**
    ```
    Constitution: ["Be accurate", "Cite sources", "Avoid jargon"]
    Draft: "The thingy does the stuff really well."
    Critique: Principle "Avoid jargon" → PASS. Principle "Be accurate" → FAIL (vague).
    Revision: "The HNSW index achieves 95% recall at 10ms latency."
    ```

    **Verbal vs. gradient reinforcement:**
    - Gradient RL: update model weights via reward signal (expensive, requires many trials).
    - Verbal RL (Reflexion): update the prompt via natural language feedback (cheap, no weights).
    - Verbal RL works because LLMs can follow natural language instructions at inference time.
    """),

    code(r"""
    import re
    import math
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import defaultdict

    rng_global = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Reflexion as verbal RL

    Standard RL: $\pi^* = \arg\max_\pi \mathbb{E}[\sum_t r_t]$, optimised via gradient.

    Reflexion: instead of updating $\pi$, update the **prompt** $p$:

    $p_{t+1} = p_0 + \text{concat}(\text{verbal\_reflections}(r_1, \ldots, r_t))$

    where $r_t$ is the textual critic output from trial $t$. The policy $\pi$ (frozen LLM)
    improves in-context through accumulated verbal feedback.

    **Why this works**: LLMs are trained to follow instructions. Verbal feedback IS an
    instruction. The LLM can act on "last time I forgot to handle empty arrays" without
    any weight updates.

    ### 4.2 Critic score function

    Let $s: \text{Output} \to [0, 1]$ be a scalar quality score. Decompose:

    $s(o) = \sum_{k} w_k \cdot c_k(o), \quad \sum_k w_k = 1$

    where $c_k$ are criterion functions (correctness, completeness, style, safety).
    Early-stop criterion: $s(o) \ge \tau$ (threshold), e.g. $\tau = 0.85$.

    ### 4.3 Self-RAG decision tokens

    The model generates four binary tokens inline:

    - **Retrieve** $\in \{0, 1\}$: is retrieval needed?
    - **Relevant** $\in \{0, 1\}$: is the retrieved doc relevant?
    - **Supported** $\in \{0, 1\}$: is the claim supported by the doc?
    - **Utility** $\in \{1, 2, 3, 4, 5\}$: how useful is the answer?

    Final answer selection: $o^* = \arg\max_{o \in O} \text{Utility}(o)$
    where $O$ is the set of candidate answers (one per retrieved doc).

    ### 4.4 Constitutional AI critique

    Given principles $P = \{p_1, \ldots, p_n\}$ and draft $d$:

    $\text{violations} = \{p_i : \text{critic}(d, p_i) = \text{FAIL}\}$

    Revision: $d_{t+1} = \text{revise}(d_t, \text{violations}_t)$

    Converges when $\text{violations} = \emptyset$ or max iterations reached.
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a — Self-critique (single-pass)
    """),

    code(r"""
    # 5a. Self-critique: agent evaluates its own output against criteria.

    CODE_REVIEW_CRITERIA = [
        ('correctness',  0.40, 'Does the code solve the stated problem correctly?'),
        ('completeness', 0.25, 'Does it handle edge cases and error conditions?'),
        ('clarity',      0.20, 'Is the code readable and well-structured?'),
        ('efficiency',   0.15, 'Is the approach efficient (no obvious O(n^2) where O(n) works)?'),
    ]

    def simulate_llm_generate(task, context='', iteration=0):
        # Simulate LLM code generation. Quality improves with context/iteration.
        base_quality = 0.50 + 0.08 * iteration
        qualities = {
            'correctness':  min(1.0, base_quality + 0.05 * len(context) / 200),
            'completeness': min(1.0, base_quality - 0.10 + 0.05 * iteration),
            'clarity':      min(1.0, base_quality + 0.10),
            'efficiency':   min(1.0, base_quality - 0.05),
        }
        # Add some noise.
        rng = np.random.default_rng(42 + iteration)
        qualities = {k: float(np.clip(v + rng.normal(0, 0.03), 0, 1)) for k, v in qualities.items()}
        code_draft = f'[Draft v{iteration+1}: task="{task[:30]}...", quality_hint={qualities}]'
        return code_draft, qualities

    def critique(draft, quality_scores, criteria):
        feedback_lines = []
        total_score = 0.0
        for name, weight, description in criteria:
            score = quality_scores.get(name, 0.5)
            total_score += weight * score
            if score < 0.7:
                feedback_lines.append(f'  [{name.upper()} {score:.2f}] NEEDS WORK: {description}')
            else:
                feedback_lines.append(f'  [{name.upper()} {score:.2f}] OK: {description}')
        verbal_feedback = '\n'.join(feedback_lines)
        return round(total_score, 3), verbal_feedback

    task = 'Write a Python function that finds the k most frequent elements in a list.'
    draft, scores = simulate_llm_generate(task, iteration=0)
    total_score, feedback = critique(draft, scores, CODE_REVIEW_CRITERIA)

    print(f'Self-critique demo:')
    print(f'Task: {task}')
    print(f'Draft: {draft[:80]}...')
    print(f'Critique:\n{feedback}')
    print(f'Score: {total_score:.3f}')
    """),

    md(r"""
    ### 5b — Reflexion loop (generate → evaluate → reflect → retry)
    """),

    code(r"""
    # 5b. Reflexion: verbal reinforcement loop with cumulative reflection memory.

    class ReflexionAgent:
        def __init__(self, max_iterations=5, threshold=0.85):
            self.max_iter = max_iterations
            self.threshold = threshold
            self.reflection_memory = []   # Verbal reflections from past trials.

        def generate(self, task, iteration):
            # Inject all past reflections as context.
            context = '\n'.join(self.reflection_memory)
            return simulate_llm_generate(task, context=context, iteration=iteration)

        def evaluate(self, draft, quality_scores):
            return critique(draft, quality_scores, CODE_REVIEW_CRITERIA)

        def reflect(self, score, verbal_feedback, iteration):
            # Simulate LLM generating a verbal reflection.
            weak_criteria = [name for name, weight, _ in CODE_REVIEW_CRITERIA
                             if name in verbal_feedback and 'NEEDS WORK' in verbal_feedback]
            reflection = (
                f'Trial {iteration+1} score={score:.2f}. '
                f'Weaknesses: {", ".join(weak_criteria) if weak_criteria else "none major"}. '
                f'Next iteration: address each NEEDS WORK criterion explicitly.'
            )
            self.reflection_memory.append(reflection)
            return reflection

        def run(self, task, verbose=True):
            trajectory = []
            for i in range(self.max_iter):
                draft, qualities = self.generate(task, iteration=i)
                score, feedback = self.evaluate(draft, qualities)
                trajectory.append({'iteration': i+1, 'score': score, 'feedback': feedback})

                if verbose:
                    print(f'Iteration {i+1}: score={score:.3f}', end='')

                if score >= self.threshold:
                    if verbose:
                        print(f' ✓ PASSED threshold ({self.threshold})')
                    return {'answer': draft, 'score': score, 'trajectory': trajectory, 'passed': True}

                reflection = self.reflect(score, feedback, i)
                if verbose:
                    print(f' → reflecting: "{reflection[:70]}..."')

            final_draft, final_qualities = self.generate(task, iteration=self.max_iter)
            final_score, _ = self.evaluate(final_draft, final_qualities)
            if verbose:
                print(f'Max iterations reached. Final score: {final_score:.3f}')
            return {'answer': final_draft, 'score': final_score, 'trajectory': trajectory, 'passed': False}

    reflexion_agent = ReflexionAgent(max_iterations=5, threshold=0.85)
    print(f'Reflexion loop for: "{task}"\n')
    result = reflexion_agent.run(task, verbose=True)
    print(f'\nReflection memory accumulated:')
    for r in reflexion_agent.reflection_memory:
        print(f'  {r}')
    """),

    md(r"""
    ### 5c — Constitutional AI critique pattern
    """),

    code(r"""
    # 5c. Constitutional AI: critique against explicit principles, revise until compliant.

    CONSTITUTION = [
        ('accuracy',      'The response must be factually accurate and specific, not vague.'),
        ('completeness',  'The response must address all parts of the question.'),
        ('no_jargon',     'Technical terms must be explained when first introduced.'),
        ('cite_reasoning','Claims must be supported with reasoning or evidence.'),
        ('conciseness',   'The response must not include unnecessary filler words.'),
    ]

    def simulate_principle_check(draft, principle_name, iteration):
        # Simulate whether draft violates each principle.
        # Violation probability decreases with each revision iteration.
        rng = np.random.default_rng(abs(hash(principle_name + str(iteration))) % (2**31))
        pass_prob = 0.50 + 0.18 * iteration
        return rng.random() < min(0.95, pass_prob)

    def simulate_revision(draft, violations, iteration):
        # Simulate LLM revising based on violation list.
        fix_notes = '; '.join(f'fixed {v}' for v in violations)
        return f'{draft} [Rev{iteration+1}: {fix_notes}]'

    class ConstitutionalAIAgent:
        def __init__(self, constitution, max_iterations=4):
            self.constitution = constitution
            self.max_iter = max_iterations

        def critique(self, draft, iteration):
            violations = []
            passes = []
            for name, description in self.constitution:
                passed = simulate_principle_check(draft, name, iteration)
                if passed:
                    passes.append(name)
                else:
                    violations.append((name, description))
            return violations, passes

        def run(self, task, initial_draft, verbose=True):
            draft = initial_draft
            trajectory = []

            for i in range(self.max_iter):
                violations, passes = self.critique(draft, iteration=i)
                trajectory.append({
                    'iteration': i+1,
                    'violations': [v[0] for v in violations],
                    'passes': passes,
                    'n_violations': len(violations),
                })
                if verbose:
                    print(f'Iteration {i+1}: {len(violations)} violations {[v[0] for v in violations]}, {len(passes)} passes')

                if not violations:
                    if verbose:
                        print('  All principles satisfied — done.')
                    return {'draft': draft, 'trajectory': trajectory, 'passed': True}

                draft = simulate_revision(draft, [v[0] for v in violations], i)

            if verbose:
                print(f'Max iterations reached. Final violations: {[v[0] for v, _ in self.critique(draft, self.max_iter)]}')
            return {'draft': draft, 'trajectory': trajectory, 'passed': False}

    cai_agent = ConstitutionalAIAgent(CONSTITUTION, max_iterations=4)
    initial_draft = '[Initial draft: some explanation of NDCG]'
    print(f'Constitutional AI critique for: "Explain NDCG"')
    result_cai = cai_agent.run('Explain NDCG', initial_draft, verbose=True)
    """),

    md(r"""
    ### 5d — Self-RAG: retrieve-and-critique
    """),

    code(r"""
    # 5d. Self-RAG: agent decides when to retrieve, then critiques retrieval quality.

    KNOWLEDGE_BASE = [
        {'id': 0, 'text': 'NDCG@k = DCG@k / IDCG@k where DCG@k = sum(rel_i / log2(i+1)).'},
        {'id': 1, 'text': 'BM25 score = sum over terms of IDF * TF_norm, where k1, b are hyperparams.'},
        {'id': 2, 'text': 'Cosine similarity = dot(a, b) / (||a|| * ||b||). Range: -1 to 1.'},
        {'id': 3, 'text': 'HNSW builds a hierarchical navigable small world graph for ANN search.'},
        {'id': 4, 'text': 'RRF: score(d) = sum_r(1 / (k + rank_r(d))) where k=60 is standard.'},
        {'id': 5, 'text': 'Attention: softmax(QK^T / sqrt(d_k)) V, where Q=XW_Q, K=XW_K, V=XW_V.'},
    ]

    def simple_embed(text, dim=16):
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.standard_normal(dim)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def retrieve(query, k=2):
        q_emb = simple_embed(query, 16)
        scored = [(cosine(q_emb, simple_embed(d['text'], 16)), d) for d in KNOWLEDGE_BASE]
        scored.sort(key=lambda x: -x[0])
        return [(d, round(s, 3)) for s, d in scored[:k]]

    def self_rag(query, retrieval_threshold=0.3, verbose=True):
        # Step 1: Should we retrieve?
        # Heuristic: retrieve if query looks factual (contains question words, formula refs).
        should_retrieve = any(kw in query.lower() for kw in
                              ['what is', 'formula', 'how does', 'explain', 'define', 'calculate'])
        if verbose:
            print(f'Query: "{query}"')
            print(f'  [Retrieve={int(should_retrieve)}] retrieve decision')

        if not should_retrieve:
            answer = f'[ANSWER without retrieval to: "{query}"]'
            if verbose:
                print(f'  [Utility=3] Answering without retrieval.')
            return answer, []

        # Step 2: Retrieve.
        docs = retrieve(query, k=2)

        # Step 3: Critique each doc for relevance.
        relevant_docs = []
        for doc, score in docs:
            relevant = score > retrieval_threshold
            if verbose:
                print(f'  [Relevant={int(relevant)}] doc: "{doc["text"][:50]}..." score={score:.3f}')
            if relevant:
                relevant_docs.append(doc)

        # Step 4: Generate with/without grounding.
        if not relevant_docs:
            answer = f'[ANSWER: no relevant docs found for "{query}" — answering from parametric knowledge]'
            if verbose:
                print(f'  [Supported=0] No relevant docs — falling back to parametric.')
        else:
            context = ' '.join(d['text'] for d in relevant_docs)
            answer = f'[GROUNDED ANSWER based on: "{context[:80]}..."]'
            if verbose:
                print(f'  [Supported=1] Generating grounded answer.')

        # Step 5: Rate utility.
        utility = 4 if relevant_docs else 2
        if verbose:
            print(f'  [Utility={utility}] Final answer utility rating.')

        return answer, relevant_docs

    print('=== Self-RAG Demo ===\n')
    self_rag('What is the NDCG formula?', verbose=True)
    print()
    self_rag('Hi, how are you today?', verbose=True)
    print()
    self_rag('Explain how HNSW works', verbose=True)
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Reflexion score improvement over iterations.
    tasks_fig = [
        'Write a Python function to find k most frequent elements',
        'Implement a binary search with edge case handling',
        'Write a SQL query with multiple JOINs and aggregation',
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue', 'seagreen', 'coral']

    for task, color in zip(tasks_fig, colors):
        agent = ReflexionAgent(max_iterations=6, threshold=0.90)
        scores = []
        reflections = []
        for i in range(6):
            context = '\n'.join(agent.reflection_memory)
            draft, quals = simulate_llm_generate(task, context=context, iteration=i)
            score, feedback = agent.evaluate(draft, quals)
            scores.append(score)
            if score < 0.90:
                reflection = agent.reflect(score, feedback, i)
            if score >= 0.90:
                break
        ax.plot(range(1, len(scores)+1), scores, 'o-', color=color, label=task[:40]+'...')

    ax.axhline(0.85, color='red', ls='--', alpha=0.7, label='Threshold (0.85)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Quality score')
    ax.set_title('Figure 1 — Reflexion: quality score improvement over iterations')
    ax.legend(fontsize=8); ax.set_ylim(0.3, 1.05)
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 1.** Quality score trajectories for three code generation tasks across
    Reflexion iterations. Each task starts below the 0.85 threshold and improves with
    each iteration as verbal reflections accumulate in the prompt context. The score
    improvement is not monotonic (noise in generation + evaluation) but the trend is
    upward. The agent stops early when the threshold is reached — not all tasks need
    the full 6 iterations. Key insight: the marginal improvement per iteration decreases
    — most of the gain happens in iteration 1 and 2. Diminishing returns justify
    setting max_iterations=3 in production for cost control.
    """),

    code(r"""
    # Figure 2 — Constitutional AI: violation count declining over iterations.
    tasks_cai = [
        'Explain cosine similarity',
        'Describe the transformer architecture',
        'What is overfitting?',
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors2 = ['steelblue', 'seagreen', 'coral']

    for task, color in zip(tasks_cai, colors2):
        agent_c = ConstitutionalAIAgent(CONSTITUTION, max_iterations=5)
        draft_c = f'[Initial draft for: {task}]'
        traj_data = []
        draft_now = draft_c
        for i in range(5):
            violations, passes = agent_c.critique(draft_now, iteration=i)
            traj_data.append(len(violations))
            if not violations:
                break
            draft_now = simulate_revision(draft_now, [v[0] for v in violations], i)
        ax.plot(range(1, len(traj_data)+1), traj_data, 'o-', color=color, label=task[:30]+'...')

    ax.set_xlabel('Iteration'); ax.set_ylabel('Number of principle violations')
    ax.set_title('Figure 2 — Constitutional AI: violations decrease over iterations')
    ax.legend(fontsize=8); ax.set_ylim(-0.5, 6)
    ax.set_yticks(range(0, 7))
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 2.** Constitutional AI violation count across revision iterations for three
    tasks. Violations start at 2–4 (out of 5 principles) and decline toward 0. The rate
    of decline depends on the task — simple explanations converge in 2 iterations while
    complex technical descriptions may take 4. The algorithm terminates as soon as violations
    hit 0, not at a fixed iteration count. Cost: each iteration is one LLM call; with 5
    principles and 3 iterations on average, CAI costs ~3 LLM calls per task vs. 1 without.
    """),

    code(r"""
    # Figure 3 — Self-RAG: retrieval decision and utility by query type.
    test_queries = [
        ('What is NDCG formula?',        True,  4),
        ('How does BM25 work?',          True,  4),
        ('Hello, how are you?',          False, 2),
        ('What time is it?',             False, 1),
        ('Explain cosine similarity',    True,  4),
        ('Write me a sonnet',            False, 3),
        ('Calculate precision at k=5',   True,  3),
    ]

    retrieve_decisions = [int(rq[1]) for rq in test_queries]
    utility_scores     = [rq[2] for rq in test_queries]
    labels = [rq[0][:25]+'...' for rq in test_queries]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(labels))
    axes[0].bar(x, retrieve_decisions, color=['steelblue' if d else 'coral' for d in retrieve_decisions], alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    axes[0].set_ylabel('Retrieved? (1=Yes, 0=No)')
    axes[0].set_title('Figure 3a — Self-RAG: retrieval decisions by query')
    axes[0].set_ylim(0, 1.5)

    axes[1].bar(x, utility_scores, color='seagreen', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Utility score (1–5)')
    axes[1].set_title('Figure 3b — Self-RAG: utility scores')
    axes[1].set_ylim(0, 6)
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 3.** Self-RAG retrieval decisions (3a) and utility scores (3b) by query type.
    Factual queries ("What is NDCG?", "How does BM25 work?") trigger retrieval (blue bars)
    and receive high utility scores when relevant documents are found. Non-factual queries
    ("Hello, how are you?") skip retrieval and receive low utility scores — the agent
    answers from parametric knowledge which is adequate for social interaction. The key
    insight: **selective retrieval** is more efficient than always-retrieve. Grounding every
    response in a retrieved document wastes retrieval cost and can hurt quality when the
    query doesn't need grounding (small talk, creative tasks).
    """),

    code(r"""
    # Figure 4 — Comparison: no reflection vs. Reflexion vs. Constitutional AI.
    # Simulate quality distributions for 20 code review tasks.
    N = 20
    rng = np.random.default_rng(99)
    no_reflection   = rng.normal(0.62, 0.10, N)
    reflexion_3iter = rng.normal(0.82, 0.07, N)
    cai_4iter       = rng.normal(0.79, 0.06, N)
    # Clip to [0, 1].
    no_reflection   = np.clip(no_reflection,   0, 1)
    reflexion_3iter = np.clip(reflexion_3iter,  0, 1)
    cai_4iter       = np.clip(cai_4iter,        0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0.3, 1.0, 15)
    ax.hist(no_reflection,   bins=bins, alpha=0.6, label=f'No reflection (mean={no_reflection.mean():.2f})',   color='coral')
    ax.hist(reflexion_3iter, bins=bins, alpha=0.6, label=f'Reflexion 3 iters (mean={reflexion_3iter.mean():.2f})', color='steelblue')
    ax.hist(cai_4iter,       bins=bins, alpha=0.6, label=f'Const. AI 4 iters (mean={cai_4iter.mean():.2f})', color='seagreen')
    ax.axvline(0.80, color='red', ls='--', alpha=0.7, label='Quality threshold 0.80')
    ax.set_xlabel('Quality score'); ax.set_ylabel('Count (tasks)')
    ax.set_title('Figure 4 — Quality distribution: no reflection vs. Reflexion vs. Constitutional AI')
    ax.legend(); plt.tight_layout(); plt.show()
    print(f'Pass rate (>0.80): no-reflection={np.mean(no_reflection > 0.80):.0%}, reflexion={np.mean(reflexion_3iter > 0.80):.0%}, CAI={np.mean(cai_4iter > 0.80):.0%}')
    """),

    md(r"""
    **Figure 4.** Quality score distributions for 20 code generation tasks across three
    strategies. **No reflection** (red): mean ~0.62, wide distribution — many tasks fail
    quality threshold. **Reflexion** (blue): mean ~0.82, distribution shifted right with
    tighter variance — most tasks pass. **Constitutional AI** (green): mean ~0.79, narrower
    than no-reflection but slightly lower than Reflexion for code tasks (CAI is better
    suited for content safety/style tasks than code correctness). Pass rate improvement:
    no-reflection ~25% → Reflexion ~80%. Cost: Reflexion adds 2–3 LLM calls per task.
    Cost-quality tradeoff: for tasks with high cost of failure (wrong code in production,
    harmful content), reflection is always worth the extra LLM calls.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Reflection without change** | Agent outputs the same draft each iteration | Verbal feedback not specific enough | Require the critic to identify specific line/criterion; use structured feedback |
    | **Reflection loop** | Agent oscillates between two drafts | Critic inconsistent across calls | Use deterministic (temperature=0) critic; check draft similarity before accepting |
    | **Score inflation** | Critic gives 0.95 to a poor draft | Self-evaluation bias (LLMs are sycophantic) | Use separate critic model; use rubric with specific binary checks |
    | **Over-refining** | Quality plateaus but agent keeps iterating | No early-stop | Stop when |score_t - score_{t-1}| < 0.01 for 2 consecutive iterations |
    | **Self-RAG retrieval on everything** | High latency; irrelevant docs | Retrieve token always fires | Train/tune the retrieve decision; fallback to parametric for conversational queries |
    | **Constitutional AI infinite loop** | Agent cannot satisfy all principles simultaneously | Conflicting principles | Check for principle conflicts at design time; prioritise (ordered principles) |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 LangGraph self-correction graph (guarded import).
    try:
        import langgraph  # noqa: F401
        print('LangGraph available.')
    except ImportError:
        lines = [
            '[langgraph not installed — self-correction graph pattern]:',
            '  from langgraph.graph import StateGraph, END',
            '  from typing import TypedDict',
            '',
            '  class State(TypedDict):',
            '      task: str',
            '      draft: str',
            '      score: float',
            '      reflections: list',
            '      iteration: int',
            '',
            '  def generate_node(state):',
            '      prompt = state["task"] + "\n".join(state["reflections"])',
            '      draft = llm.invoke(prompt)',
            '      return {"draft": draft, "iteration": state["iteration"] + 1}',
            '',
            '  def critique_node(state):',
            '      score = critic_llm.invoke(f"Score 0-1: {state[\'draft\']}")',
            '      return {"score": float(score)}',
            '',
            '  def reflect_node(state):',
            '      feedback = critic_llm.invoke(f"What is wrong with: {state[\'draft\']}")',
            '      return {"reflections": state["reflections"] + [feedback]}',
            '',
            '  def should_continue(state):',
            '      if state["score"] >= 0.85 or state["iteration"] >= 5:',
            '          return "end"',
            '      return "reflect"',
            '',
            '  graph = StateGraph(State)',
            '  graph.add_node("generate", generate_node)',
            '  graph.add_node("critique", critique_node)',
            '  graph.add_node("reflect", reflect_node)',
            '  graph.add_edge("generate", "critique")',
            '  graph.add_conditional_edges("critique", should_continue, {"reflect": "reflect", "end": END})',
            '  graph.add_edge("reflect", "generate")',
            '  app = graph.compile()',
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 OpenAI / Anthropic self-critique pattern (guarded).
    try:
        import anthropic  # noqa: F401
        print('Anthropic SDK available.')
    except ImportError:
        lines = [
            '[anthropic not installed — self-critique prompt pattern]:',
            '  import anthropic',
            '  client = anthropic.Anthropic()',
            '',
            '  def critique_with_claude(draft, task):',
            '      msg = client.messages.create(',
            '          model="claude-opus-4-8",',
            '          max_tokens=500,',
            '          messages=[{',
            '              "role": "user",',
            '              "content": f"Task: {task}\nDraft: {draft}\n\nCritique this draft.",',
            '          }],',
            '      )',
            '      return msg.content[0].text',
            '',
            '  # Reflexion loop:',
            '  reflections = []',
            '  for i in range(3):',
            '      draft = generate_with_claude(task, reflections)',
            '      score, feedback = evaluate(draft)',
            '      if score >= 0.85: break',
            '      reflections.append(feedback)',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Automated Code Review Agent

    **Scenario.** A 50-engineer startup wants to automate first-pass code review. PRs
    take 2–4 hours for human review. They want the agent to: identify bugs, suggest
    improvements, flag security issues, and verify proposed fixes — all before a human
    reviewer sees the PR.

    **Reflexion-based agent architecture:**
    - **Generator**: review the PR diff → produce review comments (bugs, improvements, security).
    - **Critic**: evaluate the review against criteria: (1) Are bugs specific (file + line)?
      (2) Are suggestions actionable? (3) Are security issues correctly classified?
    - **Reflexion loop**: if score < 0.85, reflect on what's missing → re-review with context.
    - **Verify fixes**: after developer applies fixes, agent re-reviews the updated diff.

    **Results at a 200-PR/month scale:**
    - Agent catch rate: 78% of bugs identified in human review also caught by agent.
    - False positive rate: 12% of agent-flagged issues are not real bugs (acceptable).
    - Human review time: reduced from 3h average to 1.5h (agent handles obvious issues).
    - Cost: ~$0.30/PR in LLM calls (3 reflection iterations × 2 calls = 6 calls × $0.05).
    - Monthly LLM cost: $60 (vs. ~$4,500 equivalent engineer time for first-pass review).

    **Constitutional principles for code review:**
    1. Never suggest changes to working, tested code without strong justification.
    2. Flag security vulnerabilities (SQL injection, XSS, secrets in code) with HIGH priority.
    3. Classify all comments: bug / improvement / style / security.
    4. Include the file path and line number for every comment.
    5. Rate each comment: critical / major / minor.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Separate generator and critic models.** Using the same model for generation
      and critique introduces sycophancy — the model praises its own output. Use a
      different model (or at minimum different temperature: generator=0.7, critic=0).
    - **Structured critic output.** Free-text critique is hard to parse reliably.
      Use JSON-schema-constrained output for critic: `{score: float, issues: [{criterion, severity, suggestion}]}`.
    - **Cost control.** Each reflection iteration costs 1–2 extra LLM calls. At $0.05/call,
      3 iterations costs $0.15 extra per task. For high-volume tasks (1M/day), set
      max_iterations=1 with threshold=0.75 (lower bar, fewer retries).
    - **Iteration convergence monitoring.** Log score per iteration. Alert if mean
      improvement per iteration falls below 0.02 — the critic or generator needs tuning.
    - **Human-in-the-loop for failed tasks.** If score < threshold after max_iterations,
      route to human review. Never silently accept a failed generation in production.
    - **Reflection memory hygiene.** In Reflexion, don't accumulate unbounded reflections
      across sessions — old reflections about a different context can confuse the model.
      Limit to the last 3 reflections per task type.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Reflection strategy comparison:**

    | Strategy | Mechanism | LLM calls | Best for | Limitations |
    |---|---|---|---|---|
    | Self-critique (1 pass) | Generate → evaluate | 2 | Quick quality gate | No improvement, just flagging |
    | Reflexion | Gen → eval → reflect → retry | 2N+1 | Code, reasoning tasks | N iterations × cost |
    | Constitutional AI | Gen → check principles → revise | 1+N_violations | Content safety, style | Conflicts between principles |
    | Self-RAG | Gen + conditional retrieve + groundedness check | 1–4 | Factual QA | Requires retrieval infrastructure |
    | Self-consistency | Sample N → majority vote | N | Classification, math | No verbal feedback; sampling cost |

    **When each excels:**
    - Use **Reflexion** when: the task has clear success criteria; the generator can improve with verbal feedback.
    - Use **CAI** when: output must satisfy multiple content principles (safety, tone, format).
    - Use **Self-RAG** when: some queries need grounding and some don't; you want selective retrieval.
    - Use **self-consistency** when: the task is classification/math and you want calibrated confidence.
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What is Reflexion? How is it different from reinforcement learning?"* → Reflexion stores
      verbal critiques from failed trials as episodic memory and injects them into the next trial's
      prompt. Standard RL updates model weights via gradient descent; Reflexion updates the prompt
      via natural language. No weights are changed — it's in-context learning with verbal feedback.
      The LLM is frozen; only the prompt evolves.
    - *"What is the Self-RAG approach?"* → Self-RAG generates special inline tokens that control
      retrieval: `[Retrieve]` (should I retrieve?), `[Relevant]` (is this doc on-topic?),
      `[Supported]` (is my claim grounded?), `[Utility]` (how useful?). These allow the model
      to selectively retrieve only when needed and to self-assess grounding quality — reducing
      unnecessary retrievals and improving factual accuracy.

    **Deep-dive questions**
    - *"Why does using the same model as generator and critic introduce bias?"* → LLMs are trained
      on human feedback that tends to reward coherent, confident-sounding text. The model optimises
      for its own distributional preferences. When asked to critique its own output, it applies the
      same preferences — it tends to find its own outputs acceptable (sycophancy). Mitigation:
      use a different model or explicitly prompt the critic with adversarial framing ("find all flaws").
    - *"How do you prevent Reflexion from oscillating?"* → Oscillation (A→B→A→B) happens when the
      critic is inconsistent. Fix: (a) use temperature=0 for the critic; (b) track draft similarity
      (cosine) — if new draft ≈ old draft, force a different revision strategy; (c) add convergence
      check: stop if |score_t - score_{t-1}| < epsilon for 2 iterations.

    **Common mistakes:** using temperature > 0 for the critic (inconsistent scores); not separating
    generator and critic models (sycophancy); unlimited iterations (cost explosion); accepting a
    failed generation instead of routing to human review.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Reflexion loop.** What are the 4 steps? What does "verbal reinforcement" mean vs.
       gradient-based reinforcement learning?
    2. **Constitutional AI.** What is a "constitution"? Walk through a single CAI iteration
       for the principle "the response must include a code example."
    3. **Self-RAG tokens.** Name the 4 Self-RAG tokens. For the query "What is the capital
       of France?" — which tokens fire and why?
    4. **Self-critique bias.** Why does using the same model as generator and critic introduce
       bias? Give one mitigation.
    5. **Convergence criterion.** How do you know when to stop the Reflexion loop? Give 2
       stopping conditions.
    6. **Score function decomposition.** The critic checks correctness (40%), completeness (25%),
       clarity (20%), efficiency (15%). A draft scores 0.9, 0.6, 0.8, 0.7. What is the total
       weighted score?
    7. **Cost analysis.** Reflexion adds 2 LLM calls per iteration. If each call costs $0.02
       and you run 3 iterations on 500 tasks/day, what is the daily extra cost?
    8. **CAI conflict.** You have two CAI principles: "Be concise" and "Provide complete context."
       These conflict for complex technical answers. How do you resolve it?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. What is verbal reinforcement learning? How does it differ from RLHF?
    2. Score calculation: weights = [0.4, 0.3, 0.2, 0.1], scores = [0.9, 0.7, 0.8, 0.6].
       Calculate the weighted total. Does this pass the 0.80 threshold?

    **Beginner → Intermediate (coding)**
    3. Add **early stopping** to `ReflexionAgent`: stop if the score improvement between
       iterations is less than 0.01. Measure how many iterations are saved across 10 tasks
       versus always running max_iterations.
    4. Implement a **structured critic** that returns JSON: `{"score": float, "issues":
       [{"criterion": str, "severity": "critical|major|minor", "suggestion": str}]}`.
       Verify the JSON is valid before using it (catch parse errors).

    **Intermediate (analysis)**
    5. **Sycophancy test**: generate 10 drafts of varying quality. Have the agent score
       each. Then have a "strict adversarial critic" score each ("find everything wrong").
       Compare score distributions. Is the standard critic inflated vs. the adversarial one?
    6. **Self-RAG calibration**: test 20 queries (10 factual, 10 conversational). Measure
       retrieval precision (of the retrieved queries, how many genuinely needed retrieval?)
       and retrieval recall (of the factual queries, how many triggered retrieval?). Tune
       the threshold to maximise F1.

    **Senior (design)**
    7. *System design:* design a Reflexion-based essay writing assistant. Users submit an
       essay topic; the agent writes a draft, critiques it against an academic rubric (thesis,
       evidence, structure, style), reflects, and revises up to 3 times. Design: critic prompt,
       score function, reflection memory structure, human review trigger, cost budget.
    8. *Interview:* "We have a Constitutional AI agent for content moderation with 20 principles.
       Average violation count per response is 3. Max iterations = 3. At 100K requests/day and
       $0.01/LLM call, what is the daily CAI cost? How would you reduce it by 50% without
       dropping below 2 iterations average?"
       (Expected: daily cost = 100K × (1 gen + 3 iter) × 1 call = 400K calls × $0.01 = $4K.
       Reduce by batching critique of all principles in one call per iteration: 100K × (1+2) = $3K.)
    """),

    md(r"""
    ---
    ### Summary
    Reflection gives agents the same iterative improvement loop that expert humans use: draft →
    critique → revise → repeat. **Reflexion** stores verbal critiques as episodic memory, improving
    future trials without gradient updates. **Constitutional AI** checks output against explicit
    principles and revises until compliant. **Self-RAG** embeds retrieval and grounding checks
    inline. **Self-critique** is the simplest pattern — one extra call for evaluation and targeted
    feedback. In production: separate generator and critic models, use structured critic output,
    set iteration budgets, and route failed generations to humans.

    **Next:** `35 · Multi-Agent Systems` — orchestrating multiple specialised agents to divide
    and conquer complex tasks: hierarchical orchestration, debate, competition, and validation agents.
    """),
]

build("phase6_agents/34_reflection_self_correction.ipynb", cells)
