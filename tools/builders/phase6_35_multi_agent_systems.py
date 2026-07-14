"""Builder for Notebook 35 — Multi-Agent Systems."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 35 · Multi-Agent Systems
    ### Phase 6 — Agentic AI · *ML/AI Senior Mastery Curriculum*

    > A single agent has one context window, one reasoning chain, one knowledge base.
    > Complex tasks — writing a research report, auditing a codebase, designing a system —
    > benefit from **specialisation and parallelism**: multiple agents with distinct roles
    > working in coordination. This notebook builds three multi-agent architectures from
    > scratch and teaches when each is appropriate.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Why multi-agent?** When one agent is not enough: context limits, specialisation,
      parallelism, error correction through independent validation.
    - **Hierarchical architecture**: orchestrator decomposes task → delegates to specialist
      subagents → aggregates results.
    - **Debate architecture**: multiple agents argue opposing positions → judge synthesises.
    - **Validation pipeline**: proposer agent → validator agent → convergence via agreement.
    - **Communication protocol**: structured message passing between agents.
    - **Coordination failures**: agent miscommunication, context leakage, deadlock, race conditions.
    - **LangGraph supervisor pattern** (guarded import).
    - **Business case**: research report generation (researcher + analyst + writer agents).

    **Why it matters**
    - Multi-agent systems are how production AI handles tasks that exceed a single LLM's
      capacity or require independent verification. Understanding the three architectures
      lets you choose the right one: hierarchical for pipelines, debate for decisions,
      validation for quality assurance.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Game theory and multi-agent RL.** Multi-agent research predates LLMs — AlphaGo's
    self-play (Silver et al., 2016) used two agents (current and opponent) in a competitive
    architecture. Multi-agent RL showed that agents learn faster and achieve higher quality
    through competition and cooperation than solo training.

    **Society of Mind (Minsky, 1986).** Intelligence emerges from interaction of many
    simple agents, each handling a different aspect of a problem. Large language models
    can now instantiate Minsky's vision in software.

    **LLM-based multi-agent papers (2023–2024).** ChatDev (Qian et al., 2023): software
    development with role-playing LLM agents (PM, engineer, tester). MetaGPT (Hong et al.,
    2023): structured workflows with SOPs. AutoGen (Wu et al., 2023): conversational
    multi-agent framework. AgentScope (Gao et al., 2024): distributed multi-agent
    execution. All follow the same insight: **role specialisation + communication protocol
    → better outcomes than one generalist agent**.

    **Debate for alignment (Irving et al., 2018).** Debate as a safety technique: two AI
    agents argue; a human judge picks the winner. The honest agent should always win if
    the judge can verify individual claims. Applied to quality: two agents with opposing
    views → synthesised answer is more balanced and thorough.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Three architectures at a glance:**
    ```
    Hierarchical:           Debate:                  Validation:
    [Orchestrator]          [Agent A] ← position     [Proposer] → draft
        ├── [Researcher]    [Agent B] ← position     [Validator] → critique
        ├── [Analyst]       [Judge]   → synthesis     [Proposer] → revised
        └── [Writer]                                 ... until agreement
    ```

    **When to use each:**

    | Architecture | Best for | Failure risk |
    |---|---|---|
    | Hierarchical | Long pipelines with distinct stages | Bottleneck at orchestrator; error propagation |
    | Debate | Decisions with tradeoffs; requires balanced views | Judge bias; agents agreeing too easily |
    | Validation | Quality-critical outputs; catch errors | Validator too strict (blocks progress) or too lenient |

    **Communication protocol — the key design choice:**
    - Free-text (fragile: misparse): `"Researcher found 3 papers. Analyst says revenue grew."`
    - Structured JSON (robust): `{"sender": "researcher", "findings": [...], "confidence": 0.9}`
    - Structured JSON is always preferred in production: enables type validation, logging,
      tracing, and replay.

    **Parallelism advantage:**
    - Sequential (single agent): $T = T_1 + T_2 + T_3$
    - Parallel (multi-agent): $T = \max(T_1, T_2, T_3)$ — wall-clock time is the slowest agent.
    - For N=3 equal-time tasks: parallelism gives 3× speedup.
    """),

    code(r"""
    import json
    import math
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import defaultdict

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Task decomposition and dependency ordering

    Given task $T$, decompose into subtasks $\{t_1, \ldots, t_n\}$ with dependency DAG.
    Assign each $t_i$ to specialist agent $a_j$. Execution time:

    $T_{\text{parallel}} = \sum_{l \in \text{levels}} \max_{t_i \in l} \text{time}(t_i)$

    where levels are topological levels of the DAG (see NB32).

    ### 4.2 Debate aggregation

    Each agent $a_i$ produces claim $c_i$ with confidence $\alpha_i \in [0,1]$.
    Judge aggregates by weighted consensus:

    $c_{\text{final}} = \arg\max_{c} \sum_{i: c_i = c} \alpha_i$

    For continuous scores: $s_{\text{final}} = \sum_i w_i s_i$, $\sum w_i = 1$.

    ### 4.3 Convergence in validation

    Define agreement between proposer output $p$ and validator approval $v \in \{0,1\}$.
    Convergence criterion: $v = 1$ (validator approves) or $k > K_{\max}$ (max rounds).

    Quality as function of rounds: $q(k) = 1 - (1 - q_0) \cdot e^{-\lambda k}$

    where $q_0$ is initial quality and $\lambda$ is improvement rate per round.

    ### 4.4 Communication overhead

    Each agent-to-agent message has overhead $c_{\text{msg}}$ (serialisation + LLM context
    injection). Total coordination cost: $C = n_{\text{messages}} \cdot c_{\text{msg}}$.
    For hierarchical with N workers: $n_{\text{messages}} = 2N$ (N task dispatches + N results).
    For debate: $n_{\text{messages}} = 2R \cdot N_A + 1$ (R rounds × N agents + final judge call).
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a — Message protocol and base agent
    """),

    code(r"""
    # 5a. Structured message protocol and base agent class.

    class Message:
        def __init__(self, sender, recipient, content, msg_type='task', metadata=None):
            self.sender = sender
            self.recipient = recipient
            self.content = content
            self.msg_type = msg_type   # task | result | error | debate | critique
            self.metadata = metadata or {}

        def to_dict(self):
            return {
                'sender': self.sender,
                'recipient': self.recipient,
                'content': self.content,
                'type': self.msg_type,
                'metadata': self.metadata,
            }

        def __repr__(self):
            return f'Message({self.sender} → {self.recipient}, type={self.msg_type}, content={str(self.content)[:50]})'

    class BaseAgent:
        def __init__(self, name, role):
            self.name = name
            self.role = role
            self.inbox = []
            self.log = []

        def receive(self, message):
            self.inbox.append(message)
            self.log.append({'event': 'received', 'msg': message.to_dict()})

        def send(self, recipient_name, content, msg_type='result', metadata=None):
            msg = Message(self.name, recipient_name, content, msg_type, metadata)
            self.log.append({'event': 'sent', 'msg': msg.to_dict()})
            return msg

        def process(self, message):
            raise NotImplementedError

    print('Message protocol ready.')

    # Demo.
    m = Message('researcher', 'analyst', {'papers': ['Paper A', 'Paper B'], 'summary': 'Market is growing 15% YoY'})
    print('Example message:', m)
    print('As dict:', json.dumps(m.to_dict(), indent=2))
    """),

    md(r"""
    ### 5b — Hierarchical architecture (orchestrator + specialist subagents)
    """),

    code(r"""
    # 5b. Hierarchical multi-agent system: orchestrator delegates to specialists.

    class ResearcherAgent(BaseAgent):
        def __init__(self):
            super().__init__('researcher', 'Finds relevant facts and data')

        def process(self, task_msg):
            topic = task_msg.content.get('topic', 'AI market')
            # Simulated research (in prod: retrieval tool + web search + summarise).
            findings = {
                'topic': topic,
                'facts': [
                    f'Global {topic} market size: $150B in 2024.',
                    f'{topic} adoption grew 35% YoY according to Gartner 2024.',
                    f'Top players: OpenAI (40% share), Google (25%), Anthropic (15%).',
                ],
                'sources': ['Gartner 2024', 'IDC Report', 'Forbes AI Index'],
                'confidence': 0.85,
            }
            return self.send('analyst', findings, msg_type='result')

    class AnalystAgent(BaseAgent):
        def __init__(self):
            super().__init__('analyst', 'Analyses data and draws insights')

        def process(self, data_msg):
            data = data_msg.content
            facts = data.get('facts', [])
            # Simulated analysis.
            analysis = {
                'key_insights': [
                    f'Market is large ($150B) and growing fast (35% YoY) — high opportunity.',
                    f'Market is concentrated (top 3 = 80% share) — high entry barrier.',
                    f'Recommendation: focus on a niche vertical rather than competing broadly.',
                ],
                'risk_level': 'medium',
                'opportunity_score': 0.72,
                'n_facts_used': len(facts),
            }
            return self.send('writer', analysis, msg_type='result')

    class WriterAgent(BaseAgent):
        def __init__(self):
            super().__init__('writer', 'Writes structured reports')

        def process(self, analysis_msg):
            analysis = analysis_msg.content
            insights = analysis.get('key_insights', [])
            # Simulated report writing.
            report = {
                'title': 'AI Market Opportunity Analysis',
                'executive_summary': f'The AI market presents a {analysis["risk_level"]}-risk opportunity (score: {analysis["opportunity_score"]:.2f}).',
                'key_findings': insights,
                'word_count': sum(len(i.split()) for i in insights) + 30,
                'sections': ['Executive Summary', 'Market Size', 'Key Players', 'Risks', 'Recommendation'],
            }
            return self.send('orchestrator', report, msg_type='result')

    class OrchestratorAgent(BaseAgent):
        def __init__(self):
            super().__init__('orchestrator', 'Decomposes tasks and coordinates subagents')
            self.researcher = ResearcherAgent()
            self.analyst = AnalystAgent()
            self.writer = WriterAgent()
            self.message_log = []

        def run(self, task, verbose=True):
            if verbose:
                print(f'Orchestrator: starting task "{task}"')

            # Step 1: dispatch to researcher.
            task_msg = Message('orchestrator', 'researcher', {'topic': task, 'depth': 'detailed'}, msg_type='task')
            self.message_log.append(task_msg)
            if verbose:
                print(f'  → [researcher] Research: {task}')
            research_result = self.researcher.process(task_msg)
            self.message_log.append(research_result)

            # Step 2: analyst processes research output.
            if verbose:
                print(f'  → [analyst] Analyse {len(research_result.content["facts"])} facts')
            analysis_result = self.analyst.process(research_result)
            self.message_log.append(analysis_result)

            # Step 3: writer generates report.
            if verbose:
                print(f'  → [writer] Write report (opportunity={analysis_result.content["opportunity_score"]:.2f})')
            report_result = self.writer.process(analysis_result)
            self.message_log.append(report_result)

            return report_result.content

    orchestrator = OrchestratorAgent()
    print('\n=== Hierarchical Multi-Agent Demo ===')
    report = orchestrator.run('AI and Machine Learning', verbose=True)
    print(f'\nFinal report:')
    print(f'  Title: {report["title"]}')
    print(f'  Summary: {report["executive_summary"]}')
    print(f'  Sections: {report["sections"]}')
    print(f'  Word count: {report["word_count"]}')
    print(f'  Total messages exchanged: {len(orchestrator.message_log)}')
    """),

    md(r"""
    ### 5c — Debate architecture
    """),

    code(r"""
    # 5c. Debate architecture: two agents argue; a judge synthesises.

    class DebaterAgent(BaseAgent):
        def __init__(self, name, position):
            super().__init__(name, f'Argues for: {position}')
            self.position = position

        def argue(self, topic, round_num, opponent_last=None):
            # Simulated argument generation.
            arguments_pro = {
                'Use microservices': [
                    'Independent scaling per service reduces infrastructure cost by 40%.',
                    'Team autonomy: each team deploys independently.',
                    'Fault isolation: a crash in service A does not bring down service B.',
                ],
                'Use monolith': [
                    'Simpler development: one codebase, one deploy pipeline.',
                    'Lower latency: in-process calls instead of network round-trips.',
                    'Easier debugging: single stack trace, no distributed tracing setup.',
                ],
            }
            args = arguments_pro.get(self.position, [f'My position: {self.position} is correct.'])
            arg_idx = min(round_num, len(args) - 1)
            argument = args[arg_idx]

            # Counter-argue if opponent said something.
            counter = ''
            if opponent_last and round_num > 0:
                counter = f' [Counter to opponent: their claim ignores the {round_num}-team scale.]'

            return {
                'position': self.position,
                'argument': argument + counter,
                'confidence': 0.70 + 0.05 * round_num,
                'round': round_num,
            }

    class JudgeAgent(BaseAgent):
        def __init__(self):
            super().__init__('judge', 'Synthesises debate into a recommendation')

        def synthesise(self, topic, debate_history):
            # Score each debater by argument count and confidence.
            positions = {}
            for entry in debate_history:
                pos = entry['position']
                if pos not in positions:
                    positions[pos] = {'total_confidence': 0, 'rounds': 0, 'arguments': []}
                positions[pos]['total_confidence'] += entry['confidence']
                positions[pos]['rounds'] += 1
                positions[pos]['arguments'].append(entry['argument'])

            # Synthesis: pick leading position, acknowledge tradeoffs.
            winner = max(positions, key=lambda p: positions[p]['total_confidence'])
            loser = [p for p in positions if p != winner][0] if len(positions) > 1 else None

            synthesis = {
                'recommendation': winner,
                'reasoning': f'After {sum(p["rounds"] for p in positions.values())} rounds of debate, '
                             f'"{winner}" received higher aggregate confidence scores.',
                'caveats': [f'Consider "{loser}" if scale remains below 5 teams.'] if loser else [],
                'confidence': 0.80,
            }
            return synthesis

    class DebateOrchestrator:
        def __init__(self, n_rounds=3):
            self.n_rounds = n_rounds

        def run(self, topic, position_a, position_b, verbose=True):
            agent_a = DebaterAgent('agent_a', position_a)
            agent_b = DebaterAgent('agent_b', position_b)
            judge = JudgeAgent()

            debate_history = []
            last_a, last_b = None, None

            if verbose:
                print(f'Debate topic: "{topic}"')
                print(f'  Agent A: "{position_a}"')
                print(f'  Agent B: "{position_b}"\n')

            for r in range(self.n_rounds):
                arg_a = agent_a.argue(topic, r, last_b)
                arg_b = agent_b.argue(topic, r, last_a)
                debate_history.extend([arg_a, arg_b])
                last_a, last_b = arg_a, arg_b
                if verbose:
                    print(f'Round {r+1}:')
                    print(f'  A [{arg_a["confidence"]:.2f}]: {arg_a["argument"][:70]}...')
                    print(f'  B [{arg_b["confidence"]:.2f}]: {arg_b["argument"][:70]}...')

            synthesis = judge.synthesise(topic, debate_history)
            if verbose:
                print(f'\nJudge synthesis:')
                print(f'  Recommendation: {synthesis["recommendation"]}')
                print(f'  Reasoning: {synthesis["reasoning"]}')
                if synthesis['caveats']:
                    print(f'  Caveats: {synthesis["caveats"]}')

            return synthesis, debate_history

    debate = DebateOrchestrator(n_rounds=3)
    print('=== Debate Architecture Demo ===\n')
    synthesis, history = debate.run(
        topic='Architecture for a 3-team SaaS product',
        position_a='Use microservices',
        position_b='Use monolith',
        verbose=True,
    )
    """),

    md(r"""
    ### 5d — Validation pipeline (proposer + validator)
    """),

    code(r"""
    # 5d. Validation pipeline: proposer generates code; validator critiques; iterate.

    class CodeProposerAgent(BaseAgent):
        def __init__(self):
            super().__init__('proposer', 'Generates code solutions')

        def propose(self, task, feedback=None, iteration=0):
            # Simulated code generation that improves with feedback.
            base_quality = 0.55 + 0.12 * iteration
            rng_p = np.random.default_rng(42 + iteration)
            quality = float(np.clip(base_quality + rng_p.normal(0, 0.04), 0, 1))

            issues_remaining = max(0, 4 - iteration * 2)
            code_draft = (
                f'def find_k_frequent(arr, k):\n'
                f'    # Draft v{iteration+1}: quality_estimate={quality:.2f}\n'
                f'    counts = {{}}\n'
                f'    for x in arr: counts[x] = counts.get(x, 0) + 1\n'
                f'    return sorted(counts, key=counts.get, reverse=True)[:k]'
            )
            if feedback:
                code_draft += f'\n    # Applied feedback: {feedback[:50]}'

            return {
                'code': code_draft,
                'quality_estimate': quality,
                'issues_remaining': issues_remaining,
                'iteration': iteration,
            }

    class CodeValidatorAgent(BaseAgent):
        def __init__(self, strictness=0.80):
            super().__init__('validator', 'Reviews and validates code proposals')
            self.strictness = strictness

        def validate(self, proposal):
            quality = proposal['quality_estimate']
            issues = proposal['issues_remaining']
            passed = quality >= self.strictness and issues == 0

            critique = []
            if quality < self.strictness:
                critique.append(f'Quality {quality:.2f} below threshold {self.strictness:.2f}.')
            if issues > 0:
                critique.append(f'{issues} issue(s) found: missing edge case handling, no type hints, no docstring.')
            if not critique:
                critique.append('Code meets all quality criteria.')

            return {
                'approved': passed,
                'quality': quality,
                'critique': ' '.join(critique),
                'feedback_for_proposer': '; '.join(critique) if not passed else '',
            }

    class ValidationPipeline:
        def __init__(self, max_rounds=5):
            self.proposer = CodeProposerAgent()
            self.validator = CodeValidatorAgent(strictness=0.80)
            self.max_rounds = max_rounds

        def run(self, task, verbose=True):
            feedback = None
            trajectory = []

            if verbose:
                print(f'Validation pipeline for: "{task}"')

            for r in range(self.max_rounds):
                proposal = self.proposer.propose(task, feedback=feedback, iteration=r)
                validation = self.validator.validate(proposal)
                trajectory.append({'round': r+1, 'quality': proposal['quality_estimate'],
                                   'approved': validation['approved']})

                if verbose:
                    status = 'APPROVED' if validation['approved'] else 'REJECTED'
                    print(f'  Round {r+1}: quality={proposal["quality_estimate"]:.2f} → {status}')
                    if not validation['approved']:
                        print(f'    Feedback: {validation["critique"][:70]}')

                if validation['approved']:
                    if verbose:
                        print(f'  Converged in {r+1} round(s).')
                    return {'code': proposal['code'], 'rounds': r+1,
                            'final_quality': proposal['quality_estimate'], 'trajectory': trajectory}

                feedback = validation['feedback_for_proposer']

            if verbose:
                print(f'  Max rounds reached without approval.')
            return {'code': proposal['code'], 'rounds': self.max_rounds,
                    'final_quality': proposal['quality_estimate'], 'trajectory': trajectory}

    pipeline = ValidationPipeline(max_rounds=5)
    print('=== Validation Pipeline Demo ===\n')
    result = pipeline.run('Write a function to find k most frequent elements', verbose=True)
    print(f'\nFinal code (quality={result["final_quality"]:.2f}, {result["rounds"]} rounds):')
    print(result['code'])
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Hierarchical vs sequential timing.
    stages = ['Research', 'Analysis', 'Writing']
    seq_times  = [2.5, 1.8, 3.2]   # simulated seconds
    par_times  = [max(seq_times)] * 3   # parallel = max

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(stages))
    w = 0.35
    ax.bar(x - w/2, seq_times, w, label='Sequential (single agent)', color='coral', alpha=0.8)
    ax.bar(x + w/2, par_times, w, label='Parallel (multi-agent, wall-clock)', color='steelblue', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel('Time (s)'); ax.set_title('Figure 1 — Sequential vs. parallel multi-agent timing')
    ax.axhline(sum(seq_times), color='red', ls='--', alpha=0.6,
               label=f'Sequential total: {sum(seq_times):.1f}s')
    ax.axhline(max(seq_times), color='green', ls='--', alpha=0.6,
               label=f'Parallel total: {max(seq_times):.1f}s')
    ax.legend(fontsize=9); ax.set_ylim(0, 10)
    plt.tight_layout(); plt.show()
    print(f'Speedup: {sum(seq_times)/max(seq_times):.1f}x')
    """),

    md(r"""
    **Figure 1.** Sequential (single agent) vs. parallel (multi-agent) timing for the
    three-stage research pipeline. Sequential execution time is the **sum** of all stage
    times (2.5 + 1.8 + 3.2 = 7.5s). Parallel execution time is the **maximum** of all
    stage times (3.2s) — a 2.3× speedup. In practice, parallelism requires independent
    stages (no data dependency between them). Stages with dependencies (analyst needs
    researcher output) cannot be parallelised — they form the critical path. Design
    multi-agent systems to minimise the critical path, not just the total work.
    """),

    code(r"""
    # Figure 2 — Validation pipeline quality progression.
    fig, ax = plt.subplots(figsize=(10, 4))
    result_traj = result['trajectory']
    rounds_v = [t['round'] for t in result_traj]
    qualities = [t['quality'] for t in result_traj]
    approved_rounds = [t['round'] for t in result_traj if t['approved']]

    ax.plot(rounds_v, qualities, 'o-', color='steelblue', lw=2, label='Quality per round')
    ax.axhline(0.80, color='red', ls='--', alpha=0.7, label='Validator threshold (0.80)')
    if approved_rounds:
        ax.axvline(approved_rounds[0], color='seagreen', ls='--', alpha=0.7,
                   label=f'Approved at round {approved_rounds[0]}')
    ax.set_xlabel('Round'); ax.set_ylabel('Quality score')
    ax.set_title('Figure 2 — Validation pipeline: quality improvement over rounds')
    ax.legend(); ax.set_ylim(0.3, 1.05)
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 2.** Code quality progression across validation rounds. The proposer starts
    below the validator's 0.80 threshold and improves iteratively as it incorporates the
    validator's feedback. The rate of improvement is steepest in round 1 (feedback most
    specific) and flattens as the draft approaches the threshold. In production: set the
    threshold based on task risk. For code that runs in production: threshold = 0.90.
    For internal drafts: 0.70. For creative content: 0.65. A high threshold with many
    rounds = high cost; a low threshold with few rounds = lower quality ceiling.
    """),

    code(r"""
    # Figure 3 — Debate confidence scores across rounds.
    debate2 = DebateOrchestrator(n_rounds=4)
    _, history2 = debate2.run(
        topic='LLM model serving: GPU cluster vs. serverless',
        position_a='GPU cluster (dedicated)',
        position_b='Serverless inference',
        verbose=False,
    )

    pos_a = [e for e in history2 if e['position'] == 'GPU cluster (dedicated)']
    pos_b = [e for e in history2 if e['position'] == 'Serverless inference']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([e['round']+1 for e in pos_a], [e['confidence'] for e in pos_a],
            'o-', color='steelblue', label='GPU cluster', lw=2)
    ax.plot([e['round']+1 for e in pos_b], [e['confidence'] for e in pos_b],
            's-', color='coral', label='Serverless', lw=2)
    ax.set_xlabel('Round'); ax.set_ylabel('Argument confidence')
    ax.set_title('Figure 3 — Debate: confidence trajectories across rounds')
    ax.legend(); ax.set_ylim(0.5, 1.0)
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 3.** Debate confidence trajectories. Both agents' confidence grows with each
    round as they develop stronger arguments and counter-arguments. The agent with higher
    final confidence is recommended by the judge (with caveats). In production, confidence
    is generated by the LLM as a calibrated probability ("I am 85% confident that...").
    Debate quality depends critically on the judge's ability to evaluate claims independently
    of which agent made them. A good judge should ask: "Can I verify this claim without
    trusting the agent?" Verifiable claims (cites data, testable) beat assertion-only claims.
    """),

    code(r"""
    # Figure 4 — Communication overhead: messages vs. agents.
    n_agents = [2, 3, 5, 8, 10]
    # Hierarchical: orchestrator + N workers = 2N messages (dispatch + result each).
    hier_msgs = [2 * n for n in n_agents]
    # Debate with 3 rounds: 2 * rounds * agents + 1 judge call.
    debate_msgs = [2 * 3 * n + 1 for n in n_agents]
    # Fully connected (all-to-all): each agent sends to every other = N*(N-1).
    full_msgs = [n * (n-1) for n in n_agents]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(n_agents, hier_msgs,   'o-', label='Hierarchical (2N)', color='steelblue')
    ax.plot(n_agents, debate_msgs, 's-', label='Debate (6N+1)', color='seagreen')
    ax.plot(n_agents, full_msgs,   'd-', label='Fully connected (N²-N)', color='coral')
    ax.set_xlabel('Number of agents'); ax.set_ylabel('Total messages')
    ax.set_title('Figure 4 — Communication overhead by architecture and agent count')
    ax.legend(); plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 4.** Message count as a function of agent count for three architectures.
    **Hierarchical** scales linearly — 2N messages regardless of N (orchestrator talks
    to each worker once). **Debate** scales linearly but with higher constant factor
    (6N+1 for 3 rounds). **Fully connected** (every agent talks to every other) scales
    quadratically O(N²) — infeasible beyond ~10 agents. This explains why production
    multi-agent systems always use hierarchical or debate patterns: they bound communication
    overhead. Fully connected "mesh" topologies are only appropriate for very small N.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Orchestrator bottleneck** | All agents wait for orchestrator to dispatch next task | Orchestrator is single-threaded; serial dispatch | Async dispatch; parallel fan-out |
    | **Error propagation** | Bad research → bad analysis → bad report | No validation between stages | Add quality gate: reject result if confidence < threshold |
    | **Agent sycophancy** | Debate agents agree too quickly | LLMs trained to be agreeable; no real disagreement | Explicitly instruct agents to argue their position; use different temperatures |
    | **Context leakage** | Agent B reveals Agent A's private scratchpad | Shared context window | Strict message-passing: agents share only explicit messages, not internal state |
    | **Deadlock** | Proposer waits for validator; validator waits for proposer update | Circular dependency | Timeout per round; max_rounds hard limit |
    | **Communication format drift** | Agent outputs text instead of expected JSON | No schema enforcement | Validate all messages against JSON schema; retry on parse failure |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 LangGraph multi-agent supervisor pattern (guarded).
    try:
        import langgraph  # noqa: F401
        print('LangGraph available.')
    except ImportError:
        lines = [
            '[langgraph not installed — supervisor pattern]:',
            '  from langgraph.graph import StateGraph, END',
            '  from langgraph.prebuilt import create_react_agent',
            '',
            '  # Create specialist agents.',
            '  researcher = create_react_agent(llm, tools=[search_tool, retrieve_tool])',
            '  analyst    = create_react_agent(llm, tools=[stats_tool, chart_tool])',
            '  writer     = create_react_agent(llm, tools=[format_tool])',
            '',
            '  # Supervisor routes tasks to the right agent.',
            '  def supervisor(state):',
            '      next_agent = llm.invoke(routing_prompt.format(state=state))',
            '      return next_agent',
            '',
            '  graph = StateGraph(AgentState)',
            '  graph.add_node("supervisor", supervisor)',
            '  graph.add_node("researcher", researcher)',
            '  graph.add_node("analyst", analyst)',
            '  graph.add_node("writer", writer)',
            '  graph.add_conditional_edges("supervisor",',
            '      lambda s: s["next"], {"researcher": "researcher", "analyst": "analyst",',
            '      "writer": "writer", "FINISH": END})',
            '  for agent in ["researcher", "analyst", "writer"]:',
            '      graph.add_edge(agent, "supervisor")  # always report back',
            '  app = graph.compile()',
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 CrewAI pattern (guarded).
    try:
        from crewai import Agent, Task, Crew  # noqa: F401
        print('CrewAI available.')
    except ImportError:
        lines = [
            '[crewai not installed — CrewAI pattern]:',
            '  from crewai import Agent, Task, Crew',
            '',
            '  researcher = Agent(role="Researcher", goal="Find market data",',
            '                     backstory="Expert at finding accurate market information")',
            '  analyst    = Agent(role="Analyst", goal="Draw business insights")',
            '  writer     = Agent(role="Writer", goal="Write executive reports")',
            '',
            '  task1 = Task(description="Research AI market size and trends",',
            '               agent=researcher, expected_output="Structured findings dict")',
            '  task2 = Task(description="Analyse findings and identify opportunities",',
            '               agent=analyst, expected_output="Insights with scores")',
            '  task3 = Task(description="Write the executive report", agent=writer)',
            '',
            '  crew = Crew(agents=[researcher, analyst, writer], tasks=[task1, task2, task3])',
            '  result = crew.kickoff()',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Research Report Pipeline

    **Scenario.** A management consulting firm produces 50 market analysis reports per
    month. Each takes 3 analysts 2 days = 6 analyst-days per report. They want to
    automate first-draft generation with a multi-agent pipeline.

    **Agent roles:**
    - **Researcher**: web search + RAG over internal knowledge base → structured findings.
    - **Analyst**: statistical analysis of findings → key insights, opportunity scores.
    - **Writer**: formats insights → executive report (title, summary, sections, appendix).
    - **Quality gate**: validation agent checks: Are claims cited? Is the structure correct?
      Does the word count meet the 2-page target?

    **Architecture choice: hierarchical + validation.**
    - Hierarchical for the pipeline (researcher → analyst → writer).
    - Validation after writer: quality gate checks report before delivery.
    - Why not debate? The research pipeline has a clear direction — no need to explore
      alternative plans. Debate would add cost without quality benefit here.

    **Results at 50 reports/month:**
    - First-draft generation time: 8 minutes (vs. 2 days human).
    - Human editing time: 45 minutes per report (vs. 2 days for full report).
    - Quality gate pass rate: 78% on first attempt; 95% after one revision cycle.
    - Monthly LLM cost: ~$3/report = $150/month (vs. $6,000 equivalent analyst time).
    - Analyst focus shift: from research + writing to editing + strategy (higher value).
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Async agent execution.** For independent agents, run in parallel using async
      calls (asyncio, ThreadPoolExecutor). Waiting for all to finish before proceeding
      is a barrier — only use when the next stage genuinely needs all results.
    - **Agent isolation.** Each agent should have its own context window, tool access,
      and memory. Never share internal scratchpad state between agents — only share
      explicit messages. Context leakage (agent A's reasoning appears in agent B's prompt)
      creates unpredictable behaviour.
    - **Message logging and tracing.** Log all inter-agent messages with timestamps.
      This is essential for debugging, cost attribution, and audit. Use OpenTelemetry
      spans where possible — each agent invocation is a span.
    - **Failure recovery.** If a subagent fails (timeout, LLM error), the orchestrator
      should: (a) retry with backoff; (b) fall back to a simpler strategy; (c) escalate
      to human. Never silently skip a failed stage and continue.
    - **Cost attribution.** Multi-agent systems multiply LLM call count. Track costs
      per agent per task. Set per-run cost budgets with circuit breakers.
    - **Human-in-the-loop injection.** For high-stakes tasks, add a human review step
      after the validation agent. The validation agent flags confidence < 0.85 for human
      review; high-confidence outputs go directly to delivery.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Architecture selection:**

    | Architecture | Strengths | Weaknesses | Use for |
    |---|---|---|---|
    | Hierarchical | Clear pipeline; easy to trace; linear cost | Error propagates; bottleneck at orchestrator | Document pipelines, data analysis, code generation |
    | Debate | Balanced views; catches blind spots; good for decisions | Higher LLM cost; sycophancy risk | Architecture decisions, risk analysis, policy drafting |
    | Validation | Quality control; iterative improvement | Convergence not guaranteed; cost per round | Code review, fact-checking, content moderation |
    | Fully connected | Maximum information sharing | O(N²) messages; expensive; complex | Tiny N (<4) only |

    **Single agent vs. multi-agent:**
    - Use single agent when: task fits in one context window; no parallelism needed; latency matters.
    - Use multi-agent when: task exceeds context limits; parallelism available; quality gates needed;
      specialisation improves output.
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"Why use multi-agent instead of a single large-context model?"* → Four reasons:
      (1) specialisation — a researcher prompt differs from a writer prompt; (2) parallelism
      — independent agents run concurrently, reducing wall-clock time; (3) error isolation —
      a failing subagent doesn't crash the whole system; (4) cost — smaller specialised
      calls are cheaper than one massive call with the full task.
    - *"What is the supervisor pattern in LangGraph?"* → A supervisor agent routes work
      to specialist subagents based on the current state. Each specialist reports back to
      the supervisor, which decides the next step. The supervisor is a router + state machine;
      the specialists are tool-equipped agents.

    **Deep-dive questions**
    - *"How do you prevent context leakage between agents?"* → Strict message-passing:
      agents communicate only through explicit structured messages. The orchestrator never
      passes agent A's full internal context to agent B — it passes only the message content.
      Implement as separate LLM calls with separate context windows. Never concatenate
      agent A's reasoning into agent B's prompt.
    - *"How do you handle a subagent that fails mid-pipeline?"* → Three strategies:
      (a) retry with exponential backoff (transient LLM API failure); (b) fall back to
      a simpler tool (complex analysis fails → use rule-based fallback); (c) partial result
      with metadata — continue pipeline with `{"status": "failed", "partial_result": ...}`
      and let downstream agents handle the degraded input. Always log the failure; never
      silently skip.

    **Common mistakes:** sharing full context between agents (context leakage); no cost
    budget per run (runaway cost); no timeout per agent (pipeline hangs); no validation
    between stages (garbage propagates).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Three architectures.** Name them and give one use case each. Which one is O(N²) in messages and why should you avoid it for large N?
    2. **Parallelism.** Researcher: 3s. Analyst: 2s. Writer: 4s. What is the sequential time? What is the parallel time? What is the speedup?
    3. **Communication protocol.** Why is structured JSON preferred over free-text between agents? Give one concrete failure mode from free-text.
    4. **Supervisor pattern.** Draw the LangGraph supervisor pattern: nodes, edges, routing function. When does the supervisor route to END?
    5. **Context leakage.** What is it? How does it happen in a multi-agent system? How do you prevent it?
    6. **Validation convergence.** The proposer improves quality by 0.08 per round; initial quality = 0.55; threshold = 0.85. How many rounds until convergence?
    7. **Debate sycophancy.** What is it and why does it happen with LLMs? Give two mitigations.
    8. **Cost analysis.** A hierarchical system has 4 subagents. Each subagent call costs $0.05. The orchestrator costs $0.03 per call. 1000 reports/month. What is the monthly LLM cost?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Draw the message flow for a hierarchical system with an orchestrator and 3 subagents (researcher, coder, tester). How many messages are exchanged if each subagent also sends a status update mid-task?
    2. A debate system runs 4 rounds with 2 agents. How many total messages does the judge receive? (Each round: 2 arguments + 2 counter-arguments.)

    **Beginner → Intermediate (coding)**
    3. Add a **quality gate** between researcher and analyst in the hierarchical system: if research confidence < 0.75, reject and re-request with `depth='deeper'`. Test that the gate fires on low-confidence research.
    4. Implement a **timeout** in the validation pipeline: if a round takes more than (simulated) 2 seconds, skip to the next round with a time-out flag in the validation result.

    **Intermediate (analysis)**
    5. Run the validation pipeline 20 times with different random seeds. Plot the distribution of rounds to convergence. What is the mean and P90 round count? What does P90 tell you about tail cost?
    6. Implement an **N-agent debate** (N=3, 4, 5). Measure total message count and judge synthesis quality. At what N does the quality plateau? Is the extra cost worth it?

    **Senior (design)**
    7. *System design:* design a multi-agent code review system for a 30-engineer team. 100 PRs/day, each PR has 5 files on average. Design: agent roles (reviewer, security checker, test coverage checker, summariser), architecture, communication protocol, quality gate, human escalation trigger, cost budget.
    8. *Interview:* "Our multi-agent pipeline costs $0.50/report at 1000 reports/day = $15K/month. The CFO wants to cut to $5K/month. What are 3 options?" (Expected: reduce agent count by merging roles; use smaller/cheaper models for lower-stakes stages; add caching for repeated research queries; reduce validation rounds for low-risk tasks.)
    """),

    md(r"""
    ---
    ### Summary
    Multi-agent systems outperform single agents through specialisation, parallelism, and
    independent validation. **Hierarchical**: orchestrator decomposes → specialists execute
    → results aggregate (O(N) messages). **Debate**: agents argue positions → judge
    synthesises balanced answer. **Validation**: proposer → validator → iterate until
    agreement. Always use structured message protocols (JSON), strict agent isolation,
    and per-run cost budgets. Production orchestration: LangGraph supervisor pattern or
    CrewAI for role-based workflows.

    **Phase 6 complete.** Next: **Phase 7 — Evaluation** begins with `36 · Classical ML
    Evaluation` — the complete evaluation toolkit beyond accuracy.
    """),
]

build("phase6_agents/35_multi_agent_systems.ipynb", cells)
