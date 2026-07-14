"""Builder for Notebook 31 — Agent Fundamentals."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 31 · Agent Fundamentals
    ### Phase 6 — Agentic AI · *ML/AI Senior Mastery Curriculum*

    > An LLM answers one question. An **agent** solves a problem — it perceives the
    > environment, reasons about what to do next, acts by calling tools, observes
    > the result, and repeats until done. This notebook teaches the ReAct loop from
    > first principles and builds a minimal but complete agent from scratch: tool
    > registry, thought-action-observation cycle, trajectory logging, and error
    > recovery. Everything Claude Code, GPT-4 with tools, and LangChain agents do
    > is built on this foundation.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Agent definition**: the perceive → reason → act loop; how it differs from
      a single LLM call or a chain.
    - **ReAct (Reasoning + Acting)**: the dominant agent architecture — interleaving
      Thought, Action, and Observation steps in a single prompt context.
    - **Tool registry from scratch**: define tools as Python callables with name,
      description, and argument schema; call them by parsing LLM output.
    - **Trajectory representation**: storing the full agent trace (thought, action,
      observation at each step) for debugging and auditing.
    - **Agent loop control**: max_steps guard, finish detection, error handling.
    - **Failure modes**: hallucinated tool calls, infinite loops, wrong tool
      selection, tool argument errors.
    - Production patterns with LangChain AgentExecutor (guarded).

    **Why it matters**
    - Every production AI agent — customer support bots, coding assistants, data
      analysis pipelines, Claude Code itself — is built on the ReAct pattern with
      a tool registry. Understanding the internals means you can debug agent
      misbehaviour, add new tools, constrain the action space, and design safe
      human-in-the-loop checkpoints.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Classical AI planning (1960s–1990s).** AI agents existed long before LLMs:
    expert systems (MYCIN, 1974), planning algorithms (STRIPS, 1971), and symbolic
    AI agents used hard-coded rules. They were brittle — any situation outside the
    rule base caused failure.

    **Chain-of-Thought prompting (Wei et al., 2022).** Showed that LLMs reason better
    when prompted to show intermediate steps. This was the seed of agent reasoning.

    **ReAct (Yao et al., 2023).** Combined reasoning and acting in a single prompt.
    The agent interleaves *Thought* (reasoning about the current state) and *Action*
    (calling a tool or producing output). The *Observation* from the tool is fed back
    into the context. ReAct outperformed both pure reasoning (no actions) and pure
    acting (no intermediate thoughts) on HotpotQA and ALFWorld benchmarks.

    **Toolformer (Schick et al., 2023).** Trained LLMs to self-insert API calls in
    text, showing that tool use can be learned, not just prompted.

    **Function calling / tool use APIs (2023).** OpenAI function calling (June 2023)
    and Anthropic tool use (Nov 2023) productionised structured tool calling — the
    model outputs a JSON object matching a defined schema instead of free text.
    This eliminated the fragile regex parsing of early ReAct implementations.

    **LangChain, LangGraph (2022–2024).** Popularised agent orchestration frameworks,
    making ReAct agents one-liners for practitioners. The underlying pattern remains
    the same: tool registry + thought-action-observation loop.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Single LLM call vs. agent:**
    ```
    LLM call:  [Prompt] ──► [LLM] ──► [Answer]  (one shot)

    Agent:     [Task] ──► [Thought] ──► [Action] ──► [Tool] ──► [Observation]
                              ▲                                        │
                              └────────────────────────────────────────┘
                         (repeat until Finish)
    ```

    **ReAct trace example:**
    ```
    Task: "What is today's date, and what is 2^10?"

    Thought 1: I need the current date. I'll call the date tool.
    Action 1: get_date()
    Observation 1: 2026-06-23

    Thought 2: Now I need 2^10. I'll use the calculator.
    Action 2: calculator(expression="2**10")
    Observation 2: 1024

    Thought 3: I have both answers. I'll finish.
    Action 3: finish(answer="Today is 2026-06-23 and 2^10 = 1024")
    ```

    **Why interleave thoughts?** Without intermediate reasoning, the agent has no
    working memory between actions. Thought steps allow the agent to plan the next
    action based on what it just learned from the observation — like a programmer
    reading output before writing the next line.

    **Tool registry.** Every tool has:
    - `name`: identifier the LLM uses to call it.
    - `description`: what the tool does (shown to LLM in the system prompt).
    - `schema`: parameter types and descriptions.
    - `function`: the actual Python callable.
    """),

    code(r"""
    import re
    import json
    import math
    import time
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import defaultdict

    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The agent as a Partially Observable Markov Decision Process (POMDP)

    Formally, an agent operates in a POMDP $(S, A, O, T, \Omega, R, \gamma)$:
    - $S$: state space (world state, often not fully observable)
    - $A$: action space (available tools + finish)
    - $O$: observation space (tool outputs)
    - $T(s' | s, a)$: transition (tool execution changes state)
    - $\Omega(o | s', a)$: observation model (tool output given new state)
    - $R(s, a)$: reward (task completion)
    - $\gamma$: discount factor

    In practice, LLM agents don't explicitly model $T$ or $\Omega$ — the LLM learns
    an implicit policy $\pi(a | \text{context})$ from its training.

    ### 4.2 ReAct context structure

    At step $t$, the context is:
    $$C_t = [\text{System}, \text{Task}, \text{Traj}_{1..t-1}]$$
    $$\text{Traj}_i = [\text{Thought}_i, \text{Action}_i, \text{Obs}_i]$$

    The LLM policy samples the next action:
    $$a_t \sim \pi_\theta(\cdot | C_t)$$

    The context grows linearly with the number of steps. This creates a hard limit:
    context window size / average step size = max steps $\approx L_{\max} / (3 \times L_{\text{step}})$.

    ### 4.3 Tool call parsing

    Given LLM output text $y$, extract tool call $(name, args)$:
    $$\text{match} = \text{regex}(r\text{"Action:\s*(\w+)\(([^)]*)\)"}, y)$$
    $$name = \text{match.group}(1), \quad args = \text{parse}(\text{match.group}(2))$$

    Structured tool use (OpenAI/Anthropic APIs) replaces this with a guaranteed
    JSON schema — eliminating parse errors at the cost of requiring API support.

    ### 4.4 Stopping conditions

    The agent terminates when:
    1. The LLM outputs `finish(answer=...)` — task complete.
    2. Step count $t \geq t_{\max}$ — safety guard (prevent infinite loops).
    3. A tool returns an error that cannot be recovered from.
    4. The trajectory repeats (cycle detection) — same (Thought, Action) twice.
    """),

    md(r"""
    ## 5 · Implementation from Scratch

    ### 5a Tool registry
    """),

    code(r"""
    # 5a. Tool registry: define tools as structured objects.

    class Tool:
        def __init__(self, name, description, func, schema=None):
            self.name = name
            self.description = description
            self.func = func
            self.schema = schema or {}

        def call(self, **kwargs):
            try:
                result = self.func(**kwargs)
                return str(result)
            except Exception as e:
                return f'ERROR: {type(e).__name__}: {e}'

        def describe(self):
            return f'{self.name}: {self.description}'

    # Define tools.
    ORDER_DB = {
        'ORD-001': {'status': 'shipped', 'eta': '2026-06-25', 'item': 'Laptop'},
        'ORD-002': {'status': 'processing', 'eta': '2026-06-28', 'item': 'Headphones'},
        'ORD-003': {'status': 'delivered', 'eta': '2026-06-20', 'item': 'Monitor'},
    }

    POLICY_DB = {
        'return': 'Items can be returned within 30 days of delivery. Electronics must be unused.',
        'shipping': 'Standard shipping takes 3-5 business days. Express ships in 1-2 days.',
        'warranty': 'All products come with a 1-year manufacturer warranty.',
    }

    TOOLS = {}

    def register_tool(name, description, func, schema=None):
        TOOLS[name] = Tool(name, description, func, schema)

    register_tool(
        'lookup_order',
        'Look up the status of a customer order. Args: order_id (str)',
        lambda order_id: json.dumps(ORDER_DB.get(order_id, {'error': 'Order not found'})),
        {'order_id': 'string'}
    )

    register_tool(
        'get_policy',
        'Retrieve company policy information. Args: topic (str, one of: return, shipping, warranty)',
        lambda topic: POLICY_DB.get(topic.lower(), 'Policy not found.'),
        {'topic': 'string'}
    )

    register_tool(
        'calculator',
        'Evaluate a mathematical expression. Args: expression (str)',
        lambda expression: str(eval(expression, {'__builtins__': {}}, {'math': math})),
        {'expression': 'string'}
    )

    register_tool(
        'escalate_ticket',
        'Escalate a support ticket to a human agent. Args: reason (str), priority (str: low/medium/high)',
        lambda reason, priority: f'Ticket escalated [{priority.upper()}]: {reason}. Reference ID: ESC-{abs(hash(reason)) % 9999:04d}',
        {'reason': 'string', 'priority': 'string'}
    )

    register_tool(
        'finish',
        'Signal that the task is complete. Args: answer (str)',
        lambda answer: answer,
        {'answer': 'string'}
    )

    print(f'Registered {len(TOOLS)} tools:')
    for t in TOOLS.values():
        print(f'  {t.describe()}')
    """),

    md(r"""
    ### 5b Simulated LLM reasoning
    """),

    code(r"""
    # 5b. Simulated LLM: maps task + trajectory to next Thought + Action.
    # In production, replace with an actual LLM API call.
    # This simulation is fully deterministic for notebook reproducibility.

    def simulated_llm(task, trajectory, tools):
        # Simulate context-aware reasoning based on task and observation history.
        task_lower = task.lower()
        observations = [step.get('observation', '') for step in trajectory]
        actions_taken = [step.get('action_name', '') for step in trajectory]

        # Step 0: understand what we need.
        if not trajectory:
            if 'order' in task_lower and 'ord-' in task_lower:
                order_id = re.search(r'ORD-\d+', task.upper())
                if order_id:
                    return (f'I need to look up order {order_id.group()}.',
                            'lookup_order', {'order_id': order_id.group()})
            if 'return' in task_lower or 'policy' in task_lower:
                return ('I should retrieve the return policy.',
                        'get_policy', {'topic': 'return'})
            if 'ship' in task_lower and 'policy' not in task_lower:
                return ('Let me check the shipping policy first.',
                        'get_policy', {'topic': 'shipping'})
            if any(op in task_lower for op in ['+', '-', '*', '/', '^', 'calculate', 'compute']):
                expr = re.search(r'[\d\s\+\-\*\/\^\(\)\.]+', task)
                if expr:
                    clean = expr.group().strip().replace('^', '**')
                    return (f'This is a math question. I will calculate {clean}.',
                            'calculator', {'expression': clean})
            return ('I need to understand the task better. Let me check the return policy.',
                    'get_policy', {'topic': 'return'})

        # Step 1+: based on last observation.
        last_obs = observations[-1] if observations else ''
        last_action = actions_taken[-1] if actions_taken else ''

        if last_action == 'lookup_order':
            try:
                order_data = json.loads(last_obs)
                if 'error' in order_data:
                    return (f'Order not found. I should escalate this.',
                            'escalate_ticket',
                            {'reason': f'Customer asked about order not in system: {task}',
                             'priority': 'medium'})
                status = order_data.get('status', 'unknown')
                item = order_data.get('item', 'item')
                eta = order_data.get('eta', 'unknown')
                if 'return' in task_lower and status == 'delivered':
                    return (f'Order is delivered. Customer wants to return. Let me get the return policy.',
                            'get_policy', {'topic': 'return'})
                return (f'Order status is {status} for {item}, ETA {eta}. I have enough to answer.',
                        'finish',
                        {'answer': f'Your {item} (status: {status}) is expected by {eta}.'})
            except Exception:
                pass

        if last_action == 'get_policy':
            if 'escalate' in task_lower or ('not working' in task_lower):
                return ('Customer has a complex issue. I should escalate.',
                        'escalate_ticket',
                        {'reason': task[:100], 'priority': 'high'})
            return (f'I have the policy information. I can now answer the customer.',
                    'finish',
                    {'answer': f'Here is the relevant policy: {last_obs}'})

        if last_action == 'escalate_ticket':
            return (f'The ticket has been escalated. I can inform the customer.',
                    'finish',
                    {'answer': f'Your issue has been escalated to our team. {last_obs}'})

        if last_action == 'calculator':
            return (f'Calculation complete. Answering.',
                    'finish',
                    {'answer': f'The result is: {last_obs}'})

        # Default: finish.
        return ('I have gathered enough information.',
                'finish', {'answer': last_obs or 'Task completed.'})

    print('Simulated LLM ready.')
    """),

    md(r"""
    ### 5c The ReAct agent loop
    """),

    code(r"""
    # 5c. ReAct agent loop from scratch.

    class ReActAgent:
        def __init__(self, tools, max_steps=8):
            self.tools = tools
            self.max_steps = max_steps

        def run(self, task, verbose=True):
            trajectory = []
            seen_actions = set()   # cycle detection

            if verbose:
                print(f'\n{"="*60}')
                print(f'Task: {task}')
                print('='*60)

            for step in range(self.max_steps):
                # Reason: call LLM to get next thought + action.
                thought, action_name, action_args = simulated_llm(
                    task, trajectory, self.tools)

                # Cycle detection: same (action, args) twice → abort.
                action_key = f'{action_name}:{json.dumps(action_args, sort_keys=True)}'
                if action_key in seen_actions:
                    if verbose:
                        print(f'\n[ABORT] Cycle detected at step {step+1}.')
                    trajectory.append({'step': step + 1, 'thought': thought,
                                       'action_name': 'abort', 'action_args': {},
                                       'observation': 'ABORT: cycle detected'})
                    break
                seen_actions.add(action_key)

                # Act: execute the tool.
                tool = self.tools.get(action_name)
                if tool is None:
                    observation = f'ERROR: unknown tool "{action_name}"'
                else:
                    observation = tool.call(**action_args)

                step_record = {
                    'step': step + 1,
                    'thought': thought,
                    'action_name': action_name,
                    'action_args': action_args,
                    'observation': observation,
                }
                trajectory.append(step_record)

                if verbose:
                    print(f'\nStep {step+1}:')
                    print(f'  Thought:     {thought}')
                    print(f'  Action:      {action_name}({action_args})')
                    print(f'  Observation: {observation[:100]}')

                # Terminate if finish.
                if action_name == 'finish':
                    if verbose:
                        print(f'\n[DONE] Final answer: {observation}')
                    return {'answer': observation, 'trajectory': trajectory, 'steps': step + 1}

            if verbose:
                print('\n[ABORT] Max steps reached.')
            return {'answer': None, 'trajectory': trajectory, 'steps': self.max_steps}

    agent = ReActAgent(TOOLS, max_steps=8)
    """),

    code(r"""
    # Run example 1: order status query.
    result1 = agent.run('What is the status of my order ORD-001?')
    """),

    code(r"""
    # Run example 2: multi-step — order + return policy.
    result2 = agent.run('I received order ORD-003 but want to return it. What is the return policy?')
    """),

    code(r"""
    # Run example 3: calculation.
    result3 = agent.run('What is 2**10 + 3**5?', verbose=True)
    """),

    code(r"""
    # Run example 4: escalation.
    result4 = agent.run('My product is not working and I need urgent help. Please escalate.', verbose=True)
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Agent trajectory visualisation.
    def plot_trajectory(result, title):
        traj = result['trajectory']
        n = len(traj)
        fig, ax = plt.subplots(figsize=(12, 1.5 * (n + 1)))
        ax.axis('off')

        colors = {
            'thought': '#d4e6f1',
            'action': '#d5f5e3',
            'observation': '#fdebd0',
        }

        y_pos = 1.0
        step_h = 0.25
        for step in traj:
            s = step['step']
            # Thought box.
            ax.text(0.02, y_pos, f"[{s}] Thought:", fontsize=8, fontweight='bold', va='top')
            ax.text(0.15, y_pos, step['thought'][:80], fontsize=8, va='top',
                    bbox=dict(facecolor=colors['thought'], alpha=0.6, boxstyle='round'))
            y_pos -= step_h
            # Action box.
            ax.text(0.02, y_pos, 'Action:', fontsize=8, fontweight='bold', va='top')
            ax.text(0.15, y_pos, f'{step["action_name"]}({step["action_args"]})'[:80],
                    fontsize=8, va='top',
                    bbox=dict(facecolor=colors['action'], alpha=0.6, boxstyle='round'))
            y_pos -= step_h
            # Observation box.
            ax.text(0.02, y_pos, 'Obs:', fontsize=8, fontweight='bold', va='top')
            ax.text(0.15, y_pos, str(step['observation'])[:90], fontsize=8, va='top',
                    bbox=dict(facecolor=colors['observation'], alpha=0.6, boxstyle='round'))
            y_pos -= step_h * 1.3

        ax.set_xlim(0, 1.2)
        ax.set_ylim(y_pos - 0.1, 1.1)
        ax.set_title(f'Figure 1 — Agent Trajectory: {title}\n({result["steps"]} steps)', fontsize=10)
        # Legend.
        patches = [mpatches.Patch(color=colors[k], label=k.capitalize()) for k in colors]
        ax.legend(handles=patches, loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

    plot_trajectory(result2, 'Order return query (multi-step)')
    """),

    md(r"""
    **Figure 1.** Agent trajectory for a multi-step order return query. Blue boxes
    show the agent's *Thought* at each step — the internal reasoning that justifies
    the next action. Green boxes show the *Action* — the tool call with arguments.
    Orange boxes show the *Observation* — the tool's response fed back to the agent.
    Notice that the agent adapts: it first looks up the order (Step 1), sees the
    order is delivered, then decides to fetch the return policy (Step 2), and only
    then formulates the final answer (Step 3). This three-step plan was not
    pre-specified — it emerged from the agent's tool-conditioned reasoning.
    """),

    code(r"""
    # Figure 2 — Steps used per task type.
    tasks_and_labels = [
        ('What is the status of my order ORD-001?', 'Order status'),
        ('I received order ORD-003 but want to return it. What is the return policy?', 'Order + return'),
        ('What is 2**10 + 3**5?', 'Calculation'),
        ('My product is not working and I need urgent help. Please escalate.', 'Escalation'),
        ('What is the shipping policy?', 'Policy lookup'),
    ]

    results_all = []
    for task, label in tasks_and_labels:
        r = agent.run(task, verbose=False)
        results_all.append({'label': label, 'steps': r['steps'],
                             'success': r['answer'] is not None})

    fig, ax = plt.subplots(figsize=(9, 4))
    labels  = [r['label'] for r in results_all]
    steps   = [r['steps'] for r in results_all]
    colors  = ['seagreen' if r['success'] else 'salmon' for r in results_all]
    ax.barh(labels, steps, color=colors, alpha=0.8)
    ax.set_xlabel('Steps used')
    ax.set_title('Figure 2 — Steps per task type (green = success, red = failed)')
    ax.set_xlim(0, max(steps) + 1)
    for i, s in enumerate(steps):
        ax.text(s + 0.05, i, str(s), va='center', fontsize=10)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** Steps per task type. Simple single-tool tasks (order status,
    policy lookup, calculation) complete in 2 steps: one tool call + finish.
    Multi-step tasks (order + return policy) need 3 steps: two tool calls + finish.
    Escalation tasks route quickly to the `escalate_ticket` tool then finish. The
    agent never exceeds 4 steps on these tasks because the simulated LLM is well-
    calibrated. In production with a real LLM, step counts are higher and less
    predictable — always set a `max_steps` guard (typically 10–15).
    """),

    code(r"""
    # Figure 3 — Failure mode: hallucinated tool name.
    print('=== Failure mode: hallucinated tool name ===')

    class BrokenLLM:
        # Simulates an LLM that hallucinates a non-existent tool.
        def __call__(self, task, trajectory, tools):
            if not trajectory:
                return ('I will use the magic_search tool.', 'magic_search', {'query': task})
            # After error, try to recover.
            last_obs = trajectory[-1].get('observation', '')
            if 'ERROR' in last_obs:
                return ('The tool failed. Let me use the correct search.',
                        'finish', {'answer': 'I was unable to find the information.'})
            return ('Done.', 'finish', {'answer': trajectory[-1].get('observation', 'Done.')})

    # Patch simulated_llm temporarily to test error handling.
    broken = BrokenLLM()

    class ReActAgentWithBrokenLLM(ReActAgent):
        def run(self, task, verbose=True):
            trajectory = []
            if verbose:
                print(f'Task: {task}')
            for step in range(self.max_steps):
                thought, action_name, action_args = broken(task, trajectory, self.tools)
                tool = self.tools.get(action_name)
                if tool is None:
                    observation = f'ERROR: unknown tool "{action_name}"'
                else:
                    observation = tool.call(**action_args)
                step_record = {'step': step+1, 'thought': thought,
                               'action_name': action_name, 'action_args': action_args,
                               'observation': observation}
                trajectory.append(step_record)
                if verbose:
                    print(f'  Step {step+1}: {action_name} → {observation[:60]}')
                if action_name == 'finish':
                    if verbose:
                        print(f'  Answer: {observation}')
                    return {'answer': observation, 'trajectory': trajectory, 'steps': step+1}
            return {'answer': None, 'trajectory': trajectory, 'steps': self.max_steps}

    broken_agent = ReActAgentWithBrokenLLM(TOOLS, max_steps=4)
    result_broken = broken_agent.run('Search for the latest Python release', verbose=True)
    """),

    md(r"""
    **Failure mode demonstration.** The simulated broken LLM calls `magic_search`
    (a non-existent tool), receives an `ERROR: unknown tool` observation, and then
    recovers by routing to `finish`. In production:
    - **Robust agents** should handle tool errors by trying an alternative tool or
      asking for clarification.
    - **Monitoring** should alert when the error observation rate exceeds a threshold.
    - **Tool validation** at call time (not just parse time) catches type mismatches
      before the tool executes.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Hallucinated tool** | `ERROR: unknown tool` in observation | LLM invents tool not in registry | Enumerate tools in system prompt; use structured tool calling |
    | **Infinite loop** | Agent repeats same action forever | No termination condition | Cycle detection (same action twice → abort); max_steps guard |
    | **Wrong tool** | Right result, wrong path | Tool descriptions ambiguous | Improve tool descriptions; add examples; use few-shot prompting |
    | **Argument error** | Tool call with wrong types | LLM misunderstands schema | Add argument schema to system prompt; validate before calling |
    | **Context overflow** | Agent truncates early steps | Trajectory > context window | Summarise old trajectory steps; use sliding window |
    | **Premature finish** | Agent finishes without enough info | Weak finish condition | Require explicit answer field; validate answer completeness |
    | **Tool timeout** | Agent hangs | External tool slow/unavailable | Add per-tool timeout; fallback response on timeout |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 LangChain AgentExecutor pattern (guarded).
    try:
        from langchain.agents import AgentExecutor  # noqa: F401
        lines = [
            'from langchain.agents import AgentExecutor, create_react_agent',
            'from langchain.tools import Tool',
            'from langchain_core.prompts import PromptTemplate',
            '',
            '# Define tools.',
            'tools = [',
            '    Tool(name="lookup_order", func=lookup_fn,',
            '         description="Look up customer order status. Args: order_id"),',
            '    Tool(name="get_policy", func=policy_fn,',
            '         description="Get company policy. Args: topic"),',
            ']',
            '',
            '# Create ReAct agent.',
            'agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)',
            'executor = AgentExecutor(agent=agent, tools=tools,',
            '                        max_iterations=10, handle_parsing_errors=True)',
            'result = executor.invoke({"input": "What is the status of order ORD-001?"})',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[langchain not installed — production pattern]:',
            '  from langchain.agents import AgentExecutor, create_react_agent',
            '  from langchain.tools import Tool',
            '  tools = [Tool(name="...", func=fn, description="...")]',
            '  agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)',
            '  executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)',
            '  result = executor.invoke({"input": task})',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Customer Support Agent

    **Scenario.** An e-commerce company handles 50K support tickets/day. 70% are
    routine (order status, policy questions) — currently handled by human agents
    at 5 minutes average resolution time and $3.50 cost per ticket. They want to
    automate routine tickets with an AI agent.

    **Agent capabilities:**
    - `lookup_order`: queries order management system via REST API.
    - `get_policy`: retrieves from policy knowledge base (Notebook 25–27 RAG stack).
    - `escalate_ticket`: routes to human agent queue with priority and context.
    - `send_email`: sends automated response to customer.

    **ReAct loop:**
    1. Customer submits ticket → agent reads it as task.
    2. Agent reasons about what information is needed.
    3. Agent calls tools (order lookup, policy retrieval) in sequence.
    4. Agent drafts response and calls `send_email`.
    5. If unresolvable → `escalate_ticket` with context summary.

    **Results (pilot):**
    - 65% of tickets resolved autonomously (4-step average trajectory).
    - Average resolution time: 8 seconds (vs. 5 minutes human).
    - Cost per resolved ticket: $0.04 (LLM inference).
    - Human escalation rate: 35% (complex issues, complaints, edge cases).
    - Customer CSAT: 4.1/5.0 (vs. 3.8/5.0 for human agents on routine tickets).

    **Monitoring:** trajectory length distribution (alert on >8 steps), escalation
    rate per ticket category, answer accuracy on 50-ticket weekly golden set.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Structured tool calling.** Replace free-text action parsing with the model's
      native tool use API (OpenAI function calling, Anthropic tool use). This
      eliminates regex parsing failures and argument type errors.
    - **Tool timeout and retry.** Every tool call must have a timeout (e.g. 5s for
      API calls). On timeout, the agent should observe an error and retry once or
      route to a fallback. Unbounded tool calls block the agent indefinitely.
    - **Trajectory length monitoring.** Alert when trajectories exceed a threshold
      (e.g. >8 steps) — this often indicates the agent is stuck in a reasoning loop.
    - **Human-in-the-loop checkpoints.** For high-stakes actions (send email, make
      payment, delete data), insert a confirmation step: the agent prepares the action
      and waits for human approval. Implement as a special `confirm_with_human` tool.
    - **Context window management.** At 10 steps × 500 tokens/step = 5K tokens of
      trajectory. For long-running agents (30+ steps), summarise old trajectory steps
      to free context space without losing state.
    - **Audit logging.** Log the full trajectory (task, thoughts, actions, observations,
      final answer) for every agent run. Required for debugging, compliance, and
      improvement datasets.
    - **Tool access control.** Restrict which tools each agent instance can call.
      A customer support agent should not have access to admin tools. Use a
      tool registry per agent role.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Agent paradigm comparison:**

    | Paradigm | Flexibility | Latency | Control | Debugging | Use case |
    |---|---|---|---|---|---|
    | Single LLM call | Low | Fastest | High | Easy | Simple Q&A, summarisation |
    | Fixed chain (LangChain) | Medium | Fast | High | Medium | Known multi-step workflows |
    | ReAct agent | High | Slow (N × LLM calls) | Low | Hard | Open-ended problem solving |
    | Plan-then-execute | High | Medium (1 plan + N exec) | Medium | Medium | Well-defined sub-tasks |

    **Tool calling approaches:**

    | Approach | Reliability | Flexibility | Requires | When to use |
    |---|---|---|---|---|
    | Free-text regex parsing | Low | High | Text formatting compliance | Legacy systems |
    | Structured tool calling (API) | High | High | LLM with native tool support | **Production default** |
    | JSON schema extraction | Medium | Medium | Prompt engineering | Models without native tool use |
    | Code execution (Python REPL) | Very high | Very high | Sandbox environment | Data analysis, coding agents |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What is ReAct and how does it work?"* → ReAct (Reasoning + Acting) is an
      agent paradigm that interleaves Thought, Action, and Observation steps. The LLM
      generates a Thought (reasoning), an Action (tool call), receives an Observation
      (tool result), and repeats. The full trajectory is kept in the LLM context,
      giving the agent working memory across steps.
    - *"How do you prevent an agent from looping forever?"* → Three mechanisms: (1)
      max_steps guard (hard limit); (2) cycle detection (abort if the same (action,
      args) pair repeats); (3) decreasing temperature with each step (makes the agent
      more deterministic as steps increase).

    **Deep-dive questions**
    - *"What is the relationship between an agent and a POMDP?"* → An agent operates
      in a POMDP where the state is partially observable (the agent only sees tool
      outputs, not world state). Actions are tool calls. Observations are tool results.
      The LLM implicitly learns a policy $\pi(a|C_t)$ from training, but doesn't
      explicitly model the transition function or reward.
    - *"What breaks when the context window fills up?"* → Old trajectory steps are
      truncated. The agent loses memory of earlier observations (e.g., what it found
      in step 2 when it's at step 25). Mitigation: summarise old steps; use a
      key-value memory store; implement explicit memory (Notebook 33).

    **Common mistakes:** not setting max_steps (infinite loops); no cycle detection;
    using free-text action parsing in production (fragile); not logging trajectories
    for debugging; exposing admin tools to untrusted agent inputs.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Agent vs. LLM call.** What are the 3 components of the ReAct loop? How does
       it differ from a single LLM call?
    2. **Tool registry.** What 4 fields does each tool need? Why is the description
       the most important?
    3. **Trajectory.** What is stored in each trajectory step? Why is the full
       trajectory kept in the LLM context?
    4. **Stopping conditions.** Name 3 valid stopping conditions for an agent loop.
    5. **Hallucinated tool.** What happens when the LLM outputs an action with a tool
       name not in the registry? How should the agent handle this?
    6. **Context overflow.** A ReAct agent runs for 20 steps. Each step is 500 tokens.
       The model has an 8K context. What happens at step 17? How do you fix it?
    7. **Structured tool calling.** What problem does OpenAI/Anthropic native tool
       use solve compared to free-text action parsing?
    8. **Human-in-the-loop.** Give a concrete example of when you'd add a `confirm_with_human`
       tool to the agent's registry and why.
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Write out a 4-step ReAct trajectory (Thought, Action, Observation, repeat) for
       the task: "Book a flight from London to Paris on 2026-07-01." What tools would
       you need? What would each observation contain?
    2. An agent's tool is called `search_web` in the registry but the LLM outputs
       `Action: web_search(query="...")`. What happens? How do you prevent this?

    **Beginner → Intermediate (coding)**
    3. Add a `search_policy` tool to the registry that searches POLICY_DB by keyword
       (e.g. "30 days" should return the return policy). Run the agent on the task
       "How long do I have to return a product?" and verify it finds the answer.
    4. Implement **max_retries per tool**: if a tool call raises an exception, retry
       up to 2 times with exponential backoff before returning an error observation.
       Test it with a tool that fails 50% of the time.

    **Intermediate (analysis)**
    5. Implement **trajectory compression**: after step 5, summarise steps 1–3 into a
       single "memory" string (key facts extracted) and replace those steps in the
       context. Measure: (a) context size before/after; (b) whether the agent still
       answers correctly.
    6. Implement **tool selection audit**: for each completed trajectory, compute which
       tool was called first, second, third. Over 20 diverse tasks, plot the tool call
       frequency by position. Identify which tools are used most and suggest which
       should appear first in the system prompt description.

    **Senior (design)**
    7. *System design:* design a customer support agent that handles 10K tickets/day.
       Specify: tool registry (minimum 5 tools with descriptions), max_steps, escalation
       criteria, monitoring metrics, audit logging schema, and how you handle a tool
       outage (the order system API is down for 30 minutes).
    8. *Interview:* "Our ReAct agent for code review works well in testing but loops
       in production. The average trajectory is 12 steps (test: 4 steps). What are
       3 possible root causes and how would you diagnose each?"
    """),

    md(r"""
    ---
    ### Summary
    An AI agent perceives, reasons, and acts in a loop. **ReAct** (Reasoning + Acting)
    is the dominant architecture: interleave Thought (why), Action (tool call), and
    Observation (result) in the LLM context. A **tool registry** maps tool names to
    Python callables with descriptions and schemas. Always guard with **max_steps**
    and **cycle detection**. Use the LLM's native **structured tool calling API** in
    production to eliminate regex parsing failures. Log **full trajectories** for
    debugging and improvement.

    **Next:** `32 · Planning and Tool Use` — how agents can plan multi-step tasks
    upfront (ReWOO, Tree-of-Thought) before executing, reducing errors from reactive
    step-by-step decision making.
    """),
]

build("phase6_agents/31_agent_fundamentals.ipynb", cells)
