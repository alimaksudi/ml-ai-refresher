"""Builder for Lesson AGT-02 — Planning and Tool Use."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # AGT-02 · Planning and Tool Use
    ### Section 07 — Agentic AI · *ML/AI Senior Mastery Curriculum*

    > Lesson AGT-01 built a ReAct agent that decides what to do one step at a time.
    > This works for simple tasks but struggles when multi-step tasks require
    > coordination — the agent may take an action in step 2 that conflicts with
    > what step 4 needs. **Planning** separates reasoning about the full task from
    > executing individual steps. This notebook teaches four planning strategies
    > (direct action, CoT, ReWOO, Tree-of-Thought), structured tool use with JSON
    > schemas, and tool call validation.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Why plan?** When reactive step-by-step decision making fails.
    - **Four planning strategies**: direct action, chain-of-thought, ReWOO
      (plan-all-then-execute), Tree-of-Thought (explore-then-commit).
    - **Planner from scratch**: decompose a task into sub-tasks, detect dependencies,
      execute in topological order.
    - **Tool use validation**: check tool exists, validate argument types, handle
      tool errors gracefully without crashing the agent.
    - **JSON tool schemas**: the OpenAI function calling and Anthropic tool use
      format — the production standard for structured tool invocation.
    - **ReWOO from scratch**: generate a full plan with variable references
      (`#E1`, `#E2`) before executing.
    - **Tree-of-Thought from scratch**: generate N candidate plans, score each,
      execute the best.

    **Why it matters**
    - Planning is what separates a toy chatbot from a production data analysis
      pipeline or automated code reviewer. Without planning, multi-step tasks
      degrade gracefully into confusion. Understanding when to plan (and when
      not to) is a senior engineering judgment call.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Classical AI planning.** STRIPS (Fikes & Nilsson, 1971) formalised planning
    as: given current state, goal state, and action operators with preconditions and
    effects, find a sequence of actions to reach the goal. This was the dominant AI
    planning framework for 30 years.

    **Chain-of-Thought (Wei et al., 2022).** LLMs improved on multi-step reasoning
    tasks when prompted to show step-by-step thinking. This is implicit planning —
    the chain of thought reflects a plan, but it's generated and executed
    simultaneously.

    **ReWOO (Xu et al., 2023).** "Reasoning Without Observation" — an explicit
    planner generates a full plan with variable references (`#E1 = tool(args)`)
    before any tool is called. Execution substitutes `#E1` values into dependent
    steps. Reduces LLM calls (no round-trips for each observation) but requires
    the planner to be correct upfront.

    **Tree-of-Thought (Yao et al., 2023).** Extend chain-of-thought to a tree:
    generate multiple partial plans, evaluate them (using LLM self-evaluation or
    a heuristic), and prune branches. Inspired by MCTS in game playing. Effective
    for tasks where early decisions commit to a direction and mistakes are hard
    to recover from.

    **OpenAI function calling (June 2023) / Anthropic tool use (Nov 2023).**
    Replaced free-text action parsing with structured JSON output matching a
    defined schema. The model is constrained to output valid tool invocations —
    eliminating a major class of agent failures.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Four planning strategies spectrum:**
    ```
    Direct action: Task → [LLM] → Action (one shot, no planning)
    Chain-of-Thought: Task → [LLM: step-by-step reasoning] → Action sequence
    ReWOO:  Task → [Planner: full plan] → Execute all → Answer (batch, no mid-course feedback)
    ToT:    Task → [Planner: N plans] → [Evaluator: score] → Best plan → Execute
    ```

    **When to use each:**

    | Strategy | Use when | Avoid when |
    |---|---|---|
    | Direct action | Task needs 1 tool, answer obvious | Multi-step, ambiguous |
    | CoT (ReAct) | Task is sequential, observations needed mid-way | Very long tasks |
    | ReWOO | All information available at plan time; want fewer LLM calls | Each step's args depend on previous observations |
    | Tree-of-Thought | Multiple valid approaches; early commitment is costly | Simple linear tasks; high latency budget |

    **ReWOO variable reference example:**
    ```
    Plan:
      #E1 = sql_query(query="SELECT COUNT(*) FROM orders WHERE status='pending'")
      #E2 = sql_query(query="SELECT AVG(value) FROM orders WHERE id IN (#E1_ids)")
      #E3 = format_report(data=#E2, title="Pending Orders Summary")
    Execute:
      Execute #E1 → result
      Substitute into #E2 → Execute → result
      Substitute into #E3 → Final report
    ```

    **Tree-of-Thought example:**
    ```
    Task: Analyse sales data
    Plan A: query sales → compute stats → generate chart → report
    Plan B: query sales → identify anomalies → deep-dive → report
    Plan C: query all tables → join → analyse → report
    [Evaluator]: A=0.8, B=0.6, C=0.5 → Choose A
    Execute Plan A
    ```
    """),

    code(r"""
    import re
    import json
    import math
    import time
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import defaultdict

    rng_global = random.Random(42)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Task dependency graph

    Model the task as a DAG: nodes are sub-tasks, edges are data dependencies.
    Sub-task $B$ depends on sub-task $A$ if $B$'s inputs require $A$'s output.

    **Topological sort** gives a valid execution order:
    1. Find all nodes with in-degree 0 (no unresolved dependencies).
    2. Add them to the execution queue.
    3. Remove them from the graph; repeat.

    **Parallel execution**: nodes at the same topological level can run concurrently.
    Wall-clock time = time of the critical (longest) path, not sum of all tasks.

    ### 4.2 ReWOO formal definition

    A ReWOO plan $P = [(t_1, f_1, a_1), \dots, (t_n, f_n, a_n)]$ where:
    - $t_i$ = sub-task description
    - $f_i$ = tool name
    - $a_i$ = argument template (may contain $\#E_j$ references for $j < i$)

    Execution: $e_i = f_i(\text{resolve}(a_i, \{e_1,\dots,e_{i-1}\}))$

    Variable resolution: $\text{resolve}(a, E) = a$ with each `#Ej` replaced by `E[j]`.

    ### 4.3 Tree-of-Thought search

    Generate $b$ candidate plans at each depth $d$. Evaluate with a value function $V$.
    Greedy: take the best plan at each depth (beam width = 1).
    Beam search: keep top-$k$ plans at each depth (beam width = $k$).

    Expected quality: $\mathbb{E}[\max_{i=1}^b V(P_i)] > \mathbb{E}[V(P_1)]$ when
    $\text{Var}(V)$ is high — diversity helps when the value landscape has multiple peaks.

    ### 4.4 JSON tool schema (OpenAI / Anthropic format)

    ```json
    {
      "name": "sql_query",
      "description": "Execute a SQL query on the database.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "SQL SELECT statement"},
          "limit": {"type": "integer", "description": "Max rows (default 100)"}
        },
        "required": ["query"]
      }
    }
    ```

    The model is constrained to output JSON matching this schema — eliminating
    free-text parse failures. Type validation catches `limit="all"` before execution.
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a Tool registry with validation
    """),

    code(r"""
    # 5a. Tool registry with argument validation and error handling.

    SALES_DB = {
        'orders': [
            {'id': 1, 'product': 'Laptop', 'value': 1200, 'status': 'completed', 'region': 'EU'},
            {'id': 2, 'product': 'Phone',  'value': 800,  'status': 'pending',   'region': 'US'},
            {'id': 3, 'product': 'Tablet', 'value': 600,  'status': 'completed', 'region': 'US'},
            {'id': 4, 'product': 'Laptop', 'value': 1100, 'status': 'completed', 'region': 'EU'},
            {'id': 5, 'product': 'Phone',  'value': 750,  'status': 'pending',   'region': 'APAC'},
            {'id': 6, 'product': 'Tablet', 'value': 550,  'status': 'completed', 'region': 'EU'},
        ]
    }

    class ValidatedTool:
        def __init__(self, name, description, func, schema):
            self.name = name
            self.description = description
            self.func = func
            self.schema = schema   # {param_name: {'type': str, 'required': bool}}

        def validate(self, kwargs):
            errors = []
            for param, spec in self.schema.items():
                if spec.get('required', False) and param not in kwargs:
                    errors.append(f'Missing required param: {param}')
                if param in kwargs:
                    expected_type = spec.get('type')
                    if expected_type == 'string' and not isinstance(kwargs[param], str):
                        errors.append(f'Param {param} must be string, got {type(kwargs[param]).__name__}')
                    if expected_type == 'integer' and not isinstance(kwargs[param], int):
                        errors.append(f'Param {param} must be int, got {type(kwargs[param]).__name__}')
            return errors

        def call(self, **kwargs):
            errors = self.validate(kwargs)
            if errors:
                return {'error': 'ValidationError', 'details': errors}
            try:
                return self.func(**kwargs)
            except Exception as e:
                return {'error': type(e).__name__, 'message': str(e)}

    TOOLS_V = {}

    def reg(name, desc, func, schema):
        TOOLS_V[name] = ValidatedTool(name, desc, func, schema)

    reg('query_orders',
        'Query orders table with optional filter. Args: filter_status (str, optional), filter_region (str, optional)',
        lambda filter_status=None, filter_region=None: [
            r for r in SALES_DB['orders']
            if (filter_status is None or r['status'] == filter_status)
            and (filter_region is None or r['region'] == filter_region)
        ],
        {'filter_status': {'type': 'string', 'required': False},
         'filter_region': {'type': 'string', 'required': False}})

    reg('compute_stats',
        'Compute statistics (mean, sum, count) for a list of numeric values. Args: values (list), metric (str)',
        lambda values, metric: {
            'sum': sum(values), 'mean': sum(values)/len(values) if values else 0,
            'count': len(values), 'max': max(values) if values else 0, 'min': min(values) if values else 0
        }.get(metric, sum(values)),
        {'metric': {'type': 'string', 'required': True}})

    reg('format_report',
        'Format a report string from data. Args: title (str), summary (str)',
        lambda title, summary: f'[REPORT] {title}\n{summary}',
        {'title': {'type': 'string', 'required': True},
         'summary': {'type': 'string', 'required': True}})

    reg('finish', 'Signal completion. Args: answer (str)',
        lambda answer: answer,
        {'answer': {'type': 'string', 'required': True}})

    print(f'Registered {len(TOOLS_V)} validated tools.')

    # Demonstrate validation.
    print('\nTool validation demo:')
    result = TOOLS_V['compute_stats'].call(values=[100, 200, 300], metric='mean')
    print(f'  Valid call: {result}')
    result_bad = TOOLS_V['compute_stats'].call(values=[100], metric=42)   # wrong type
    print(f'  Bad type:   {result_bad}')
    result_miss = TOOLS_V['compute_stats'].call(values=[100])              # missing required
    print(f'  Missing arg: {result_miss}')
    """),

    md(r"""
    ### 5b Chain-of-Thought planning (ReAct-style)
    """),

    code(r"""
    # 5b. CoT planning: agent reasons one step at a time (from Lesson AGT-01).
    # Abbreviated version for comparison baseline.

    def cot_plan_and_execute(task, verbose=True):
        # Simulated sequential reasoning: determine steps from task keywords.
        steps = []
        task_lower = task.lower()
        if 'pending' in task_lower:
            steps.append(('query_orders', {'filter_status': 'pending'}))
        elif 'eu' in task_lower:
            steps.append(('query_orders', {'filter_region': 'EU'}))
        else:
            steps.append(('query_orders', {}))
        steps.append(('compute_stats', {'values': '__PREV__', 'metric': 'sum'}))
        steps.append(('format_report', {'title': 'Sales Analysis', 'summary': '__PREV__'}))
        steps.append(('finish', {'answer': '__PREV__'}))

        prev_result = None
        trajectory = []
        for tool_name, args in steps:
            if verbose:
                print(f'  CoT → {tool_name}({args})')
            filled = {}
            for k, v in args.items():
                if v == '__PREV__':
                    if tool_name == 'compute_stats':
                        filled['values'] = [r['value'] for r in prev_result]
                    elif tool_name == 'format_report':
                        filled['summary'] = str(prev_result)
                    elif tool_name == 'finish':
                        filled['answer'] = str(prev_result)
                    else:
                        filled[k] = prev_result
                else:
                    filled[k] = v
            result = TOOLS_V[tool_name].call(**filled)
            prev_result = result
            trajectory.append({'tool': tool_name, 'args': filled, 'result': result})
            if tool_name == 'finish':
                return result, trajectory

        return prev_result, trajectory

    print('CoT execution for "Analyse pending orders":')
    answer, traj = cot_plan_and_execute('Analyse pending orders', verbose=True)
    print(f'Answer: {answer}')
    """),

    md(r"""
    ### 5c ReWOO — plan all steps upfront
    """),

    code(r"""
    # 5c. ReWOO: generate full plan with #E variable references, then execute.

    class ReWOOPlanner:
        def __init__(self, tools):
            self.tools = tools

        def generate_plan(self, task):
            # Simulate LLM generating a full plan for a data analysis task.
            # In production: call LLM with planning prompt.
            task_lower = task.lower()

            if 'eu' in task_lower and 'revenue' in task_lower:
                plan = [
                    {'id': '#E1', 'tool': 'query_orders',
                     'args': {'filter_region': 'EU'},
                     'description': 'Get all EU orders'},
                    {'id': '#E2', 'tool': 'compute_stats',
                     'args': {'values': '#E1.values', 'metric': 'sum'},
                     'description': 'Sum EU order values'},
                    {'id': '#E3', 'tool': 'format_report',
                     'args': {'title': 'EU Revenue Report', 'summary': 'Total EU revenue: #E2'},
                     'description': 'Format the report'},
                    {'id': '#E4', 'tool': 'finish',
                     'args': {'answer': '#E3'},
                     'description': 'Return the report'},
                ]
            elif 'pending' in task_lower:
                plan = [
                    {'id': '#E1', 'tool': 'query_orders',
                     'args': {'filter_status': 'pending'},
                     'description': 'Get pending orders'},
                    {'id': '#E2', 'tool': 'compute_stats',
                     'args': {'values': '#E1.values', 'metric': 'count'},
                     'description': 'Count pending orders'},
                    {'id': '#E3', 'tool': 'format_report',
                     'args': {'title': 'Pending Orders Report', 'summary': 'Pending count: #E2'},
                     'description': 'Format the report'},
                    {'id': '#E4', 'tool': 'finish',
                     'args': {'answer': '#E3'},
                     'description': 'Return the report'},
                ]
            else:
                plan = [
                    {'id': '#E1', 'tool': 'query_orders', 'args': {},
                     'description': 'Get all orders'},
                    {'id': '#E2', 'tool': 'compute_stats',
                     'args': {'values': '#E1.values', 'metric': 'mean'},
                     'description': 'Compute mean order value'},
                    {'id': '#E3', 'tool': 'format_report',
                     'args': {'title': 'Sales Summary', 'summary': 'Mean order value: #E2'},
                     'description': 'Format the report'},
                    {'id': '#E4', 'tool': 'finish',
                     'args': {'answer': '#E3'},
                     'description': 'Return answer'},
                ]
            return plan

        def resolve(self, arg_val, evidence):
            # Substitute #E references into arg values.
            if isinstance(arg_val, str):
                for eid, eresult in evidence.items():
                    if eid + '.values' in arg_val:
                        # Return numeric values from list of dicts.
                        if isinstance(eresult, list):
                            return [r.get('value', 0) for r in eresult if isinstance(r, dict)]
                        return eresult
                    if eid in arg_val:
                        arg_val = arg_val.replace(eid, str(eresult))
            return arg_val

        def execute(self, plan, verbose=True):
            evidence = {}
            for step in plan:
                eid = step['id']
                tool_name = step['tool']
                tool = self.tools.get(tool_name)
                if tool is None:
                    evidence[eid] = f'ERROR: unknown tool {tool_name}'
                    continue
                # Resolve arg values.
                resolved_args = {k: self.resolve(v, evidence) for k, v in step['args'].items()}
                result = tool.call(**resolved_args)
                evidence[eid] = result
                if verbose:
                    print(f'  {eid} [{tool_name}] → {str(result)[:80]}')
                if tool_name == 'finish':
                    return result, evidence
            return list(evidence.values())[-1], evidence

        def run(self, task, verbose=True):
            if verbose:
                print(f'Task: {task}')
                print('--- Planning phase ---')
            plan = self.generate_plan(task)
            if verbose:
                for step in plan:
                    print(f'  {step["id"]}: {step["tool"]}({step["args"]}) — {step["description"]}')
                print('--- Execution phase ---')
            return self.execute(plan, verbose=verbose), plan

    planner = ReWOOPlanner(TOOLS_V)

    print('\n=== ReWOO: EU Revenue Analysis ===')
    (answer, evidence), plan = planner.run('What is the total EU revenue?', verbose=True)
    print(f'\nFinal answer: {answer}')

    print('\n=== ReWOO: Pending Orders ===')
    (answer2, _), _ = planner.run('How many pending orders do we have?', verbose=True)
    print(f'\nFinal answer: {answer2}')
    """),

    md(r"""
    ### 5d Tree-of-Thought planning
    """),

    code(r"""
    # 5d. Tree-of-Thought: generate N candidate plans, score each, execute the best.

    def generate_candidate_plans(task):
        # Simulate LLM generating 3 alternative plans for a data analysis task.
        task_lower = task.lower()
        candidates = [
            {
                'name': 'Plan A: Direct stats',
                'steps': [
                    {'tool': 'query_orders', 'args': {}, 'desc': 'Get all orders'},
                    {'tool': 'compute_stats', 'args': {'metric': 'mean'}, 'desc': 'Compute mean'},
                    {'tool': 'format_report', 'args': {'title': 'Summary'}, 'desc': 'Format'},
                ],
                'risk': 'low',
                'completeness': 0.7,
            },
            {
                'name': 'Plan B: Segment then analyse',
                'steps': [
                    {'tool': 'query_orders', 'args': {'filter_status': 'completed'}, 'desc': 'Completed orders'},
                    {'tool': 'compute_stats', 'args': {'metric': 'sum'}, 'desc': 'Total revenue'},
                    {'tool': 'query_orders', 'args': {'filter_status': 'pending'}, 'desc': 'Pending orders'},
                    {'tool': 'compute_stats', 'args': {'metric': 'count'}, 'desc': 'Pending count'},
                    {'tool': 'format_report', 'args': {'title': 'Full Analysis'}, 'desc': 'Format'},
                ],
                'risk': 'medium',
                'completeness': 0.95,
            },
            {
                'name': 'Plan C: Region breakdown',
                'steps': [
                    {'tool': 'query_orders', 'args': {'filter_region': 'EU'}, 'desc': 'EU orders'},
                    {'tool': 'query_orders', 'args': {'filter_region': 'US'}, 'desc': 'US orders'},
                    {'tool': 'compute_stats', 'args': {'metric': 'sum'}, 'desc': 'EU revenue'},
                    {'tool': 'format_report', 'args': {'title': 'Regional Analysis'}, 'desc': 'Format'},
                ],
                'risk': 'medium',
                'completeness': 0.85,
            },
        ]
        return candidates

    def score_plan(plan_candidate):
        # Simulate LLM self-evaluation: score based on completeness and step efficiency.
        n_steps = len(plan_candidate['steps'])
        completeness = plan_candidate['completeness']
        efficiency = 1.0 / n_steps   # fewer steps = more efficient
        risk_penalty = {'low': 0.0, 'medium': -0.05, 'high': -0.15}[plan_candidate['risk']]
        score = 0.6 * completeness + 0.3 * efficiency + risk_penalty
        return round(score, 3)

    task = 'Provide a full analysis of our sales data'
    candidates = generate_candidate_plans(task)
    print(f'Tree-of-Thought for: "{task}"\n')
    print(f'{"Plan":35s} {"Steps":6s} {"Score":6s} {"Selected"}')
    print('-' * 60)
    scored = [(c, score_plan(c)) for c in candidates]
    best_plan, best_score = max(scored, key=lambda x: x[1])
    for plan, score in scored:
        marker = '◄ BEST' if plan is best_plan else ''
        print(f'{plan["name"]:35s} {len(plan["steps"]):6d} {score:6.3f} {marker}')

    print(f'\nExecuting best plan: {best_plan["name"]}')
    print(f'Steps: {[s["desc"] for s in best_plan["steps"]]}')
    """),

    md(r"""
    ### 5e Task dependency graph and topological execution
    """),

    code(r"""
    # 5e. Dependency-aware planner: topological sort for parallel-safe execution.

    class DependencyPlanner:
        def __init__(self):
            self.tasks = {}
            self.deps = defaultdict(set)

        def add_task(self, task_id, tool, args, depends_on=None):
            self.tasks[task_id] = {'tool': tool, 'args': args, 'result': None}
            if depends_on:
                for dep in depends_on:
                    self.deps[task_id].add(dep)

        def topo_sort(self):
            # Kahn's algorithm.
            in_degree = {t: 0 for t in self.tasks}
            for t, deps in self.deps.items():
                in_degree[t] = len(deps)
            queue = [t for t, d in in_degree.items() if d == 0]
            order = []
            while queue:
                node = queue.pop(0)
                order.append(node)
                for t in self.tasks:
                    if node in self.deps[t]:
                        in_degree[t] -= 1
                        if in_degree[t] == 0:
                            queue.append(t)
            return order

        def execute(self, tools, verbose=True):
            order = self.topo_sort()
            if verbose:
                print(f'Execution order: {order}')
            for task_id in order:
                task = self.tasks[task_id]
                tool = tools.get(task['tool'])
                # Resolve dependency references.
                resolved = {}
                for k, v in task['args'].items():
                    if isinstance(v, str) and v.startswith('#'):
                        dep_id = v[1:]
                        dep_result = self.tasks.get(dep_id, {}).get('result')
                        if isinstance(dep_result, list):
                            resolved[k] = [r.get('value', 0) for r in dep_result if isinstance(r, dict)]
                        else:
                            resolved[k] = str(dep_result) if dep_result is not None else ''
                    else:
                        resolved[k] = v
                result = tool.call(**resolved) if tool else f'ERROR: unknown tool {task["tool"]}'
                task['result'] = result
                if verbose:
                    print(f'  [{task_id}] {task["tool"]} → {str(result)[:70]}')
            return self.tasks

    dp = DependencyPlanner()
    dp.add_task('T1', 'query_orders', {'filter_region': 'EU'})
    dp.add_task('T2', 'query_orders', {'filter_region': 'US'})
    dp.add_task('T3', 'compute_stats', {'values': '#T1', 'metric': 'sum'}, depends_on=['T1'])
    dp.add_task('T4', 'compute_stats', {'values': '#T2', 'metric': 'sum'}, depends_on=['T2'])
    dp.add_task('T5', 'format_report',
                {'title': 'Regional Revenue', 'summary': 'EU: #T3, US: #T4'},
                depends_on=['T3', 'T4'])
    dp.add_task('T6', 'finish', {'answer': '#T5'}, depends_on=['T5'])

    print('Dependency planner execution:')
    tasks_result = dp.execute(TOOLS_V, verbose=True)
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — ReWOO vs CoT: LLM call count comparison.
    strategies = {
        'Direct action': {'llm_calls': 1, 'tool_calls': 1},
        'CoT (ReAct)': {'llm_calls': 4, 'tool_calls': 4},
        'ReWOO': {'llm_calls': 2, 'tool_calls': 4},
        'Tree-of-Thought (b=3)': {'llm_calls': 5, 'tool_calls': 4},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = list(strategies.keys())
    llm_c = [strategies[n]['llm_calls'] for n in names]
    tool_c = [strategies[n]['tool_calls'] for n in names]

    x = range(len(names))
    axes[0].bar(x, llm_c, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    axes[0].set_ylabel('LLM calls'); axes[0].set_title('Figure 1a — LLM calls per strategy')

    axes[1].bar(x, tool_c, color='seagreen', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel('Tool calls'); axes[1].set_title('Figure 1b — Tool calls per strategy')

    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** LLM call count vs. tool call count for each planning strategy.
    **Direct action** is minimal: 1 LLM call, 1 tool call. **CoT (ReAct)** uses 1
    LLM call per step (thought + action generation) — N steps = N LLM calls. **ReWOO**
    uses only 2 LLM calls (1 for planning, 1 for final answer formatting) regardless
    of plan length — but requires the planner to get the full plan right upfront.
    **Tree-of-Thought** uses more LLM calls (1 per candidate plan + evaluation) but
    selects the best strategy before executing any tools — reducing wasted tool calls
    on suboptimal paths. Key insight: **LLM call cost >> tool call cost** in most
    production systems (LLM costs $10–$50/M tokens; tool calls are $0.01/call). ReWOO
    and ToT reduce LLM calls at the cost of planning sophistication.
    """),

    code(r"""
    # Figure 2 — Dependency graph visualisation.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    node_positions = {
        'T1': (0.1, 0.7), 'T2': (0.1, 0.3),
        'T3': (0.4, 0.7), 'T4': (0.4, 0.3),
        'T5': (0.7, 0.5), 'T6': (0.9, 0.5),
    }
    node_labels = {
        'T1': 'T1\nquery EU', 'T2': 'T2\nquery US',
        'T3': 'T3\nsum(EU)', 'T4': 'T4\nsum(US)',
        'T5': 'T5\nformat', 'T6': 'T6\nfinish',
    }
    colors = ['#d4e6f1', '#d4e6f1', '#d5f5e3', '#d5f5e3', '#fdebd0', '#f9ebea']

    for (tid, pos), color in zip(node_positions.items(), colors):
        ax.add_patch(plt.Circle(pos, 0.07, color=color, ec='gray', lw=1.5, transform=ax.transAxes))
        ax.text(pos[0], pos[1], node_labels[tid], ha='center', va='center', fontsize=8,
                transform=ax.transAxes, fontweight='bold')

    edges = [('T1', 'T3'), ('T2', 'T4'), ('T3', 'T5'), ('T4', 'T5'), ('T5', 'T6')]
    for src, dst in edges:
        sx, sy = node_positions[src]
        dx, dy = node_positions[dst]
        ax.annotate('', xy=(dx - 0.07*(dx-sx)/abs(dx-sx+1e-9), dy),
                    xytext=(sx + 0.07*(dx-sx)/abs(dx-sx+1e-9), sy),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                    xycoords='axes fraction', textcoords='axes fraction')

    ax.text(0.1, 0.95, 'Level 0 (parallel)', ha='center', fontsize=9, style='italic',
            transform=ax.transAxes)
    ax.text(0.4, 0.95, 'Level 1 (parallel)', ha='center', fontsize=9, style='italic',
            transform=ax.transAxes)
    ax.text(0.7, 0.95, 'Level 2', ha='center', fontsize=9, style='italic',
            transform=ax.transAxes)
    ax.set_title('Figure 2 — Task dependency graph (topological levels → parallel execution)')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** Task dependency graph for the regional revenue analysis. Blue nodes
    (T1, T2) are independent — they can run in **parallel** (Level 0). Green nodes
    (T3, T4) depend on T1 and T2 respectively — they can also run in parallel once
    their dependencies complete (Level 1). Orange node T5 depends on both T3 and T4
    — it runs after both Level 1 tasks complete. T6 is sequential last. Total wall-clock
    time = time(T1) + time(T3) + time(T5) + time(T6) (critical path), not the sum of
    all tasks. The topological sort gives the execution order; the level structure
    reveals parallelism opportunities.
    """),

    code(r"""
    # Figure 3 — Tool validation: error rate by validation type.
    categories = ['Missing required\narg', 'Wrong type', 'Unknown tool', 'Tool exception', 'Valid call']
    sim_rates   = [15, 20, 10, 5, 50]   # typical % in unvalidated vs. validated agents

    before_rates = [15, 20, 10, 5, 50]   # without validation (fail silently)
    after_rates  = [5,  3,  2,  5, 85]   # with validation (caught early)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = range(len(categories))
    for ax, (rates, title) in zip(axes, [
        (before_rates, 'Without validation'),
        (after_rates,  'With schema validation'),
    ]):
        colors_b = ['salmon' if r != rates[-1] else 'seagreen' for r in rates]
        ax.bar(x, rates, color=colors_b, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylabel('% of tool calls'); ax.set_title(f'Figure 3 — {title}')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** Tool call error rates before and after argument validation. Without
    validation (left), 50% of calls may fail due to type errors, missing args, or
    unknown tools — often silently (the tool crashes, the agent gets a confusing error
    observation and may loop). With schema validation (right), errors are caught at
    the boundary, the agent receives a structured error message, and can adapt. The
    most impactful validation: **required argument checking** (catches missing args)
    and **type coercion** (catches `limit="all"` when `limit: integer` is required).
    """),

    code(r"""
    # Figure 4 — Tree-of-Thought score distribution.
    task_tot = 'Provide a comprehensive analysis of our sales data'
    candidates_tot = generate_candidate_plans(task_tot)
    scores_tot = [score_plan(c) for c in candidates_tot]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = ['gold' if s == max(scores_tot) else 'steelblue' for s in scores_tot]
    ax.bar([c['name'] for c in candidates_tot], scores_tot, color=colors_bar, alpha=0.8)
    ax.set_ylabel('Plan score (0–1)')
    ax.set_title('Figure 4 — Tree-of-Thought: plan scores (gold = selected)')
    ax.set_ylim(0, 1)
    for i, (c, s) in enumerate(zip(candidates_tot, scores_tot)):
        ax.text(i, s + 0.02, f'{s:.3f}', ha='center', fontsize=10)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** Tree-of-Thought plan scoring. Three candidate plans are scored on
    completeness (does it address all aspects of the task?) and efficiency (fewer
    steps = lower cost). The gold bar is the selected plan. In production, the score
    function is either another LLM call ("Rate this plan 0–10 for addressing the task")
    or a heuristic (step count, tool coverage, estimated cost). For complex tasks,
    generating 5–10 candidates and selecting the best reduces the probability of
    committing to a suboptimal plan early.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **ReWOO over-planning** | Plan references `#E3` before `#E2` resolves | Planner skips dependency check | Validate plan before execution: check all `#E` refs exist |
    | **ToT evaluation bias** | LLM always picks Plan A | Self-evaluation biased toward first plan generated | Temperature > 0 for plan generation; diverse prompts |
    | **Dependency cycle** | Topo sort fails, agent hangs | Task A depends on B and B depends on A | Detect cycles in dependency graph before execution |
    | **Tool argument hallucination** | Valid tool, invalid arg value | LLM invents column names, dates | Validate arg values against allowed enums; use schema examples |
    | **Stale plan** | Plan references data that changed | Long-running plan; data updated mid-execution | Re-validate data freshness before each step in long plans |
    | **Silent tool failure** | Tool returns empty result, agent continues | No error, just empty data | Check result emptiness; treat empty as error if required |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 Anthropic tool use JSON schema format (production standard).
    tool_schema_example = {
        'name': 'query_orders',
        'description': 'Query the orders database with optional filters.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'filter_status': {
                    'type': 'string',
                    'description': 'Filter by order status: completed, pending, cancelled',
                    'enum': ['completed', 'pending', 'cancelled'],
                },
                'filter_region': {
                    'type': 'string',
                    'description': 'Filter by region: EU, US, APAC',
                },
            },
            'required': [],
        }
    }

    print('Anthropic tool use schema:')
    print(json.dumps(tool_schema_example, indent=2))

    openai_schema = {
        'type': 'function',
        'function': {
            'name': 'query_orders',
            'description': 'Query the orders database.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'filter_status': {'type': 'string', 'enum': ['completed', 'pending']},
                    'filter_region': {'type': 'string'},
                },
                'required': [],
            }
        }
    }

    print('\nOpenAI function calling schema:')
    print(json.dumps(openai_schema, indent=2))
    """),

    code(r"""
    # 8.2 LangGraph plan-and-execute pattern (guarded).
    try:
        import langgraph  # noqa: F401
        lines = [
            'from langgraph.prebuilt import create_react_agent',
            'from langgraph.graph import StateGraph',
            '',
            '# LangGraph plan-and-execute workflow:',
            'graph = StateGraph(AgentState)',
            'graph.add_node("planner", plan_step)',
            'graph.add_node("executor", execute_step)',
            'graph.add_node("replan", replan_step)',
            'graph.add_edge("planner", "executor")',
            'graph.add_conditional_edges("executor", should_continue,',
            '    {"continue": "replan", "end": END})',
            'graph.add_edge("replan", "executor")',
            'app = graph.compile()',
            'result = app.invoke({"task": task})',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[langgraph not installed — production pattern]:',
            '  from langgraph.prebuilt import create_react_agent',
            '  # Build a stateful graph with planner + executor nodes.',
            '  # Planner generates a plan; executor runs each step.',
            '  # Conditional edge re-plans if execution fails.',
            '  graph = StateGraph(AgentState)',
            '  graph.add_node("planner", plan_fn)',
            '  graph.add_node("executor", execute_fn)',
            '  app = graph.compile()',
            '  result = app.invoke({"task": task})',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Data Analysis Agent

    **Scenario.** A B2B SaaS company's data team gets 200 ad-hoc SQL analysis
    requests per week ("What are our EU revenue trends this quarter?"). Each takes
    a data analyst 30 minutes on average. They want to automate 60% of these with
    a planning agent.

    **Agent architecture:**
    - **Planner** (ReWOO): given a natural language analysis request, generate a
      plan: `#E1 = get_schema()` → `#E2 = write_sql(#E1, query)` → `#E3 = run_sql(#E2)`
      → `#E4 = analyse_results(#E3)` → `#E5 = generate_report(#E4)`.
    - **Tools**: `get_schema()` (returns DB schema), `write_sql(schema, request)` (LLM
      generates SQL), `run_sql(query)` (executes against read-only replica),
      `analyse_results(df)` (computes stats), `generate_report(analysis)` (formats PDF).

    **Why ReWOO over ReAct?** The plan structure is stable for data analysis — always
    the same 4-5 steps. ReWOO reduces LLM calls from 5 (one per step in ReAct) to
    2 (plan + final format). At 200 requests/week, this saves ~600 LLM calls/week.

    **Results:**
    - 65% of requests handled autonomously (plan succeeds without human review).
    - Average time to report: 45 seconds (vs. 30 minutes human).
    - 35% require human review: complex joins, ambiguous requests, data quality issues.
    - Monthly LLM cost: ~$120 (vs. $14,400 equivalent analyst time).
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Schema validation at tool boundary.** Every tool call must validate types
      and required fields before execution. Return a structured error (not a Python
      exception) so the agent can reason about what went wrong.
    - **Read-only tools first.** Design agents to prefer read-only tools (query,
      look up) over write tools (send email, update DB, delete). Require explicit
      `confirm` step before write operations.
    - **Plan caching.** For repeated task patterns ("weekly EU revenue report"),
      cache the plan structure. Reuse with new date parameters. Reduces planning LLM
      calls to near zero for routine tasks.
    - **Error propagation in plans.** If `#E2` fails, should `#E3` (which depends
      on `#E2`) run? Default: skip dependent steps and surface the error. Alternative:
      use fallback values (empty result) and continue.
    - **Plan audit log.** Store: task, generated plan, execution results per step,
      final answer. Required for debugging, cost attribution, and improvement datasets.
    - **Human approval gates.** For write operations, output the plan and pause:
      `[APPROVE?] The agent will: send email to 5K customers, delete 200 records,
      post to Slack`. Human approves/rejects before execution resumes.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Planning strategy comparison:**

    | Strategy | Upfront planning | Adapts to observations | LLM calls | Best for |
    |---|---|---|---|---|
    | Direct action | None | N/A | 1 | Single-step tasks |
    | CoT / ReAct | Implicit (per step) | Yes | N (steps) | Adaptive multi-step |
    | ReWOO | Full upfront | No | 2 | Stable multi-step, cost-sensitive |
    | Tree-of-Thought | Multiple upfront | No (picks one) | B+1 (B=branches) | High-stakes, multiple valid approaches |

    **Tool use format comparison:**

    | Format | Reliability | Requires | Works without | Use for |
    |---|---|---|---|---|
    | Free-text regex | Low | Text compliance | Native API | Legacy/research only |
    | JSON extraction | Medium | Prompt engineering | Native API | Open-source models |
    | Native function calling | High | OpenAI/Anthropic API | — | **Production default** |
    | Python REPL | Very high | Sandbox | Native API | Code/data agents |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What is the difference between ReAct and ReWOO?"* → ReAct generates one
      thought-action pair at a time, interleaving with tool observations. ReWOO
      generates a complete plan with variable references before executing anything.
      ReWOO uses fewer LLM calls but cannot adapt mid-execution; ReAct adapts but
      uses N LLM calls for N steps.
    - *"Why does Tree-of-Thought help for some tasks but not others?"* → ToT helps
      when: (a) early decisions are hard to reverse, (b) there are multiple qualitatively
      different approaches, (c) the evaluation signal is available before full execution.
      It doesn't help for simple linear tasks where there's only one sensible plan —
      generating alternatives wastes LLM calls.

    **Deep-dive questions**
    - *"How do you validate tool calls in production?"* → Three layers: (1) schema
      validation (types, required fields) before calling the tool — return a structured
      error; (2) business logic validation inside the tool (e.g. SQL injection check);
      (3) result validation (is the output empty, is it the expected type) — trigger
      retry or alternate plan.
    - *"What is the JSON tool schema format and what does it guarantee?"* → The schema
      defines tool name, description, and `input_schema` (JSON Schema object with
      property types, descriptions, enums, and required fields). The LLM API guarantees
      that the output matches the schema — no parsing required. `required: ["query"]`
      guarantees the field is present; `enum: ["EU", "US"]` guarantees the value is valid.

    **Common mistakes:** choosing ReWOO when tool outputs are needed mid-plan (step N
    truly depends on step N-1 observation); not validating tool schemas (agent crashes
    on type error); building ToT without a scoring function (random plan selection);
    exposing write tools without approval gate.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **ReAct vs ReWOO.** What is the fundamental difference? Give a task where ReAct
       is better and one where ReWOO is better.
    2. **Tree-of-Thought.** What are the 3 phases? What is the role of the scoring
       function?
    3. **Dependency graph.** What is a topological sort? When can two tasks run in
       parallel?
    4. **Tool validation layers.** Name 3 layers of tool call validation. At which
       layer does `filter_status=42` (wrong type) get caught?
    5. **JSON tool schema.** What does `"required": ["query"]` guarantee? What does
       `"enum": ["EU", "US"]` prevent?
    6. **ReWOO variable references.** In `#E3 = compute_stats(values=#E2, metric=sum)`,
       what must be resolved before #E3 can execute? What happens if #E2 failed?
    7. **Human approval gate.** Give a concrete example of a write-tool action that
       should require human approval before execution. How do you implement it?
    8. **LLM call cost.** A ReAct agent uses 8 LLM calls for a 7-step task. A ReWOO
       agent uses 2 calls for the same task. If each LLM call costs $0.01 and the
       task runs 1000 times/day, what is the daily cost difference?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Draw the dependency graph for: "Get EU orders, get US orders, compute EU stats,
       compute US stats, compare EU and US, write final report." Which tasks can run
       in parallel? What is the critical path?
    2. An agent uses ReAct for a 10-step task at $0.01/LLM call. A ReWOO agent does
       the same in 2 calls. At 500 tasks/day, what is the monthly cost saving?

    **Beginner → Intermediate (coding)**
    3. Extend `DependencyPlanner` with **cycle detection**: before topological sort,
       detect any cycle (task A → task B → task A) and raise a `PlanError`. Test it
       with a deliberately cyclic plan.
    4. Implement a **plan validator** for ReWOO: before execution, check that all
       `#E` references in step args refer to earlier steps only, and that all tool
       names exist in the registry. Return a list of validation errors.

    **Intermediate (analysis)**
    5. Implement **plan caching**: hash the task description (or its normalised form),
       cache the generated plan. On cache hit, re-execute the cached plan with updated
       date parameters. Measure cache hit rate on 20 similar analysis requests.
    6. Compare the quality of direct-action, CoT, and ReWOO on a benchmark of 10
       analysis tasks (5 simple, 5 complex). Score each: does the agent produce the
       correct answer? Which strategy wins on simple tasks? Which on complex?

    **Senior (design)**
    7. *System design:* design a data analysis agent for a retail company. 500 SQL
       query requests/day. DB has 50 tables. Some queries need 3 joins and 2 aggregation
       steps. Design: planning strategy, tool registry (at least 4 tools), validation
       layer, error handling, human review gate for queries touching sensitive tables.
    8. *Interview:* "We have a ReAct agent that generates a 12-step SQL pipeline.
       We want to cut LLM costs by 60%. What are 3 options and what does each sacrifice?"
       (Expected: ReWOO — loses adaptability; plan caching — loses generalisation;
       smaller planning model — loses plan quality; justify tradeoffs.)
    """),

    md(r"""
    ---
    ### Summary
    Planning separates reasoning about *what to do* from *doing it*. **CoT / ReAct**
    plans one step at a time (adaptive but expensive). **ReWOO** plans all steps
    upfront (efficient but inflexible). **Tree-of-Thought** generates and evaluates
    multiple plans before committing (best quality, highest cost). **Dependency graphs**
    reveal parallelism. **Tool validation** with JSON schemas eliminates a major
    class of agent failures. Always add human approval gates for write operations.

    **Related lesson:** `AGT-03 · Memory Systems` — how agents maintain state across many turns and
    sessions: working memory (context), episodic memory (trajectory logs), semantic
    memory (knowledge base retrieval), and procedural memory (learned behaviours).
    """),
]

build("07_ai_agents/02_planning_and_tool_use.ipynb", cells)
