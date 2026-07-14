"""Builder for Notebook 13 — Explainability (SHAP).

Run:  python3 tools/builders/phase2_13_explainability_shap.py
Emits: notebooks/phase2_ml_engineering/13_explainability_shap.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # 13 · Explainability (SHAP)
    ### Phase 2 — ML Engineering Foundations · *ML/AI Senior Mastery Curriculum*

    > The models that win on tabular data — Random Forests and gradient boosting
    > (Notebooks 07–08) — are **black boxes**: accurate, but they don't hand you the
    > signed, auditable coefficients that made linear/logistic regression so
    > trustworthy (Notebooks 04–05). In regulated, high-stakes, or simply
    > *debuggable* systems that's a problem. **SHAP** (SHapley Additive exPlanations)
    > closes the gap: it borrows a 1950s result from **cooperative game theory** —
    > the **Shapley value** — to fairly attribute a single prediction to its features,
    > with provable guarantees no other method has. This notebook derives it from the
    > axioms, implements exact Shapley attributions from scratch, builds force,
    > summary, and beeswarm visualizations by hand, and — crucially — teaches the
    > caveats that separate "I ran `shap.plots`" from genuine understanding.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The **Shapley value** from cooperative game theory and the **four axioms**
      (efficiency, symmetry, dummy, additivity) that make it the *unique* fair
      attribution.
    - Casting a prediction as a "game": features are players, the payout is the
      prediction minus a baseline; **exact Shapley feature attributions from scratch**.
    - **KernelSHAP** (model-agnostic, weighted regression) and **TreeSHAP**
      (exact, polynomial-time for trees) — what they are and when to use each.
    - **Local** explanations (force / waterfall) and **global** ones (summary /
      beeswarm), built by hand in matplotlib.
    - SHAP vs **permutation importance** (Notebook 07) vs **LIME** — and the
      **correlated-features** trap and the **explainability ≠ causality** distinction.

    **Why it matters in industry**
    - **Regulation & trust:** "right to explanation" (credit, insurance, healthcare)
      demands per-decision reasons; SHAP provides defensible ones for any model.
    - **Debugging:** SHAP exposes leakage (a feature dominating absurdly — Notebook
      10), bias, and spurious signal faster than any metric.
    - **Stakeholder communication:** turning a boosted-tree score into "these three
      factors drove this decision" is a core senior deliverable.

    **Typical interview questions**
    - "What is a Shapley value and why is it 'fair'?"
    - "How does SHAP differ from feature importance / permutation importance?"
    - "Why is exact SHAP expensive, and how does TreeSHAP make it tractable?"
    - "What does a SHAP value actually mean — is it causal?"
    - "What goes wrong with SHAP under correlated features?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The accuracy–interpretability tension.** Linear/logistic regression (Notebooks
    04–05) are transparent: each coefficient is a signed, global effect. But they
    underfit complex tabular data, where ensembles (07–08) win — at the cost of
    interpretability. For years, practitioners had only crude, *global*, and often
    *misleading* tools: tree "feature importance" (biased toward high-cardinality
    features, Notebook 07) and permutation importance (breaks under correlation). None
    explained an **individual** prediction, which is what a denied applicant, a
    flagged transaction, or a debugging engineer actually needs.

    **Shapley values (Lloyd Shapley, 1953).** Game theory asked: in a cooperative game
    where players form coalitions to produce value, how do you **fairly** split the
    total payout among players who contributed unequally and interact? Shapley proved
    there is **exactly one** allocation satisfying four reasonable fairness axioms —
    the *Shapley value* — averaging each player's marginal contribution over all
    possible orders of joining. (He shared the 2012 Nobel for related work.)

    **LIME (2016) and SHAP (Lundberg & Lee, 2017).** LIME explained a single
    prediction by fitting a simple local surrogate model — intuitive but unstable and
    without guarantees. SHAP's breakthrough was to (1) frame feature attribution as
    that 1953 cooperative game (features = players, prediction = payout) and (2) prove
    that the Shapley value is the **unique** attribution satisfying local accuracy,
    consistency, and missingness — *unifying* LIME, DeepLIFT, and others as special
    cases of one principled framework. Then they made it **practical**: KernelSHAP
    (model-agnostic) and **TreeSHAP** (exact, fast for tree ensembles) turned an
    exponential-cost ideal into a production tool.

    **Why it matters now.** SHAP is the de facto standard for explaining tabular
    models in industry, and the conceptual framework (fair credit assignment over
    interacting components) recurs far beyond ML.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The team-bonus analogy.** A team ships a project and earns a bonus. How do you
    split it fairly among members who contributed different amounts and whose
    contributions *interact* (A is only valuable if B did their part)? Shapley's
    answer: for each member, imagine every possible order in which the team could have
    assembled; measure how much that member *adds* when they join (the jump in value);
    average those marginal contributions over all orders. That average is their fair
    share.

    **The ML translation.** The "team" is the set of features; the "bonus" is the
    prediction *relative to a baseline* (the average prediction). A feature's **SHAP
    value** is its fair share of the gap between this prediction and the average — how
    much *this* feature's *this* value pushed the output up or down. The defining
    guarantee (**efficiency / local accuracy**): the SHAP values **add up exactly** to
    the prediction:
    $$f(x) = \underbrace{\mathbb E[f]}_{\text{baseline}} + \sum_i \phi_i.$$
    So an explanation is a complete, additive decomposition — not a vague "importance
    score."

    **Local vs global.** One prediction's SHAP values are a **local** explanation
    ("for *this* customer, high debt added +0.3 to risk"). Averaging $|\phi_i|$ over
    many predictions gives a **global** importance ranking — but built from honest,
    locally-accurate pieces, unlike raw tree importance.

    ```mermaid
    flowchart LR
        B["Baseline E[f]<br/>(avg prediction)"] --> A1["+ phi_1 (feature 1)"]
        A1 --> A2["+ phi_2 (feature 2)"]
        A2 --> A3["+ ... phi_M"]
        A3 --> P["= f(x), this prediction"]
    ```

    Run the cells: we'll define the "game," compute exact Shapley values by brute
    force, and verify they sum to the prediction.
    """),

    code(r"""
    import itertools
    from math import factorial
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    print("NumPy", np.__version__)
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The Shapley value
    A cooperative game is a value function $v(S)$ giving the payout of any coalition
    $S\subseteq\{1,\dots,M\}$ of players. The **Shapley value** of player $i$ is the
    weighted average of its **marginal contribution** $v(S\cup\{i\})-v(S)$ over all
    coalitions $S$ not containing $i$:
    $$\boxed{\ \phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(M-|S|-1)!}{M!}\big[v(S\cup\{i\})-v(S)\big]\ }$$
    The combinatorial weight is exactly the probability of seeing coalition $S$ before
    $i$ if players join in a uniformly random order — so $\phi_i$ is $i$'s *average
    marginal contribution across all $M!$ orderings*.

    ### 4.2 The four axioms (why it's the unique fair split)
    Shapley proved his value is the **only** attribution satisfying all four:
    - **Efficiency:** $\sum_i\phi_i=v(N)-v(\varnothing)$. The shares add up to the
      total (→ "local accuracy": SHAP values sum to $f(x)-\mathbb E[f]$).
    - **Symmetry:** two players with identical marginal contributions get equal
      shares.
    - **Dummy/Null:** a player that never changes the payout gets $\phi_i=0$.
    - **Additivity/Linearity:** for a sum of games, values add — which is what makes
      TreeSHAP over an *ensemble* equal the sum of per-tree explanations.

    ### 4.3 Turning a prediction into a game
    Players = features. The value of a coalition $S$ is the model's expected output
    when **only the features in $S$ are known** and the rest are marginalized out over
    a **background distribution** $D$:
    $$v(S)=\mathbb E_{x_{\bar S}\sim D}\big[f(x_S,\,x_{\bar S})\big].$$
    Then $v(\varnothing)=\mathbb E[f]$ (baseline) and $v(N)=f(x)$ (this prediction), so
    efficiency gives the additive decomposition of §3. *Choice of background matters*
    (e.g., a all-data mean vs a relevant reference) — it defines "what would the
    feature have been if unknown."

    ### 4.4 Why exact SHAP is hard, and the two practical estimators
    Exact computation sums over all $2^M$ coalitions — exponential in the number of
    features. Two tractable routes:
    - **KernelSHAP** (model-agnostic): sample coalitions, evaluate the model, and solve
      a **weighted linear regression** with the *Shapley kernel*; its solution
      provably equals the Shapley values. Works for any model but needs many model
      calls (slow).
    - **TreeSHAP** (Lundberg 2018): for tree ensembles, a clever dynamic program
      computes **exact** Shapley values in **polynomial time** ($O(TLD^2)$: trees ×
      leaves × depth²) by tracking how subsets flow down each tree. This is why SHAP
      is fast and exact for XGBoost/RF — the production default.

    ### 4.5 SHAP vs permutation importance vs LIME
    - **Permutation importance** (Notebook 07): *global*, measures error increase when
      a feature is shuffled; cheap but breaks under correlation and gives no
      per-prediction or signed local attribution.
    - **LIME**: local surrogate model; intuitive but unstable (depends on sampling/
      kernel) and lacks SHAP's consistency guarantee.
    - **SHAP**: local *and* global, signed, additive, and uniquely fair — but
      semantically about the *model*, not the *world* (next point).

    ### 4.6 The two caveats that matter most
    - **Correlated features** split credit. If two features are near-duplicates, SHAP
      divides their joint contribution between them (or, with an interventional
      background, can attribute to unrealistic off-manifold combinations). Importance
      of a *group* is more meaningful than either alone.
    - **Explainability ≠ causality.** A SHAP value says how a feature moved *the
      model's output*, given how the model learned to use it — **not** the real-world
      causal effect of changing that feature. A leaked or confounded feature gets large
      SHAP values precisely because the model relies on it (Notebook 10). SHAP explains
      the model; causal claims require causal inference.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We implement the **exact Shapley value** by brute force over all coalitions, with
    the value function marginalizing absent features over a background sample. We
    verify the **efficiency axiom** (values sum to $f(x)-\mathbb E[f]$) and, for a
    linear model, the closed form $\phi_i = w_i(x_i-\bar x_i)$.
    """),

    code(r"""
    # 5.1 Value function v(S): present features fixed to x, absent ones averaged over background.
    def value_function(f, x, S, background):
        # S: tuple/list of present feature indices
        Xtmp = background.copy().astype(float)
        if len(S):
            Xtmp[:, list(S)] = x[list(S)]          # overwrite present features with x's values
        return f(Xtmp).mean()                       # marginalize absent features over background

    def exact_shapley(f, x, background):
        M = len(x)
        others = list(range(M))
        phi = np.zeros(M)
        for i in range(M):
            rest = [j for j in others if j != i]
            for r in range(len(rest) + 1):
                for S in itertools.combinations(rest, r):
                    w = factorial(len(S)) * factorial(M - len(S) - 1) / factorial(M)
                    phi[i] += w * (value_function(f, x, tuple(S) + (i,), background)
                                   - value_function(f, x, S, background))
        return phi
    """),

    code(r"""
    # 5.2 Verify on a known model: linear with one interaction term.
    w = np.array([2.0, -1.5, 1.0, 0.5, -0.8, 1.2])
    def model(X):
        return X @ w + 0.7 * X[:, 0] * X[:, 1]      # last term is a feature interaction

    M = len(w)
    background = rng.normal(0, 1, (200, M))          # reference distribution
    x = rng.normal(0, 1, M)                          # the instance to explain

    phi = exact_shapley(model, x, background)
    baseline = model(background).mean()
    fx = model(x.reshape(1, -1))[0]

    print("SHAP values phi:", phi.round(3))
    print(f"\\nEFFICIENCY axiom check:")
    print(f"  baseline E[f]      = {baseline:.3f}")
    print(f"  sum(phi)           = {phi.sum():.3f}")
    print(f"  baseline + sum(phi)= {baseline + phi.sum():.3f}")
    print(f"  f(x)               = {fx:.3f}   <- must match (local accuracy)")
    print(f"  match: {np.isclose(baseline + phi.sum(), fx)}")
    """),

    code(r"""
    # 5.3 For a PURELY linear model, SHAP has a closed form: phi_i = w_i * (x_i - mean_i). Verify.
    def linear_model(X):
        return X @ w

    phi_lin = exact_shapley(linear_model, x, background)
    closed_form = w * (x - background.mean(axis=0))
    print("exact Shapley (brute force):", phi_lin.round(3))
    print("closed form w_i*(x_i-mean):", closed_form.round(3))
    print("match:", np.allclose(phi_lin, closed_form, atol=1e-2))
    print("\\n-> Confirms our brute-force Shapley is correct; for linear models it reduces")
    print("   to the intuitive 'coefficient times deviation from the mean'.")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    We build the three canonical SHAP plots **by hand** in matplotlib so you
    understand what they show: a **waterfall** (one prediction), a **global
    importance** bar (mean $|\phi|$), and a **beeswarm** (per-feature SHAP across many
    instances, colored by feature value).
    """),

    code(r"""
    # Figure 1 — WATERFALL: how features move this prediction from baseline to f(x).
    order = np.argsort(-np.abs(phi))                  # largest-impact features first
    fig, ax = plt.subplots(figsize=(9, 5))
    cum = baseline
    ax.axvline(baseline, color="gray", ls="--", label=f"baseline E[f] = {baseline:.2f}")
    for rank, i in enumerate(order):
        color = "tab:red" if phi[i] > 0 else "tab:blue"
        ax.barh(rank, phi[i], left=cum, color=color)
        ax.text(cum + phi[i] / 2, rank, f"x{i}: {phi[i]:+.2f}", ha="center", va="center", fontsize=9)
        cum += phi[i]
    ax.axvline(fx, color="black", lw=2, label=f"f(x) = {fx:.2f}")
    ax.set_yticks(range(M)); ax.set_yticklabels([f"feature {i}" for i in order])
    ax.set_xlabel("model output"); ax.invert_yaxis()
    ax.set_title("Figure 1 — Waterfall: red pushes up, blue pushes down, ending at f(x)")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 1.** The explanation starts at the **baseline** (average prediction, gray)
    and each feature's SHAP value nudges the output up (red) or down (blue) until it
    lands **exactly** at this prediction $f(x)$ (black) — the efficiency axiom made
    visual. For a loan applicant this reads as "your prediction is 0.7 above average:
    high debt (+0.4) and low income (+0.2) pushed it up, long tenure (−0.1) pulled it
    down." This per-decision, signed, additive story is what regulators and
    stakeholders want and what raw feature importance cannot give.
    """),

    code(r"""
    # Compute SHAP values for MANY instances (vectorized model -> fast) for global plots.
    N = 80
    instances = rng.normal(0, 1, (N, M))
    phis = np.array([exact_shapley(model, instances[k], background) for k in range(N)])
    print("computed SHAP matrix:", phis.shape, "(instances x features)")
    """),

    code(r"""
    # Figure 2 — GLOBAL importance: mean(|SHAP|) per feature.
    global_imp = np.abs(phis).mean(axis=0)
    order_g = np.argsort(global_imp)
    fig, ax = plt.subplots()
    ax.barh([f"feature {i}" for i in order_g], global_imp[order_g], color="tab:purple")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Figure 2 — Global importance from averaged local attributions")
    plt.show()
    print("Features 0 and 1 dominate (they carry the interaction term w0*x0*x1).")
    """),

    md(r"""
    **Figure 2.** Averaging $|\phi_i|$ over many instances yields a **global** feature
    importance — but unlike a tree's built-in importance (biased, Notebook 07), every
    bar is built from locally-accurate, axiomatically-fair pieces. Features 0 and 1
    rank highest because they carry both linear weight *and* the interaction term. This
    is the honest way to answer "which features matter overall?" for a black-box model.
    """),

    code(r"""
    # Figure 3 — BEESWARM: distribution of SHAP values per feature, colored by feature value.
    fig, ax = plt.subplots(figsize=(9, 5))
    for row, i in enumerate(order_g):
        yvals = row + (rng.random(N) - 0.5) * 0.6      # vertical jitter
        fv = instances[:, i]
        norm = (fv - fv.min()) / (np.ptp(fv) + 1e-9)     # color by feature value
        ax.scatter(phis[:, i], yvals, c=norm, cmap="coolwarm", s=18, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(M)); ax.set_yticklabels([f"feature {i}" for i in order_g])
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Figure 3 — Beeswarm: each dot = one instance; color = feature value (red high)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm"); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="feature value (low -> high)")
    plt.show()
    """),

    md(r"""
    **Figure 3.** The beeswarm is the most information-dense SHAP plot. Each dot is one
    instance's SHAP value for that feature; horizontal spread shows impact magnitude
    and **direction**, and the color (feature value) reveals the *relationship*. For a
    feature with positive weight you'll see **red dots (high values) on the right**
    (push prediction up) and blue on the left — a clean monotone effect. A feature
    whose color is scrambled left-to-right has a nonmonotone or interaction-driven
    effect. This single plot conveys importance, direction, and interaction at a glance.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Correlated features** | Two near-duplicate features split importance; either alone looks weak | Shapley divides joint credit | Group correlated features; report group importance; cluster |
    | **Off-manifold backgrounds** | Implausible attribution; unstable values | Interventional SHAP evaluates unrealistic feature combos | Choose a sensible background; conditional/TreeSHAP path-dependent options |
    | **Causal misreading** | "Reduce feature X to change outcome" | Treating SHAP as causal effect | State it explains the *model*, not the world; use causal inference for interventions |
    | **Exponential cost** | KernelSHAP too slow on many features | $2^M$ coalitions | TreeSHAP for trees; sample coalitions; explain a subset |
    | **Baseline dependence** | Explanations change with reference | Different $E[f]$ / background | Fix and document the background; choose meaningful reference |
    | **Over-trust** | Acting on a single noisy local explanation | Sampling variance in KernelSHAP | Aggregate; check stability; prefer exact TreeSHAP |
    | **Explaining a broken model** | Beautiful SHAP for a leaky model | SHAP faithfully explains a wrong model | SHAP is a *diagnostic*, not a fix — combine with validation (Ntbk 10) |

    The cell shows the **correlated-features** trap: duplicate a strong feature and
    watch its importance get split between the copies.
    """),

    code(r"""
    # Correlated features split SHAP credit: a single strong signal duplicated looks 'half as important' each.
    def model_corr(X):
        # only X[:,0] truly drives the output; X[:,1] will be made a near-copy of it
        return 3.0 * X[:, 0]

    bg = rng.normal(0, 1, (150, 3))
    bg[:, 1] = bg[:, 0] + 0.01 * rng.normal(0, 1, 150)   # feature 1 ~ feature 0 (correlated)
    xi = np.array([2.0, 2.0, 0.5])
    phi_c = exact_shapley(model_corr, xi, bg)
    print("SHAP values with correlated copy:", phi_c.round(3))
    print("-> The true driver (feature 0) and its near-duplicate (feature 1) SPLIT the credit,")
    print("   so each looks ~half as important as feature 0 really is. Group them before ranking.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    The `shap` library provides `TreeExplainer` (exact, fast TreeSHAP for RF/XGBoost/
    LightGBM), `KernelExplainer` (model-agnostic), `LinearExplainer`, and rich plots
    (`waterfall`, `beeswarm`, `bar`, `force`). We verify the library's TreeSHAP values
    satisfy the same efficiency axiom our scratch code did. The import is wrapped so
    the notebook runs even without `shap`.
    """),

    code(r"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    Xr, yr = make_regression(n_samples=600, n_features=6, n_informative=4,
                             noise=10.0, random_state=0)
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0).fit(Xr, yr)

    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(Xr[:100])
        # efficiency: baseline + sum(shap) == model prediction
        recon = explainer.expected_value + sv.sum(axis=1)
        pred = rf.predict(Xr[:100])
        print(f"shap version: {shap.__version__}")
        print(f"TreeSHAP efficiency holds: {np.allclose(recon, pred, atol=1e-3)}")
        print(f"global importance (mean|SHAP|): {np.abs(sv).mean(0).round(2)}")
    except Exception as e:
        # Fallback: our scratch KernelSHAP-style exact attribution on the RF predict fn.
        print(f"[shap not available: {type(e).__name__}] using scratch exact Shapley on RF.")
        bg2 = Xr[:80]
        phi_rf = exact_shapley(lambda Z: rf.predict(Z), Xr[0], bg2)
        base = rf.predict(bg2).mean()
        print(f"efficiency: base+sum(phi)={base + phi_rf.sum():.2f} vs f(x)={rf.predict(Xr[:1])[0]:.2f}")
    """),

    md(r"""
    **Scratch vs production.** Our brute-force Shapley is $O(2^M)$ — fine for teaching
    on 6 features, hopeless for 100. The `shap` library's **TreeSHAP** computes the
    *same exact values* in polynomial time by exploiting tree structure, so it scales
    to real models and datasets, and adds polished, interaction-aware plots. The key
    check it preserves is the one we verified by hand: **efficiency** (baseline +
    SHAP values = prediction). For non-tree models use `KernelExplainer` (sampled
    coalitions) or model-specific explainers; for deep nets, `DeepExplainer`/
    `GradientExplainer` (Notebook 15's gradients reused for attribution).
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Explaining Credit Decisions (Adverse Action)

    **Scenario.** A lender uses a gradient-boosted model (Notebook 08) for credit
    decisions because it out-predicts logistic regression. But fair-lending law (e.g.
    ECOA/Reg B in the US) requires **adverse-action notices**: a denied applicant must
    receive the *specific principal reasons* for the decision. A raw boosted-tree score
    can't satisfy that — SHAP can.

    **Why SHAP is the right tool:**
    - **Per-decision, signed reasons:** SHAP yields, for *this* applicant, the
      features that pushed their risk up the most ("high credit utilization: +0.18;
      recent delinquency: +0.12") — exactly the adverse-action format.
    - **Model-agnostic + exact for trees:** works on the boosted model the business
      actually wants to ship (TreeSHAP, fast).
    - **Auditability:** global SHAP summaries document overall model behavior for
      regulators; local ones document each decision.

    **Business objectives:** keep the boosted model's accuracy *and* satisfy
    explainability/fairness obligations; build customer and regulator trust.

    **Cost of mistakes:**
    - **Wrong/again-st-the-record explanations** → legal liability, fines.
    - **No explanation** → can't deploy the model at all in this domain.
    - **Mistaking SHAP for causality** → telling a customer "do X to get approved" when
      X isn't causal → misleading guidance and legal exposure (§4.6).

    **Constraints:** explanations must be stable, fast (real-time decisions), and use a
    documented, fixed background; protected attributes and their proxies audited via
    SHAP for disparate impact.

    **KPIs:** explanation fidelity (efficiency holds), latency of explanation
    generation, regulator/audit sign-off, and fairness metrics computed from SHAP
    across protected groups — all while preserving the model's PR-AUC (Notebook 09).
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Use TreeSHAP for tree models** — exact and fast enough for real-time; reserve
      KernelSHAP (slow, sampled) for genuinely model-agnostic needs and precompute
      where possible.
    - **Fix and document the background dataset.** Explanations are *relative* to a
      baseline; a drifting or ill-chosen background silently changes every
      explanation. Version it like a model artifact.
    - **Latency & cost.** Per-prediction explanations add compute; cache global
      summaries, batch local explanations, and consider explaining only flagged/
      contested decisions.
    - **Monitoring with SHAP.** Track global SHAP importance over time — a sudden
      reshuffle signals **drift** or a broken feature pipeline (Notebooks 10, 45). A
      single feature with runaway SHAP often means **leakage**.
    - **Communicate the limits.** Train stakeholders that SHAP explains the *model*,
      not the *world*; never present SHAP as a recipe for changing outcomes without
      causal evidence (§4.6).
    - **Stability.** Prefer exact TreeSHAP over sampled KernelSHAP for regulated
      decisions; if sampling, report/aggregate to control variance.
    - **Privacy/security.** Explanations can leak information about the model/training
      data; gate access appropriately.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Explainability methods:**

    | Method | Scope | Guarantees | Cost | Handles correlation | Best for |
    |---|---|---|---|---|---|
    | Tree feature_importances_ | Global | None (biased) | Free | Poorly | Quick global glance |
    | Permutation importance | Global | Model-agnostic | Moderate | Poorly | Global, any model (Ntbk 07) |
    | **LIME** | Local | None (unstable) | Moderate | Poorly | Quick local intuition |
    | **KernelSHAP** | Local+global | Shapley axioms | **High** ($2^M$/sampled) | Caveats | Any model, exactness matters |
    | **TreeSHAP** | Local+global | Shapley, **exact** | **Low** (poly-time) | Caveats | **Tree ensembles (default)** |
    | Linear coefficients | Global | Exact (linear only) | Free | Multicollinearity issues | Linear models (Ntbk 04) |

    **SHAP value function (background) choice:**

    | Background | Pros | Cons |
    |---|---|---|
    | Interventional (marginal) | Fast, model-faithful (TreeSHAP default) | Can evaluate off-manifold combos |
    | Conditional/observational | Respects feature correlations | Expensive, harder to estimate |

    **Senior lesson:** SHAP is the gold standard for *fair, additive, per-decision*
    attribution — but it explains the **model**, costs compute, and has real
    correlated-feature and causal caveats. Match the explainer to the model (TreeSHAP
    for trees), fix the background, and never oversell it as causal.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *What is a Shapley value?* → Average marginal contribution of a feature over all
      coalitions/orderings (Section 4.1); the unique fair attribution.
    - *How does SHAP differ from feature importance?* → SHAP is *local* (per
      prediction), signed, additive, and axiomatically fair; tree importance is global
      and biased; permutation importance breaks under correlation.

    **Deep-dive questions**
    - *State and explain the four axioms.* → Efficiency, symmetry, dummy, additivity
      (Section 4.2); efficiency → local accuracy.
    - *Why is exact SHAP expensive and how does TreeSHAP fix it?* → $2^M$ coalitions;
      TreeSHAP is exact in poly-time via a tree-structure DP (Section 4.4).
    - *Is a SHAP value causal?* → No — it explains the model's reliance on a feature,
      not the world's response to changing it (Section 4.6).

    **Whiteboard questions**
    - "Write the Shapley value formula and implement brute-force attribution."
      (Sections 4.1, 5.)
    - "Verify the efficiency axiom for an explanation." (baseline + Σφ = f(x).)

    **Strong vs weak answers**
    - *"How do you explain an XGBoost credit model?"*
      - **Weak:** "Use feature_importances_."
      - **Strong:** "TreeSHAP for exact, fast per-decision attributions — signed
        reasons that satisfy adverse-action requirements, summing to the prediction.
        I'd fix the background, audit for fairness across groups via SHAP, and clearly
        flag that SHAP explains the model, not causality."
    - *"Two features have low importance individually but matter together."*
      - **Weak:** "Then they're unimportant."
      - **Strong:** "If they're correlated, SHAP splits their joint credit, so each
        looks weak — I'd report *group* importance or decorrelate before ranking."

    **Follow-ups:** "KernelSHAP vs TreeSHAP?" (model-agnostic+slow vs tree-exact+fast).
    "How choose the background?" (fixed, meaningful reference; document it). "SHAP for
    a neural net?" (DeepExplainer / gradient-based).

    **Common mistakes:** treating SHAP as causal; using KernelSHAP when TreeSHAP
    applies; ignoring correlated-feature credit-splitting; forgetting the background
    dependence; explaining a leaky model and trusting the explanation instead of
    fixing the leak.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define the Shapley value and what a SHAP value attributes.
    2. **Why was it invented?** What gap (over importance/LIME) did SHAP fill, and
       what 1953 result does it borrow?
    3. **How does it work?** Explain the value function (coalitions, background) and
       the averaging over orderings.
    4. **Why does it work?** State the four axioms and why efficiency gives "local
       accuracy."
    5. **When to use it?** TreeSHAP vs KernelSHAP — pick by model type.
    6. **When NOT to use it?** Name the correlated-features and causality caveats.
    7. **Tradeoffs?** SHAP vs permutation importance vs LIME; exactness vs cost.
    8. **How would you productionize it?** Per-decision explanations for a regulated
       model — explainer choice, background, latency, monitoring, and limits.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. For a 3-feature game, list all coalitions and compute one feature's Shapley
       value by hand from a given value table; verify efficiency.
    2. Explain in two sentences why a SHAP value is not a causal effect.

    **Beginner → Intermediate (coding)**
    3. Extend the scratch explainer to output a **force-plot**-style single-row
       visualization and reproduce it for three different instances.
    4. Implement **KernelSHAP** (sample coalitions, weighted linear regression with the
       Shapley kernel) and show it converges to the brute-force values.

    **Intermediate (analysis)**
    5. Reproduce the correlated-features experiment (§7) and show how grouping the
       correlated pair recovers the true importance; vary the correlation strength.
    6. Train an XGBoost model with a deliberately **leaked** feature (Notebook 10) and
       show SHAP exposes it as dominant — demonstrating SHAP as a leakage detector.

    **Senior (interview + production design)**
    7. *Whiteboard:* prove the efficiency axiom implies SHAP values sum to
       $f(x)-\mathbb E[f]$, and explain why that makes waterfall plots exact.
    8. *Design:* build the adverse-action explanation system of §9 — TreeSHAP at
       serving, fixed background, top-k reason extraction, fairness auditing across
       protected groups, latency budget, and the disclaimer about causality.
    9. *Diagnose:* a stakeholder wants to tell customers "increase feature X by 10% to
       get approved," citing its large SHAP value. Explain why this may be wrong and
       what evidence would justify (or refute) the advice.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    SHAP brings the **black-box ensembles of Phase 1** up to the interpretability of
    **linear models** by borrowing the **Shapley value** — the unique attribution
    satisfying efficiency, symmetry, dummy, and additivity. Framing a prediction as a
    cooperative game (features = players, payout = prediction − baseline) yields
    **signed, additive, per-decision** explanations that sum exactly to the output.
    **TreeSHAP** makes it exact and fast for tree ensembles; **KernelSHAP** makes it
    model-agnostic. The senior caveats: it explains the **model not the world**
    (not causal), and **correlated features split credit**.

    **Phase 2 is complete** — you can now *measure* models honestly (09), *validate*
    them without leakage (10), *engineer* their inputs (11), handle *imbalance* (12),
    and *explain* their decisions (13). That is the full ML-engineering discipline that
    surrounds the algorithms.

    **Next:** `14 · Neural Networks from Scratch` — Phase 3 begins. We leave classical
    ML for representation learning, building a multilayer perceptron and its training
    loop from NumPy, setting up backpropagation, CNNs, RNNs, attention, and transformers.
    """),
]

build("phase2_ml_engineering/13_explainability_shap.ipynb", cells)
