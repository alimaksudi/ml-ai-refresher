"""Builder for Lesson NLP-05 — Hallucination and Guardrails."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # NLP-05 · Hallucination and Guardrails
    ### Section 05 — Modern NLP and LLMs · *ML/AI Senior Mastery Curriculum*

    > LLMs generate tokens by maximising $p(y_t | y_{<t}, x)$ — not by checking
    > facts. When the model doesn't know something, it generates the most probable-
    > looking continuation of the prompt, which is often a confident fabrication.
    > This is **hallucination**. This notebook teaches: why it happens (mathematically),
    > how to detect it (consistency checking, NLI, SelfCheckGPT), and how to build
    > **guardrail pipelines** that intercept unsafe inputs and outputs before they
    > reach users.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Hallucination taxonomy**: intrinsic vs extrinsic, closed-domain vs open-domain,
      and why the distinction matters for mitigation strategy.
    - **Root causes**: distribution shift between pre-training data and query, greedy/
      overconfident decoding, and the attention sycophancy problem.
    - **Detection methods**: consistency checking (SelfCheckGPT), NLI-based entailment,
      and retrieval-grounding.
    - **Guardrail pipeline**: input classification (toxic / PII / off-topic), output
      validation (factual, format, safety), and factual grounding via RAG.
    - **Constitutional AI** and the NeMo Guardrails pattern.
    - Implementing a **consistency-checking hallucination detector** and a **rule-based
      guardrail pipeline** from scratch.
    - Visualising hallucination rate vs temperature and how it interacts with model size.

    **Why it matters**
    - Hallucination is the primary production risk in every LLM deployment. A senior
      ML engineer must be able to (1) quantify hallucination rate, (2) build detection,
      and (3) choose the right mitigation for the risk profile.

    **Typical interview questions**
    - "What is hallucination and why does it happen?"
    - "How does SelfCheckGPT detect hallucination without a reference document?"
    - "What is the difference between intrinsic and extrinsic hallucination?"
    - "How do you build a guardrail pipeline for a production LLM?"
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Pre-neural retrieval**: before LLMs, information retrieval systems retrieved
    documents and extracted spans — they couldn't hallucinate because they only returned
    text already in the corpus. The risk was *retrieval failure*, not fabrication.

    **Seq2seq hallucination (2020).** Maynez et al. found that neural abstractive
    summarisation models routinely fabricated facts not in the source document —
    "intrinsic" (contradicting the source) and "extrinsic" (adding facts not in the
    source) hallucinations. Human evaluators rated 30–60% of neural summaries as
    containing unfaithful content.

    **LLM hallucination at scale (2022–2023).** With LLMs capable of generating
    fluent text on any topic, hallucination became a safety and trust crisis: ChatGPT
    fabricated legal citations (the "Mata v. Avianca" case, 2023); medical LLMs
    invented drug interactions. This drove research into detection and mitigation.

    **SelfCheckGPT (Manakul et al., 2023)** introduced the key insight: sample the
    model multiple times on the same prompt. Consistent facts across samples are likely
    real; inconsistent facts are likely hallucinated. No reference document needed.

    **Constitutional AI (Bai et al., 2022, Anthropic)** addressed a related problem:
    training the model to avoid harmful outputs by self-critiquing and revising its own
    answers against a written "constitution" of principles.

    **RAG as grounding (Lewis et al., 2020; Section 06).** Factual grounding via retrieval-
    augmented generation dramatically reduces extrinsic hallucination: the model is
    forced to generate from retrieved evidence rather than from parametric memory.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Why LLMs hallucinate.** The model learns $p(y|x)$ from training data. At
    inference, if $x$ is out-of-distribution (a fact not in pre-training data, or a
    domain-specific entity), the model has no reliable signal — but it still generates
    the most probable-looking continuation. The fundamental problem: the model cannot
    distinguish "I know this" from "I'm generating a plausible-sounding continuation."

    **Hallucination taxonomy.**

    ```mermaid
    flowchart TD
        H["Hallucination"] --> I["Intrinsic\n(contradicts source)"]
        H --> E["Extrinsic\n(adds unsupported facts)"]
        I --> CI["Closed-domain\n(source given in context)\nFix: faithfulness training, NLI check"]
        E --> OD["Open-domain\n(no source, relies on parametric memory)\nFix: RAG grounding, self-check"]
    ```

    **SelfCheckGPT intuition.** If you ask the model "What year was X born?" and sample
    it 5 times, responses for true facts will be consistent (all say 1965); responses
    for hallucinated facts will disagree (1963, 1967, 1965, 1968, 1964). Consistency =
    reliability. Inconsistency = likely hallucination.

    **Guardrail pipeline intuition.** A guardrail is a filter at the boundary of the
    LLM — intercept before (input guard) and after (output guard). Input guards catch
    toxic queries, PII leakage, and off-topic requests. Output guards check factual
    grounding, format validity, and safety.

    ```mermaid
    flowchart LR
        U["user query"] --> IG["Input Guard\n(toxic/PII/off-topic)"]
        IG -->|safe| LLM["LLM\n+ RAG context"]
        IG -->|blocked| R1["rejection message"]
        LLM --> OG["Output Guard\n(factual/format/safety)"]
        OG -->|pass| U2["user response"]
        OG -->|fail| R2["fallback/retry"]
    ```
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from collections import Counter

    rng = np.random.default_rng(42)
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # Known facts and hallucinated facts for our simulation.
    TRUE_FACTS = [
        "The Eiffel Tower is located in Paris.",
        "Python is a programming language.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits the Sun.",
        "Shakespeare wrote Hamlet.",
    ]
    HALLUCINATED = [
        "The Eiffel Tower was built in 1756.",
        "Python was invented in 1985.",
        "Water boils at 80 degrees Celsius at sea level.",
        "The Moon is 1000 km from Earth.",
        "Shakespeare wrote Don Quixote.",
    ]
    print(f"True facts: {len(TRUE_FACTS)}, Hallucinated facts: {len(HALLUCINATED)}")
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Why hallucination is probabilistic

    At inference, the model samples:
    $$y_t \sim p_\theta(y_t \mid y_{<t}, x)$$
    This is a probability distribution — not a fact lookup. When the query concerns
    an entity $e$ that appeared rarely in pre-training, $p_\theta$ is uncertain:
    the model spreads probability over many plausible answers. Greedy or low-
    temperature sampling picks the mode, which may be systematically wrong.

    **Key insight:** the model *cannot* signal uncertainty through this distribution
    unless it was explicitly trained to do so (via RLHF with "I don't know" as a
    preferred response for uncertain queries). By default, all outputs are equally
    "confident" in token probability terms.

    ### 4.2 SelfCheckGPT consistency score

    Sample $n$ responses $\{r_1, \dots, r_n\}$ from the same prompt. For a claim
    $c$ in $r_1$, compute the fraction of other samples that support $c$:
    $$\text{consistency}(c) = \frac{1}{n-1}\sum_{i=2}^n \mathbf{1}[c \text{ supported by } r_i]$$
    A claim with consistency $> 0.7$ across $n=5$ samples is likely factual.
    Consistency $< 0.4$ is a strong hallucination signal.

    Support can be measured by:
    - **NLI** (natural language inference): does $r_i$ entail $c$?
    - **n-gram overlap** (BERTScore, ROUGE-like): how similar are the claim and
      the sampled response token-by-token?
    - **QA-based**: generate a question from $c$, answer it from each $r_i$, check
      if answers match.

    ### 4.3 NLI-based factual grounding

    Given a claim $c$ and a reference passage $p$ (e.g., retrieved by RAG), classify
    the relation:
    $$\text{NLI}(c, p) \in \{\text{ENTAIL}, \text{NEUTRAL}, \text{CONTRADICT}\}$$
    An output with high **CONTRADICT** probability is hallucinated relative to the
    source. An output with high **NEUTRAL** probability added extrinsic facts not in
    the source (extrinsic hallucination). Only **ENTAIL** is fully grounded.

    ### 4.4 Hallucination rate

    $$\text{HR} = \frac{\text{\# responses containing hallucination}}{\text{total responses}}$$
    Empirically measured on a benchmark (TruthfulQA, HaluEval, FEVER) using: human
    annotation, NLI classifiers, or factual lookup against a knowledge base.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    ### 5a Consistency-checking hallucination detector (SelfCheckGPT-style)
    """),

    code(r"""
    # 5a. Simulate N LLM samples and measure consistency.
    # We use token-level n-gram similarity as a proxy for NLI entailment.

    def ngram_overlap(text1, text2, n=2):
        def ngrams(text, n):
            tokens = re.findall(r'\w+', text.lower())
            return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()
        a, b = ngrams(text1, n), ngrams(text2, n)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)   # Jaccard similarity

    def simulate_llm_samples(fact, is_true, n_samples=5, temperature=1.0, seed=0):
        rng2 = np.random.default_rng(seed)
        samples = []
        for i in range(n_samples):
            if is_true:
                # True facts: model tends to agree (with noise at high temperature)
                noise_prob = 0.1 * temperature
                if rng2.random() < noise_prob:
                    # Occasionally introduce a variation
                    words = fact.split()
                    idx = rng2.integers(0, len(words))
                    words[idx] = rng2.choice(['approximately', 'roughly', 'nearly'])
                    samples.append(' '.join(words))
                else:
                    samples.append(fact)
            else:
                # Hallucinated facts: model disagrees between samples
                words = fact.split()
                n_changes = max(1, int(rng2.normal(2, 1) * temperature))
                for _ in range(n_changes):
                    if len(words) > 2:
                        idx = rng2.integers(0, len(words))
                        replacements = ['not', 'possibly', 'approximately', 'rarely', 'usually']
                        words[idx] = rng2.choice(replacements)
                samples.append(' '.join(words))
        return samples

    def consistency_score(samples):
        if len(samples) < 2:
            return 1.0
        ref = samples[0]
        scores = [ngram_overlap(ref, s) for s in samples[1:]]
        return float(np.mean(scores))

    print("Consistency scores (higher = more reliable):")
    print(f"{'Fact':<50} {'True?':>5} {'Consistency':>12}")
    print('-' * 70)
    for fact, is_true in list(zip(TRUE_FACTS, [True]*5)) + list(zip(HALLUCINATED, [False]*5)):
        samples = simulate_llm_samples(fact, is_true, n_samples=7, temperature=0.8)
        cscore = consistency_score(samples)
        flag = 'OK' if (cscore > 0.6) == is_true else 'MISSED'
        print(f"{fact[:48]:<50} {str(is_true):>5} {cscore:>10.3f}  {flag}")
    """),

    md(r"""
    ### 5b Rule-based guardrail pipeline from scratch
    """),

    code(r"""
    # 5b. Rule-based guardrail pipeline.
    # Production systems combine rule-based + ML classifiers; we demonstrate rules here.

    import re as re_module

    # Patterns for common guardrail checks.
    TOXIC_PATTERNS = [
        r'\b(kill|murder|attack|hack|bomb|weapon)\b',
        r'\b(racist|sexist|slur)\b',
    ]
    PII_PATTERNS = [
        r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',   # SSN
        r'\b\d{16}\b',                             # credit card (simplified)
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # phone
    ]
    OFF_TOPIC_KEYWORDS = ['casino', 'lottery', 'gambling', 'bet on']

    def input_guard(text, allowed_topics=None):
        text_lower = text.lower()
        for pat in TOXIC_PATTERNS:
            if re_module.search(pat, text_lower):
                return False, 'TOXIC'
        for pat in PII_PATTERNS:
            if re_module.search(pat, text):
                return False, 'PII_DETECTED'
        for kw in OFF_TOPIC_KEYWORDS:
            if kw in text_lower:
                return False, 'OFF_TOPIC'
        return True, 'PASS'

    def output_guard(response, context_docs=None, min_length=5, max_length=2000):
        if len(response.split()) < min_length:
            return False, 'TOO_SHORT'
        if len(response.split()) > max_length:
            return False, 'TOO_LONG'
        # Check factual grounding: if context provided, require some overlap.
        if context_docs:
            context_text = ' '.join(context_docs).lower()
            response_words = set(re_module.findall(r'\w+', response.lower()))
            context_words  = set(re_module.findall(r'\w+', context_text))
            overlap = len(response_words & context_words) / (len(response_words) + 1)
            if overlap < 0.15:
                return False, 'LOW_GROUNDING'
        # Toxic content in output.
        for pat in TOXIC_PATTERNS:
            if re_module.search(pat, response.lower()):
                return False, 'TOXIC_OUTPUT'
        return True, 'PASS'

    # Test the guardrail pipeline.
    test_cases = [
        ("How do I improve my Python skills?", True),
        ("My SSN is 123-45-6789, help me reset my account", False),
        ("I want to attack this codebase", False),
        ("What is the capital of France?", True),
        ("Best casino bonuses and gambling tips", False),
    ]
    print("Input guardrail results:")
    for text, expected_pass in test_cases:
        passed, reason = input_guard(text)
        status = 'PASS' if passed else 'BLOCK'
        match = 'OK' if passed == expected_pass else 'WRONG'
        print(f"  [{status}] ({reason}) '{text[:45]}'  {match}")
    """),

    code(r"""
    # 5b.2 Output guardrail with grounding check.
    context = ["Paris is the capital of France. It is located in northern France."]
    output_tests = [
        ("Paris is the capital of France, located in northern France.", True),
        ("Paris is the capital of Germany.", True),         # short but not grounded
        ("Yes.", False),                                    # too short
        ("The Eiffel Tower in Paris was built in 1889.", True),
    ]
    print("Output guardrail results (with context):")
    for resp, _ in output_tests:
        passed, reason = output_guard(resp, context_docs=context)
        print(f"  [{'PASS' if passed else 'BLOCK'}] ({reason}) '{resp[:55]}'")
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Hallucination rate vs temperature.
    # Simulate: at low temperature, model is consistent (low hallucination);
    # at high temperature, model is creative but inconsistent (high hallucination).
    temperatures = np.linspace(0.1, 2.0, 30)
    hallucination_rates = []
    for T in temperatures:
        rates = []
        for fact, is_true in list(zip(TRUE_FACTS, [True]*5)) + list(zip(HALLUCINATED, [False]*5)):
            samples = simulate_llm_samples(fact, is_true, n_samples=5, temperature=T,
                                           seed=int(T*100))
            cscore = consistency_score(samples)
            # "Hallucinated" if: true fact with low consistency, OR halluci with any score
            if is_true:
                hallu = cscore < 0.5
            else:
                hallu = True   # hallucinated facts are always "hallucinations"
            rates.append(hallu)
        hallucination_rates.append(np.mean(rates))

    fig, ax = plt.subplots()
    ax.plot(temperatures, hallucination_rates, 'o-', color='#d62728')
    ax.set_xlabel("temperature"); ax.set_ylabel("hallucination rate (fraction of facts)")
    ax.set_title("Figure 1 — Hallucination rate vs temperature (simulated)")
    ax.axvline(1.0, ls='--', color='gray', alpha=0.6, label='T=1.0 (default)')
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 1.** Simulated hallucination rate as a function of decoding temperature.
    At low temperature ($T < 0.5$), the model is conservative — it picks the most
    probable tokens, which for well-known facts tends to be correct. At high temperature
    ($T > 1.5$), the model explores more of the probability distribution, increasing
    diversity but also increasing the chance of generating incorrect tokens. The
    production lesson: **use $T < 0.7$ for factual tasks** (extraction, Q&A) and
    $T \in [0.7, 1.0]$ for creative tasks. Never use $T > 1.0$ in production unless
    you have specific reason (brainstorming) and output guardrails.
    """),

    code(r"""
    # Figure 2 — SelfCheckGPT: consistency score distribution for true vs hallucinated facts.
    n_samples = 7
    true_scores, hallu_scores = [], []
    for seed in range(50):
        for i, fact in enumerate(TRUE_FACTS):
            samps = simulate_llm_samples(fact, True, n_samples, temperature=0.9, seed=seed*10+i)
            true_scores.append(consistency_score(samps))
        for i, fact in enumerate(HALLUCINATED):
            samps = simulate_llm_samples(fact, False, n_samples, temperature=0.9, seed=seed*10+i)
            hallu_scores.append(consistency_score(samps))

    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 25)
    ax.hist(true_scores, bins=bins, alpha=0.6, label='True facts', color='#2ca02c')
    ax.hist(hallu_scores, bins=bins, alpha=0.6, label='Hallucinations', color='#d62728')
    ax.axvline(0.6, ls='--', color='black', label='Threshold (0.60)')
    ax.set_xlabel("consistency score (n-gram Jaccard, N=7 samples)")
    ax.set_ylabel("count"); ax.legend()
    ax.set_title("Figure 2 — SelfCheckGPT consistency: true vs hallucinated facts")
    plt.show()
    # Report detection performance.
    threshold = 0.60
    tp = sum(s < threshold for s in hallu_scores)
    fp = sum(s < threshold for s in true_scores)
    tn = sum(s >= threshold for s in true_scores)
    fn = sum(s >= threshold for s in hallu_scores)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    print(f"Hallucination detection at threshold={threshold}:")
    print(f"  Precision: {precision:.2f}  Recall: {recall:.2f}")
    """),

    md(r"""
    **Figure 2.** Consistency score distributions for true facts (green) and
    hallucinations (red) under SelfCheckGPT with $n=7$ samples. True facts cluster
    near 1.0 (high consistency across samples); hallucinations cluster near 0.0–0.4
    (samples disagree). A threshold of 0.60 separates them reasonably well. The
    overlap in the 0.4–0.7 range is the hard zone — ambiguous facts or facts that the
    model "partially knows." In practice, NLI-based SelfCheckGPT achieves precision and
    recall of 0.75–0.85 on standard benchmarks, better than the simple n-gram version
    demonstrated here.
    """),

    code(r"""
    # Figure 3 — Guardrail pipeline block rate by category.
    test_inputs = [
        ("How do I improve my Python skills?", "valid"),
        ("What is machine learning?", "valid"),
        ("Explain neural networks to me", "valid"),
        ("My email is user@example.com, fix my account", "pii"),
        ("My SSN is 123-45-6789", "pii"),
        ("I want to attack the server", "toxic"),
        ("Help me build a weapon", "toxic"),
        ("Best gambling sites to bet on sports", "off_topic"),
        ("Casino bonus codes for blackjack", "off_topic"),
        ("How do neural networks learn?", "valid"),
    ]
    results = Counter()
    for text, category in test_inputs:
        passed, reason = input_guard(text)
        results[reason] += 1

    reasons = list(results.keys())
    counts = [results[r] for r in reasons]
    fig, ax = plt.subplots()
    colors = ['#2ca02c' if r == 'PASS' else '#d62728' for r in reasons]
    ax.bar(reasons, counts, color=colors)
    ax.set_xlabel("Guard result"); ax.set_ylabel("count")
    ax.set_title("Figure 3 — Input guardrail block rate by category")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** The input guardrail blocks requests by category: toxic content,
    PII detection, and off-topic queries. Allowed requests (PASS, green) proceed to
    the LLM. The rule-based system here is fast (microseconds) and has zero false
    positives on our test set — but it can miss adversarial rewrites ("How do I
    att@ck a server?"). Production systems add an ML classifier (DistilBERT fine-tuned
    on toxic content) as a second layer for higher recall.
    """),

    code(r"""
    # Figure 4 — Grounding: hallucination rate with vs without RAG context.
    # Simulate: RAG forces the model to attend to retrieved evidence.
    n_trials = 200
    no_rag_rates = []
    with_rag_rates = []
    for seed in range(n_trials):
        rng3 = np.random.default_rng(seed)
        # Without RAG: ~40% hallucination on domain-specific questions
        no_rag_rates.append(rng3.random() < 0.40)
        # With RAG: ~10% hallucination (model may still ignore context)
        with_rag_rates.append(rng3.random() < 0.10)

    labels_rag = ['Without RAG', 'With RAG\n(factual grounding)']
    rates_rag = [np.mean(no_rag_rates)*100, np.mean(with_rag_rates)*100]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels_rag, rates_rag, color=['#d62728', '#2ca02c'])
    ax.set_ylabel("Hallucination rate (%)"); ax.set_ylim(0, 60)
    ax.set_title("Figure 4 — RAG grounding reduces hallucination rate (simulated)")
    for bar, rate in zip(bars, rates_rag):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 1, f"{rate:.0f}%",
                ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** The dramatic effect of RAG grounding on hallucination rate.
    Without context, the model relies purely on parametric memory — which may be
    outdated, domain-specific, or simply wrong. With retrieved context, the model can
    condition on evidence, reducing hallucination by 3–4× in practice (Lewis et al.,
    2020; multiple RAG papers confirm 50–80% reduction). The remaining ~10% comes from
    the model ignoring context ("context faithfulness failure") or the retrieved context
    itself being incorrect. Section 06 (RAG-01 through RAG-08) addresses both.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Intrinsic hallucination** | Output contradicts provided source | Model ignores context; attention dilution | Explicit instruction to cite source; NLI output guard |
    | **Extrinsic hallucination** | Output adds facts not in source | Parametric memory overrides context | RAG (Section 06); SelfCheckGPT detection |
    | **Guardrail over-blocking** | Legitimate queries blocked | Overly broad rules / classifier false positives | Precision-tuned classifiers; user appeal path |
    | **Guardrail under-blocking** | Harmful content passes | Adversarial rewrite bypasses rules | ML classifier + rule combination; red-teaming |
    | **Sycophancy** | LLM agrees with incorrect user assertion | RLHF training on user approval → model pleases | Adversarial training; "I disagree" examples in RLHF |
    | **Calibration failure** | High-confidence wrong answers | Training didn't reward uncertainty | Verbalize-uncertainty training; temperature calibration |
    | **RAG hallucination** | Model hallucinates even with context | Relevant chunks not retrieved (Section 06); model ignores chunks | Improve retrieval (RAG-01); faithfulness-trained model |
    | **Prompt-injection via retrieved docs** | Retrieved malicious document hijacks LLM | Retrieved text treated as trusted | Separate retrieved vs instruction delimiters; input classify retrieved text |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 SelfCheckGPT with actual LLM (guarded).
    try:
        from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
        device = "cpu"
        selfcheck = SelfCheckNLI(device=device)
        print("SelfCheckGPT available:")
        print("  selfcheck.predict(sentences, sampled_passages, num_samples=5)")
        print("  Returns per-sentence hallucination probability.")
    except Exception as e:
        print(f"[selfcheckgpt not available: {type(e).__name__}]")
        lines = [
            "SelfCheckGPT production pattern:",
            "  from selfcheckgpt.modeling_selfcheck import SelfCheckNLI",
            "  selfcheck = SelfCheckNLI(device='cuda')",
            "  sentences = passage.split('.')",
            "  samples = [llm(prompt) for _ in range(5)]  # sample N=5",
            "  scores = selfcheck.predict(sentences, samples, num_samples=5)",
            "  # scores[i] = P(hallucinated) for sentence i",
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 NeMo Guardrails pattern (guarded).
    lines = [
        "NeMo Guardrails production pattern (requires nemoguardrails):",
        "  from nemoguardrails import RailsConfig, LLMRails",
        "  config = RailsConfig.from_path('./config')",
        "  rails = LLMRails(config)",
        "  response = await rails.generate_async(messages=[",
        "      {'role': 'user', 'content': user_query}",
        "  ])",
        "",
        "Config defines:",
        "  - 'define rails' blocks for input/output topical rails",
        "  - Custom flows: if user says X, do Y",
        "  - Jailbreak detection: canonical patterns",
        "  - Factual grounding: mandatory RAG before sensitive claims",
    ]
    print('\n'.join(lines))
    """),

    code(r"""
    # 8.3 Instructor library for structured output with hallucination guardrails.
    try:
        import instructor
        from pydantic import BaseModel, field_validator

        class FactualAnswer(BaseModel):
            answer: str
            confidence: float
            sources_cited: list

            @field_validator('confidence')
            def confidence_range(cls, v):
                if not 0 <= v <= 1:
                    raise ValueError('confidence must be in [0,1]')
                return v

        print("instructor + pydantic guardrail pattern:")
        print("  import instructor")
        print("  client = instructor.patch(openai.OpenAI())")
        print("  resp = client.chat.completions.create(")
        print("      model='gpt-4o', response_model=FactualAnswer, ...)")
        print("  # Validation error raised if model outputs invalid structure")
        # Demo with synthetic data.
        fake = FactualAnswer(answer="Paris", confidence=0.95, sources_cited=["Wikipedia"])
        print(f"  Example response: {fake.model_dump()}")
    except Exception as e:
        print(f"[instructor/pydantic not available: {type(e).__name__}]")
        print("instructor forces the LLM to output valid JSON matching the Pydantic schema.")
        print("Validation failure triggers an automatic retry (up to 3 times).")
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Medical Q&A with Hallucination Guardrails

    **Scenario.** A healthcare AI startup deploys an LLM to answer patient questions
    about medications. Hallucinated drug dosages or interaction warnings pose direct
    patient harm risk.

    **Hallucination risk profile:**
    - Drug names, dosages, and interactions are highly specific — a model trained on
      general web text may have incorrect or outdated information.
    - Even a 5% hallucination rate means 1 in 20 answers is potentially harmful.

    **Mitigation stack (in layers):**
    1. **RAG grounding (Section 06)**: retrieve from a curated, version-locked drug
       database (FDA labels, UpToDate). Model must cite retrieved passage.
    2. **NLI output guard**: check that every factual claim in the response is entailed
       by the retrieved passage. Block if CONTRADICT detected.
    3. **SelfCheckGPT**: sample $N=3$ responses; block any claim that is not consistent
       across samples (consistency < 0.7).
    4. **Structured output**: force the model to output `{answer, citations, confidence}`;
       refuse to send responses with `confidence < 0.8` to the user without a human
       review flag.
    5. **Human escalation**: all `confidence < 0.6` responses and all NLI-flagged
       responses are escalated to a pharmacist queue.

    **KPIs:** hallucination rate on a gold medical QA set (target <2%), citation
    rate (>95%), escalation rate (proxy for model confidence), p99 latency
    (RAG + SelfCheckGPT adds ~2s; target <5s).

    **Cost of mistakes:** FDA Class II medical device liability; clinical negligence
    claims. The cost of a false positive (over-blocking a safe answer) is much lower
    than a false negative (allowing a hallucinated drug dosage through).
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Guardrail latency.** Rule-based input guards add <1ms. ML classifiers (Bert-
      based) add 20–50ms. SelfCheckGPT with $N=3$ calls triples LLM cost and latency.
      Choose layers based on risk profile and latency budget.
    - **Threshold tuning.** Consistency score and NLI thresholds must be calibrated on
      a domain-specific evaluation set — thresholds that work for general QA may not
      work for medical or legal text where precision is paramount.
    - **Escape rate monitoring.** Continuously sample production outputs and run an
      independent evaluator (LLM-as-judge, Lesson EVAL-05) to estimate live hallucination
      rate. Alert if it exceeds a SLA threshold (Lesson PROD-05).
    - **RAG as the primary mitigation.** For factual domains, RAG grounding (Section 06)
      gives the largest reduction in hallucination at moderate cost. SelfCheckGPT and
      NLI guards are secondary checks that catch what RAG misses.
    - **Constitutional AI / RLAIF.** At training time, fine-tune the model to output
      "I don't know" or "I'm not certain" on low-confidence claims. This requires:
      collecting examples where the model should express uncertainty (hard) and
      including them in RLHF/DPO training (Lesson NLP-03).
    - **Adversarial red-teaming.** Regularly attack your own guardrail pipeline with
      adversarial prompts (prompt injection, adversarial rewrites, jailbreaks). Fix
      gaps before attackers find them.
    - **Logging.** Log all blocked queries and the block reason. Review weekly for:
      (1) false positives (legitimate queries blocked); (2) novel attack patterns not
      covered by current rules.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Hallucination mitigation strategies:**

    | Strategy | Hallucination reduction | Latency overhead | Cost overhead | Limitations |
    |---|---|---|---|---|
    | RAG grounding | High (50–80%) | +100–500ms | Low | Requires good retrieval; context window |
    | Temperature reduction | Medium (20–40%) | None | None | Reduces diversity; may increase refusals |
    | SelfCheckGPT (N=5) | High (60–70%) | +5x LLM cost | High | Only detects, doesn't correct |
    | NLI output guard | Medium | +50ms (classifier) | Low | Requires reference text |
    | Constitutional AI (training) | Medium-high | None (at inference) | High (training) | One-time cost |
    | Human review | Near-100% | High (minutes) | Very high | Only feasible for low-volume |

    **Guardrail trade-off: precision vs recall:**

    | Priority | Setting | Consequence |
    |---|---|---|
    | High precision (few false alarms) | High threshold | May miss harmful outputs |
    | High recall (catch everything harmful) | Low threshold | More false positives → user friction |
    | **Medical / legal** | **High recall** | Over-blocking is acceptable; missing harmful is not |
    | **Consumer chatbot** | **Balanced** | User experience matters; acceptable risk level is higher |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What is hallucination and why does it happen?"* → Hallucination is when the
      LLM generates confident text that is factually incorrect or not grounded in the
      provided context. It happens because the model generates the most probable-
      sounding token, not the factually correct one — it has no fact-lookup mechanism.
      For out-of-distribution or rare facts, the probability distribution is uncertain
      and the model generates plausible-sounding but wrong continuations.
    - *"How does SelfCheckGPT detect hallucination without a reference?"* → It exploits
      the observation that the model is more consistent when recalling real facts
      (training data has many supporting sentences) and inconsistent when generating
      hallucinations (the probability mass is diffuse). Sample the model $N$ times on
      the same prompt; facts with high cross-sample consistency are likely real; facts
      with low consistency are likely hallucinated.

    **Deep-dive questions**
    - *"What's the difference between intrinsic and extrinsic hallucination?"* →
      Intrinsic: the output contradicts the provided source document. Extrinsic: the
      output adds facts not in the source (may be true or false — the point is they're
      not grounded in the given context). Different mitigations: intrinsic → faithfulness
      training, NLI guards; extrinsic → RAG, citation enforcement.
    - *"How do you build a production guardrail pipeline?"* → Three layers: (1) **input
      guard** (rule-based + ML classifier: toxic, PII, off-topic); (2) **LLM + RAG**
      (factual grounding reduces hallucination at the source); (3) **output guard**
      (NLI check against context, schema validation, SelfCheckGPT for high-risk claims,
      human escalation for low-confidence). Monitor escape rate in production.

    **Whiteboard questions**
    - "Design a guardrail pipeline for a medical Q&A system." (§9, §5b)
    - "What is the SelfCheckGPT algorithm? Describe the consistency metric." (§4.2, §5a)

    **Strong vs weak answers**
    - *"How do we reduce hallucination in our LLM-powered product?"*
      - **Weak:** "Use a bigger model or set temperature to 0."
      - **Strong:** "Start with RAG grounding (Section 06) — it gives the largest
        reduction at manageable cost. Add NLI-based output grounding checks for claims
        against retrieved context. For high-risk claims, add SelfCheckGPT (N=3).
        Temperature 0.3–0.5 for factual tasks. Monitor hallucination rate with an
        automatic evaluator (Lesson EVAL-05) and set a SLA threshold with alerts. Fine-
        tune on 'I don't know' examples via DPO for cases where the model should
        express uncertainty."

    **Common mistakes:** confusing hallucination with bias; thinking temperature=0
    eliminates hallucination (it reduces it but doesn't eliminate it); not knowing the
    intrinsic/extrinsic taxonomy; conflating SelfCheckGPT (sampling-based) with NLI
    (entailment-based).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Define hallucination.** Name the two types in the intrinsic/extrinsic taxonomy.
    2. **Why does it happen?** Describe the LLM's generation mechanism and why it can't
       "know" it's wrong.
    3. **SelfCheckGPT.** Describe the algorithm. What is the consistency metric? What
       does low consistency mean?
    4. **NLI grounding.** What are the three NLI labels? Which one indicates
       hallucination relative to a source document?
    5. **Guardrail pipeline.** Name three input guard checks and two output guard checks
       from §5b.
    6. **RAG effect.** Why does RAG reduce hallucination? What failure mode remains?
    7. **Temperature tradeoff.** Why does lower temperature reduce hallucination? What
       does it sacrifice?
    8. **Medical scenario.** In the §9 system, why is high recall (blocking more)
       preferred over high precision (blocking less)?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Classify each as intrinsic or extrinsic hallucination: (a) A summary that says
       "the CEO resigned" when the article says "the CEO stepped down temporarily"; (b)
       A biography that says a politician "won the 2016 Nobel Prize" when the source
       article never mentions any prize.
    2. Why does temperature=0 (greedy decoding) not fully eliminate hallucination?
       Give a concrete example where greedy decoding would hallucinate.

    **Beginner → Intermediate (coding)**
    3. Extend the `input_guard` function with an email-domain allowlist: only allow
       emails from `@company.com` domains. Test it on both legitimate and rejected
       addresses.
    4. Implement a BERTScore-style consistency metric using TF-IDF cosine similarity
       instead of n-gram Jaccard: embed each sample response as a TF-IDF vector
       (using `sklearn.TfidfVectorizer`), compute cosine similarity between the first
       sample and each subsequent one, and report the mean as the consistency score.

    **Intermediate (analysis)**
    5. Implement the full SelfCheckGPT NLI-based variant: given a claim sentence $c$
       and $N$ sampled passages $\{r_i\}$, use `sklearn`'s logistic regression trained
       on synthetic entailment/contradiction pairs (word overlap as feature) to predict
       the NLI label for each $(c, r_i)$ pair. Report precision/recall of hallucination
       detection.
    6. Build an end-to-end hallucination detection pipeline: (1) input guard, (2) LLM
       call (simulated), (3) consistency check with $N=5$ samples, (4) output guard.
       Test on 10 prompts (5 factual, 5 hallucination-prone) and report the overall
       detection rate and false positive rate.

    **Senior (interview + production design)**
    7. *Design:* the hallucination monitoring system for the medical Q&A product in
       §9. Include: automatic evaluator model selection (Lesson EVAL-05), sampling rate,
       alert thresholds, escalation workflow, retraining trigger conditions (Notebook
       46), and dashboarding (Lesson PROD-05).
    8. *Cost-quality tradeoff:* you can run SelfCheckGPT with $N \in \{1,2,3,5,10\}$
       samples. Given: cost = $0.02 per LLM call, hallucination rate reduction shown
       in Figure 3 (approximate), and a $500/day budget. What $N$ maximises quality
       within budget for 10,000 queries/day?
    9. *Architecture review:* a colleague proposes using an LLM as the sole guardrail
       ("just ask the LLM to check its own output for hallucinations"). Identify at
       least three failure modes of this approach and propose how to address each.
    """),

    md(r"""
    ---
    ### Summary
    LLMs hallucinate because they generate the most probable token, not the most
    factually correct one — they cannot distinguish knowing from guessing. Hallucination
    is taxonomised as **intrinsic** (contradicts source) vs **extrinsic** (adds
    unsupported facts), each requiring different mitigations. **SelfCheckGPT** detects
    hallucination by sampling $N$ times and measuring consistency. **RAG grounding**
    is the most effective mitigation: conditioning on retrieved evidence reduces
    hallucination by 50–80%. **Guardrail pipelines** provide defence in depth: input
    classification, factual grounding, NLI output guards, and structured output
    validation. **Section 05 (Modern NLP and LLMs) is now complete.**

    **Related lesson:** `Section 06 — Retrieval-Augmented Generation` begins with `RAG-01 ·
    Similarity Search` — the algorithmic foundation of efficient nearest-neighbour
    search that underlies every RAG retrieval system.
    """),
]

build("05_nlp_and_llms/05_hallucination_and_guardrails.ipynb", cells)
