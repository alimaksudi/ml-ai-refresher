# ML / AI Refresher — Zero-Math Foundations to Senior Engineering

A 62-notebook curriculum that starts with mathematical and Python prerequisites,
then teaches Machine Learning and AI from first principles through production
system design. Later phases remain useful as a refresher for experienced engineers.

## Start here

If mathematical notation, algebra, calculus, probability, Python, or NumPy are
new, complete the prerequisite phase in order:

1. `00A · Mathematical Language and Arithmetic`
2. `00B · Algebra, Functions, and Graphs`
3. `00D · Python, NumPy, and Jupyter Foundations`
4. `00C · Calculus and Probability Intuition`
5. `00E · Practical Python, Pandas, Debugging, and Tests`

Each prerequisite notebook includes worked examples, guided practice,
independent exercises, solutions, and a readiness threshold. Experienced learners
may use the readiness checks to decide whether to continue directly to Phase 0.

After working through this you should be able to:

- Explain core concepts **without notes**.
- **Derive** the key algorithms from scratch.
- **Implement** simplified versions in pure NumPy.
- **Build** production-ready systems and reason about their failure modes.
- **Discuss tradeoffs** with the confidence expected of a Senior/Staff engineer.
- **Pass senior ML/AI interviews.**

## Canonical learning path

Notebook numbers are stable file identifiers, not the teaching order. The canonical
order and direct prerequisites live in
[`docs/CURRICULUM_PATH.json`](docs/CURRICULUM_PATH.json) and are embedded into every
generated notebook. In particular, the core ML spine intentionally teaches:

> data workflow (03A) → linear regression and squared loss (04) → optimization
> (03) → logistic regression (05) → metrics (09) → validation/leakage (10) →
> experiment tracking (44) → feature engineering (11) → trees and ensembles (06–08)

The Deep Learning spine is:

> PyTorch foundations (13A) → neural networks from scratch (14) → backpropagation
> (15) → stable neural training (15A) → CNN (16) and NLP representations (20) →
> RNN/LSTM (17) → attention (18) → transformers (19)

Run `make validate` to reject unknown, missing, or forward-pointing prerequisites.
New learners should use the two-pass study process and cumulative gates in
[`docs/STUDENT_MASTERY_PATH.md`](docs/STUDENT_MASTERY_PATH.md).
Integration work is defined at phase boundaries in
[`docs/PHASE_MINI_PROJECTS.md`](docs/PHASE_MINI_PROJECTS.md).

## Teaching method

Every topic notebook teaches in the same order:

> **Intuition → History → Mathematics → Scratch Implementation → Library Implementation → Production → Tradeoffs**

and answers, for each topic, the ten questions: *what is it, why was it
invented, what problem does it solve, how does it work intuitively, how does it
work mathematically, how would we implement it from scratch, when to use it,
when NOT to use it, what are the tradeoffs, and how does it behave in
production.*

## Standard notebook structure

Topic notebooks follow the same 14-section path. Capstones may adapt the names
while retaining foundations, implementation, failure analysis, evaluation,
tradeoffs, teach-back, and assessed exercises:

1. Learning Objectives
2. Historical Motivation
3. Intuition & Visual Understanding
4. Mathematical Foundations
5. Manual Implementation from Scratch (NumPy / stdlib only)
6. Visualization
7. Failure Modes
8. Production Library Implementation
9. Realistic Business Case Study
10. Production Considerations
11. Tradeoff Analysis
12. Senior-Level Interview Preparation
13. Teach-Back Section
14. Exercises, Self-Check, and Solutions

Every new formula follows the repository's [curriculum quality standard](docs/CURRICULUM_STANDARD.md):
read it aloud, define symbols, calculate a small example, explain the meaning,
and state its use and limits.

## Repository layout

```
ml-ai-refresher/
├── README.md
├── PROGRESS.md              # curriculum checklist (the build tracker)
├── requirements.txt
├── notebooks/               # generated .ipynb files, grouped by phase
│   ├── phase_minus1_onboarding/
│   ├── phase0_foundations/
│   ├── phase1_classical_ml/
│   └── ...
├── projects/
│   └── wine_classifier/    # real-data training, API, tests, monitoring, Docker
└── tools/
    ├── nbbuild.py           # md()/code()/build() helpers
    ├── build_all.py         # regenerate all (or some) notebooks
    └── builders/            # SOURCE OF TRUTH — one Python file per notebook
```

The notebooks are **build artifacts**. To change a notebook, edit its builder
in `tools/builders/` and regenerate. This keeps the content reviewable as plain
Python (no fragile JSON) and fully reproducible.

## Building the notebooks

```bash
pip install -r requirements.txt
python3 tools/build_all.py          # build everything
python3 tools/build_all.py 01       # build just notebook 01
jupyter lab notebooks/              # open and run
```

Optional library demonstrations are grouped by purpose rather than silently
required by the core path:

```bash
pip install -r requirements-nlp-rag.txt
pip install -r requirements-agents.txt
pip install -r requirements-evaluation-production.txt
```

## Quality checks

```bash
make build                 # regenerate all notebooks from builders
make validate              # source sync, structure, syntax, and clean outputs
make execute-foundations   # execute prerequisite + Phase 0 notebooks
make execute-capstone      # execute the applied capstone notebook
make execute-all           # execute all 62 notebooks
make capstone-test          # train and test the deployable vertical slice
```

Code cells are written to be runnable top-to-bottom; notebooks ship without
saved outputs so you execute them yourself (that *is* the learning).

## Roadmap

See **[PROGRESS.md](PROGRESS.md)** for the full 62-notebook roadmap and live
build status across the prerequisite phase and Phases 0–10 (Foundations → Classical ML → ML Engineering →
Deep Learning → NLP/LLMs → RAG → Agents → Evaluation → Production → System
Design → Applied Capstone). The runnable project is
documented in [projects/wine_classifier/README.md](projects/wine_classifier/README.md).
Primary papers and implementation references are collected in
[docs/REFERENCES.md](docs/REFERENCES.md).
