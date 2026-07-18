# ML / AI Refresher — Zero-Math Foundations to Senior Engineering

A 65-notebook curriculum that starts with mathematical and Python prerequisites,
then teaches Machine Learning and AI from first principles through production
system design. Later sections remain useful as a refresher for experienced engineers.

## Start here

If mathematical notation, algebra, calculus, probability, Python, or NumPy are
new, complete the prerequisite section in order:

1. `PRE-01 · Mathematical Language and Arithmetic`
2. `PRE-02 · Algebra, Functions, and Graphs`
3. `PRE-03 · Python, NumPy, and Jupyter Foundations`
4. `PRE-04 · Calculus and Probability Intuition`
5. `PRE-05 · Practical Python, Pandas, Debugging, and Tests`

Each prerequisite notebook includes worked examples, guided practice,
independent exercises, solutions, and a readiness threshold. Experienced learners
may use the readiness checks to decide whether to continue to ML foundations.

After working through this you should be able to:

- Explain core concepts **without notes**.
- **Derive** the key algorithms from scratch.
- **Implement** simplified versions in pure NumPy.
- **Build** production-ready systems and reason about their failure modes.
- **Discuss tradeoffs** with the confidence expected of a Senior/Staff engineer.
- **Pass senior ML/AI interviews.**

## Canonical learning path

Semantic lesson IDs are stable identifiers. File numbers show local order within a
section, while the authoritative teaching order and direct prerequisites live in
[`docs/CURRICULUM_PATH.json`](docs/CURRICULUM_PATH.json) and are embedded into every
generated notebook. In particular, the core ML spine intentionally teaches:

> data workflow (FND-03) → linear regression and squared loss (CML-01) →
> optimization (FND-04) → logistic regression (CML-02) → metrics (MLE-01) →
> validation/leakage (MLE-02) → experiment tracking (PROD-04) → feature engineering
> (MLE-03) → trees and ensembles (CML-03 through CML-05)

The Deep Learning spine is:

> PyTorch foundations (DL-01) → neural networks from scratch (DL-02) →
> backpropagation (DL-03) → stable neural training (DL-04) → convolutional networks
> (DL-05) and text representations (NLP-01) → RNN/LSTM (DL-06) → attention (DL-07)
> → transformers (DL-08) → offline tiny-language-model mastery gate → GPT/BERT/T5
> model families (NLP-06) → sentence embeddings (NLP-02) → pretraining/data (NLP-03)
> → instruction tuning/LoRA (NLP-07) → preference alignment (NLP-08) → evaluation,
> prompting, and guardrails → RAG

Before RAG, pass `make tiny-lm-checkpoint`, `make transformer-families-checkpoint`,
`make sentence-embeddings-checkpoint`, `make language-model-adaptation-checkpoint`,
and `make prompt-evaluation-checkpoint`.
The first project trains a decoder-only model
end to end—without an API key—and tests shifted targets, causal masking, backprop,
validation, checkpoint loading, generation, and character-versus-BPE tokenization.
Cross-tokenizer likelihood is compared with bits per character. Naive and KV-cached
generation must produce equivalent logits before their latency is compared. Passing cells is not sufficient;
complete each project's human mastery checkpoint. The later gates prove attention-mask
behavior across GPT/BERT/T5 and train a local semantic retriever against TF-IDF and
untrained baselines.

Run `make validate` to reject unknown, missing, or forward-pointing prerequisites.
New learners should use the two-pass study process and cumulative gates in
[`docs/STUDENT_MASTERY_PATH.md`](docs/STUDENT_MASTERY_PATH.md).
Integration work is defined at section boundaries in
[`docs/INTEGRATION_PROJECTS.md`](docs/INTEGRATION_PROJECTS.md).
Repository names and stable ID rules are defined in
[`docs/REPOSITORY_NAMING_STANDARD.md`](docs/REPOSITORY_NAMING_STANDARD.md).

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
├── notebooks/               # generated .ipynb files, grouped by curriculum section
│   ├── 00_prerequisites/
│   ├── 01_ml_foundations/
│   ├── 02_classical_ml/
│   └── ...
├── projects/
│   ├── wine_classifier/       # real-data training, API, tests, monitoring, Docker
│   ├── digit_classifier/      # deep-learning experiment and mastery checkpoint
│   ├── tiny_language_model/   # offline decoder training and pre-RAG gate
│   ├── transformer_families/  # GPT, BERT, and T5 masks, objectives, and mastery gate
│   ├── prompt_evaluation/      # controlled local prompt comparison and release gate
│   └── rag_foundations/       # measured retrieval and grounded-answer checkpoints
└── tools/
    ├── nbbuild.py           # md()/code()/build() helpers
    ├── build_all.py         # regenerate all (or some) notebooks
    └── builders/            # SOURCE OF TRUTH — mirrors the notebook section tree
```

The notebooks are **build artifacts**. To change a notebook, edit its builder
in `tools/builders/` and regenerate. This keeps the content reviewable as plain
Python (no fragile JSON) and fully reproducible.

## Building the notebooks

```bash
pip install -r requirements.txt
python3 tools/build_all.py          # build everything
python3 tools/build_all.py RAG-05   # build one lesson by semantic ID
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
make execute-foundations   # execute prerequisite + ML foundation notebooks
make execute-capstone      # execute the applied capstone notebook
make execute-all           # execute all 62 notebooks
make capstone-test          # train and test the deployable vertical slice
```

Code cells are written to be runnable top-to-bottom; notebooks ship without
saved outputs so you execute them yourself (that *is* the learning).

## Roadmap

See **[PROGRESS.md](PROGRESS.md)** for the full 62-notebook roadmap and live
build status across Sections 00–11 (Prerequisites → Foundations → Classical ML → ML Engineering →
Deep Learning → NLP/LLMs → RAG → Agents → Evaluation → Production → System
Design → Applied Capstone). The runnable project is
documented in [projects/wine_classifier/README.md](projects/wine_classifier/README.md).
The measured retrieval, grounded-answer, and persistent vector-store projects are
documented in [projects/rag_foundations/README.md](projects/rag_foundations/README.md).
Primary papers and implementation references are collected in
[docs/REFERENCES.md](docs/REFERENCES.md).
