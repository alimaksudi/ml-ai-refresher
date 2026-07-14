# Repository Naming Standard

This repository uses names that remain readable in a file browser and stable when the
teaching order changes.

## Curriculum section directories

Notebook and builder directories use a two-digit curriculum-section prefix followed by
a lowercase `snake_case` description:

```text
00_prerequisites
01_ml_foundations
02_classical_ml
03_ml_engineering
04_deep_learning
05_nlp_and_llms
06_rag
07_ai_agents
08_evaluation
09_production_ml
10_system_design
11_capstone
```

Do not use negative phase numbers, unpadded phase numbers, spaces, or mixed casing.
Section numbers describe the broad curriculum progression. The exact lesson route is
defined by `docs/CURRICULUM_PATH.json` because prerequisite lessons can cross sections.

## Lesson files

Lesson files use a two-digit local position and a descriptive lowercase slug:

```text
05_vector_databases.ipynb
05_vector_databases.py
```

The notebook and its builder must have the same relative path and stem. Builders live
under `tools/builders/`; generated notebooks live under `notebooks/`.

Use established technical abbreviations when they improve readability (`rag`, `nlp`,
`llm`, `cnn`, `rnn`, `mlops`). Use words such as `and` instead of punctuation in file
names. Do not add letter suffixes such as `03a`; insert or reorder lessons through the
canonical curriculum path instead.

## Stable lesson IDs

Dependencies use semantic IDs rather than historical global numbers:

| Prefix | Section |
|---|---|
| `PRE` | Prerequisites |
| `FND` | ML foundations |
| `CML` | Classical ML |
| `MLE` | ML engineering |
| `DL` | Deep learning |
| `NLP` | NLP and LLMs |
| `RAG` | Retrieval-augmented generation |
| `AGT` | AI agents |
| `EVAL` | Evaluation |
| `PROD` | Production ML |
| `SYS` | System design |
| `CAP` | Capstone |

IDs use `<PREFIX>-<two-digit-local-position>`, for example `RAG-05`. An ID identifies
a lesson and must not be reused. Teaching order is the order of entries in
`CURRICULUM_PATH.json`, not alphabetical ID order.

## Python packages and projects

Python packages and project directories use lowercase `snake_case`. A runnable project
uses the standard layout:

```text
projects/project_name/
├── README.md
├── requirements.txt
├── src/project_name/
└── tests/
```

Generated local indexes, caches, virtual environments, and notebook checkpoints must
remain ignored. Versioned evaluation reports and small teaching artifacts may be
committed when they are required to reproduce a mastery gate.
