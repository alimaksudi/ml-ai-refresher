# Curriculum Quality Standard

This curriculum has two audiences:

1. A learner starting with no mathematical background.
2. An experienced engineer using later phases as a refresher.

The prerequisite section must make the first audience ready for Section 01. Later
phases may assume earlier notebooks, but must link to the prerequisite concept
when they introduce notation that depends on it.

## Formula contract

Every new formula must include these five parts:

1. **Read it aloud** in plain language.
2. **Symbols**: define every new symbol, subscript, superscript, operator, and
   bracket. Include type or shape when relevant: scalar, vector, matrix, set,
   probability, or unit.
3. **Small example**: substitute concrete numbers and compute the result.
4. **Meaning**: explain what the result tells the learner.
5. **Use and limits**: state when the formula applies and one common misuse.

Symbols already defined in the same notebook may be referenced instead of
redefined. A formula must never depend on unexplained notation from a later
notebook.

## Exercise contract

Each notebook follows this progression:

1. **Worked example**: every step is shown.
2. **Guided practice**: hints identify the first step.
3. **Independent practice**: no procedural hint.
4. **Challenge**: combines concepts or introduces a production constraint.

Every exercise set must include:

- a self-check or expected result;
- a concise solution or scoring rubric;
- common mistakes for the foundational exercises;
- an estimated time;
- explicit prerequisites when the exercise reaches outside the notebook.

“Beginner” means a learner who has completed the listed prerequisites, not a
working ML engineer.

## Evidence contract

- Historical claims, benchmark numbers, and industry statistics need a source.
- Synthetic data, mock LLMs, hash embeddings, proxy metrics, and simulated
  latency must be labelled where results are presented.
- A proxy must not be named as though it were the real construct. For example,
  lexical overlap is a *lexical support proxy*, not faithfulness itself.
- Production guidance must separate a teaching implementation from a deployable
  implementation.

## Notebook contract

All topic notebooks retain the 14-section learning path:

1. Learning Objectives
2. Historical Motivation
3. Intuition and Visual Understanding
4. Mathematical Foundations
5. Manual Implementation from Scratch
6. Visualization
7. Failure Modes and Common Mistakes
8. Library or Tool Implementation
9. Realistic Case Study
10. Production or Learning Considerations
11. Tradeoff Analysis
12. Readiness or Interview Preparation
13. Teach-Back
14. Exercises, Self-Check, and Solutions

The 14 sections are containers, not a reason to omit beginner-facing guidance or
create 21 repetitive headings. Every generated lesson also includes a **Student
Lesson Companion** and **Lesson Close** with the following contract:

1. **Practical problem before history:** identify the decision or task, why it is
   difficult, the previous baseline, its limitation, and why this concept is needed.
2. **Simple concept and analogy:** explain the concept without unexplained jargon;
   use either a real-life analogy or a concrete visual example, and state where the
   analogy stops being accurate.
3. **Use / avoid / alternatives:** give explicit situations for use, non-use, and a
   simpler or more appropriate alternative. A compact comparison table is preferred.
4. **Intuition before formula:** no new formula appears before plain-language
   intuition and a tiny example.
5. **Implementation reading contract:** the first code path uses descriptive names,
   small data, useful comments, intermediate output, and explicit shapes or units.
6. **Expected-result contract:** state expected shape/range/value, what it means,
   what indicates broken code, and what indicates a valid but weak result.
7. **Debugging contract:** include symptom, likely cause, evidence to inspect, and a
   scoped fix. Conceptual limitations do not replace executable debugging guidance.
8. **Related-concept comparison:** compare purpose, strengths, weaknesses, and use
   conditions rather than only listing methods.
9. **Student check and close:** finish with five short checks, a plain-language
   summary, one decision rule, and a one-sentence memory aid.
10. **Projects at phase boundaries:** lessons use small independent applications;
    integration mini-projects belong after a coherent group of lessons.

Technical terminology must be explained immediately on first use. Senior interview
and production material is an extension path; it must not hide the beginner core.

Capstones may adapt the section names, but must still include objectives,
foundations, implementation, failure analysis, evaluation, tradeoffs,
teach-back, and assessed exercises.

## Repository quality gates

The repository must be able to verify that:

- every builder produces its documented notebook;
- generated notebooks match their builders;
- code cells compile;
- committed notebooks contain no saved execution output;
- foundational formulas include symbol explanations;
- the prerequisite and Section 01 notebooks execute top-to-bottom;
- the complete notebook suite can be executed on demand.
