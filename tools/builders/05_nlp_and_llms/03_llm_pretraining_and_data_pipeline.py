"""Build NLP-03: from governed documents to a resumable pretraining experiment."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # NLP-03 · LLM Pretraining and the Data Pipeline

    **Prerequisites:** FND-04, MLE-02, PROD-04, DL-08, NLP-01, NLP-06, and the tiny-LM checkpoint

    **Estimated mastery time:** 10–14 hours, including the checkpoint

    **Next canonical lesson:** NLP-07 · Instruction Tuning and LoRA

    A correct training loop can still produce an untrustworthy model. The data may be
    unlicensed, duplicated, contaminated by evaluation text, dominated by one source,
    packed across unsafe boundaries, or impossible to reproduce.

    This lesson builds the work surrounding next-token optimization:

    ```text
    source approval → manifest → normalization → quality and safety filters
    → deduplication → contamination audit → document-level split
    → tokenizer compatibility → tokenization → boundary-aware packing
    → token accounting → training → domain and retention validation
    → atomic checkpoint → exact resume test
    ```

    No hosted model or downloaded dataset is required.
    """),

    md(r"""
    ## 1 · What you will be able to do

    - distinguish base pretraining, continued pretraining, instruction tuning, and retrieval;
    - define a document manifest with provenance, license, consent, and content fields;
    - normalize text without silently destroying meaningful distinctions;
    - implement exact hashing and near-duplicate shingle comparison;
    - explain why deduplication and contamination checks happen before splitting;
    - split by source or document before token windows are created;
    - preserve tokenizer/checkpoint compatibility during continued pretraining;
    - pack documents with explicit boundary tokens and loss rules;
    - calculate next-token loss, perplexity, token counts, and a rough FLOP budget;
    - design source mixtures without confusing documents, characters, and tokens;
    - measure domain improvement and base-domain retention separately;
    - define a complete checkpoint and prove interrupted training can resume;
    - decide when prompting, RAG, fine-tuning, or no model change is more appropriate.
    """),

    md(r"""
    ## 2 · Four interventions solve different problems

    | Intervention | Updates base weights? | Best suited to | Main risk |
    |---|---:|---|---|
    | prompting | no | task framing with existing capability | fragile instructions/context |
    | RAG | no | current or traceable external knowledge | retrieval and grounding failures |
    | continued pretraining | yes | broad domain language distribution | forgetting, contamination, compute |
    | instruction tuning | yes or adapters | response format and task-following behavior | narrow overfit and regressions |
    | training from scratch | yes, from random start | a new base model with sufficient governed data/compute | enormous cost and data burden |

    Continued pretraining uses the same causal next-token objective as base pretraining,
    but starts from an existing checkpoint and changes the training distribution.

    A new policy document published every week is usually a retrieval problem. A large,
    stable corpus with specialized language may justify continued pretraining. Make the
    choice from a measured failure, not from the availability of text.
    """),

    code(r"""
    import hashlib
    import json
    import math
    import random
    import re
    import sys
    from dataclasses import asdict, dataclass
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    random.seed(42)
    np.random.seed(42)
    _ = torch.manual_seed(42)
    """),

    md(r"""
    ## 3 · Start with a document contract

    Raw text alone is not enough. A useful manifest keeps decisions auditable.

    | Field | Why it exists |
    |---|---|
    | `document_id` | stable lineage and deletion |
    | `source` and `source_version` | provenance and grouping |
    | `license` / permitted use | legal training authority |
    | `collected_at` | time-aware audits |
    | `language` | mixture and quality slices |
    | `text` or content pointer | training content |
    | `content_sha256` | exact identity after declared normalization |
    | `split` | frozen train/development/test membership |
    | filter decisions | why a row was kept or removed |

    PII, secrets, malware, unsafe material, opt-outs, and deletion requests require
    policy and review beyond simple notebook rules. “Publicly reachable” does not mean
    “licensed and appropriate for training.”
    """),

    code(r"""
    @dataclass(frozen=True)
    class Document:
        document_id: str
        source: str
        source_version: str
        license: str
        collected_at: str
        language: str
        text: str


    documents = [
        Document("astro-001", "licensed_handbook", "2026-01", "internal-training", "2026-01-10", "en",
                 "A planet travels around a star. A moon travels around a planet."),
        Document("astro-002", "licensed_handbook", "2026-01", "internal-training", "2026-01-10", "en",
                 "  a PLANET travels around a star.\nA moon travels around a planet.  "),
        Document("astro-003", "licensed_notes", "2026-02", "internal-training", "2026-02-02", "en",
                 "A telescope observes distant light. Astronomers study stars and comets."),
        Document("astro-004", "licensed_notes", "2026-02", "internal-training", "2026-02-02", "en", "ok"),
        Document("astro-005", "licensed_notes", "2026-03", "internal-training", "2026-03-05", "en",
                 "Astronomers study distant stars and observe their light with a telescope."),
    ]
    display(pd.DataFrame([asdict(document) for document in documents]))
    """),

    md(r"""
    ## 4 · Normalization and exact deduplication

    A declared normalization policy makes superficial differences comparable. Here we
    case-fold and collapse whitespace. This is appropriate for the small English
    example, not a universal multilingual policy. Unicode normalization, punctuation,
    code formatting, case, and whitespace may carry meaning in other corpora.

    Exact normalized duplicates should be clustered **before splitting**. Otherwise
    one copy can enter training and another evaluation, producing memorization disguised
    as generalization.
    """),

    code(r"""
    def normalize_text(text):
        return " ".join(text.casefold().split())


    def text_sha256(text):
        return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


    exact_groups = {}
    for document in documents:
        exact_groups.setdefault(text_sha256(document.text), []).append(document.document_id)

    print("exact duplicate groups:")
    for digest, ids in exact_groups.items():
        if len(ids) > 1:
            print(digest[:12], ids)
    assert any(set(ids) == {"astro-001", "astro-002"} for ids in exact_groups.values())
    """),

    md(r"""
    ## 5 · Near duplicates need approximate evidence

    Exact hashes miss edits and paraphrases. One transparent approximation uses word
    shingles: contiguous groups of $n$ words. For shingle sets $A$ and $B$:

    $$
    J(A,B)=\frac{|A\cap B|}{|A\cup B|}
    $$

    A high Jaccard score means strong surface overlap, not semantic identity. Thresholds
    must be calibrated by source and reviewed because boilerplate can create false
    matches while paraphrases can remain low. Large pipelines often use MinHash or
    locality-sensitive hashing to avoid all-pairs comparison.
    """),

    code(r"""
    def word_shingles(text, width=3):
        words = normalize_text(text).split()
        return {tuple(words[index:index + width]) for index in range(len(words) - width + 1)}


    def jaccard(left, right):
        union = left | right
        return len(left & right) / len(union) if union else 1.0


    near_duplicate_rows = []
    for left_index, left in enumerate(documents):
        for right in documents[left_index + 1:]:
            score = jaccard(word_shingles(left.text), word_shingles(right.text))
            near_duplicate_rows.append({"left": left.document_id, "right": right.document_id, "jaccard": score})
    near_duplicate_table = pd.DataFrame(near_duplicate_rows).sort_values("jaccard", ascending=False)
    display(near_duplicate_table.head())
    """),

    md(r"""
    ## 6 · Quality and safety filters must leave an audit trail

    A filter should return a decision and reason, not silently drop a row. Length is
    only a teaching example; real quality signals may include language confidence,
    encoding corruption, repeated boilerplate, source reputation, PII/secrets, unsafe
    payloads, and model-assisted review with human calibration.
    """),

    code(r"""
    def quality_decision(document):
        normalized = normalize_text(document.text)
        if document.license != "internal-training":
            return False, "license_not_approved"
        if len(normalized.split()) < 5:
            return False, "too_short"
        return True, "kept"


    filter_log = [
        {"document_id": document.document_id, "keep": quality_decision(document)[0],
         "reason": quality_decision(document)[1], "sha256": text_sha256(document.text)}
        for document in documents
    ]
    display(pd.DataFrame(filter_log))
    """),

    md(r"""
    ## 7 · Contamination is defined against protected evaluation assets

    Exact substring matching is a floor, not a complete audit. Check normalized hashes,
    passages, benchmark prompts and answers, near duplicates, source lineage, and time.
    Keep protected evaluation text out of tokenizer fitting, filtering thresholds,
    hyperparameter selection, and training—not only gradient updates.

    A contamination report should record the protected-set version, matching method,
    threshold, matched spans or clusters, and action taken.
    """),

    code(r"""
    protected_evaluation = [
        "A telescope observes distant light.",
        "Which instrument observes distant light?",
    ]


    def exact_contamination(document_text, protected_texts):
        candidate = normalize_text(document_text)
        return [
            protected for protected in protected_texts
            if normalize_text(protected) in candidate or candidate in normalize_text(protected)
        ]


    contamination_rows = [
        {"document_id": document.document_id,
         "matches": exact_contamination(document.text, protected_evaluation)}
        for document in documents
    ]
    display(pd.DataFrame(contamination_rows))
    print("Exact matching cannot detect a semantic rewrite of a protected answer.")
    """),

    md(r"""
    ## 8 · Split documents before creating token windows

    The safe order is:

    ```mermaid
    flowchart LR
        A[duplicate/source groups] --> B[frozen document split]
        B --> C[fit tokenizer on allowed training text]
        C --> D[tokenize each partition]
        D --> E[pack within each partition]
        E --> F[next-token windows]
    ```

    Randomly splitting overlapping windows leaks nearly identical contexts. Group by
    duplicate cluster, source, entity, author, time, or other dependency that could
    cross the boundary. A held-out domain-development set chooses training decisions;
    a final test remains sealed.
    """),

    code(r"""
    # A small group split: one source version cannot appear in multiple partitions.
    approved_unique = [documents[0], documents[2], documents[4]]
    split_by_version = {
        "train": [document for document in approved_unique if document.source_version in {"2026-01", "2026-02"}],
        "development": [document for document in approved_unique if document.source_version == "2026-03"],
    }
    for split_name, split_documents in split_by_version.items():
        print(split_name, [document.document_id for document in split_documents])
    train_versions = {document.source_version for document in split_by_version["train"]}
    development_versions = {document.source_version for document in split_by_version["development"]}
    assert train_versions.isdisjoint(development_versions)
    """),

    md(r"""
    ## 9 · Keep tokenizer and checkpoint compatible

    Token ID 17 must mean the same token expected by embedding row 17. Re-fitting or
    reordering a tokenizer while loading old weights silently changes meaning or causes
    shape failure.

    Continued pretraining normally preserves the base tokenizer. If domain text creates
    inefficient segmentation, measure fertility (tokens per source unit) first. Adding
    tokens requires a deliberate vocabulary migration: resize embedding and LM-head
    matrices, initialize new rows, preserve old IDs, version artifacts, and regression
    test old text. It is not a harmless tokenizer swap.
    """),

    code(r"""
    base_vocabulary = {"<PAD>": 0, "<EOS>": 1, "a": 2, "planet": 3, "star": 4}
    unsafe_refit = {"<PAD>": 0, "<EOS>": 1, "a": 2, "star": 3, "planet": 4}
    print("base ID 3 means:", next(token for token, token_id in base_vocabulary.items() if token_id == 3))
    print("refit ID 3 means:", next(token for token, token_id in unsafe_refit.items() if token_id == 3))
    print("Loading old embedding row 3 under the refit tokenizer would change its meaning.")
    """),

    md(r"""
    ## 10 · Pack documents without erasing boundaries

    Concatenating documents can create a false next-token target from the end of one
    document to the beginning of an unrelated one. Insert an explicit end-of-document
    token, or reset/mask loss and attention according to the training design.

    Packing reduces padding waste. It does not authorize cross-document attention.
    Some models allow packed segments to attend across boundaries; others use a
    block-diagonal mask. Record the choice because it changes the objective.
    """),

    code(r"""
    EOS_ID = 1
    tokenized_documents = [[2, 3, 4], [2, 4, 3, 4], [3, 2]]


    def pack_documents(token_rows, block_size, eos_id):
        stream, boundaries = [], []
        for row in token_rows:
            stream.extend(row)
            stream.append(eos_id)
            boundaries.append(len(stream) - 1)
        windows = []
        for start in range(0, len(stream) - block_size, block_size):
            chunk = stream[start:start + block_size + 1]
            windows.append((chunk[:-1], chunk[1:]))
        return stream, boundaries, windows


    packed_stream, boundary_positions, packed_windows = pack_documents(
        tokenized_documents, block_size=5, eos_id=EOS_ID
    )
    print("packed stream:", packed_stream)
    print("EOS positions:", boundary_positions)
    print("input → target windows:", packed_windows)
    assert all(packed_stream[position] == EOS_ID for position in boundary_positions)
    """),

    md(r"""
    ## 11 · Objective, perplexity, and accounting

    For tokens $x_1,\ldots,x_T$:

    $$
    J=-\frac{1}{T-1}\sum_{t=1}^{T-1}
    \log p_\theta(x_{t+1}\mid x_{1:t})
    $$

    $T$ is sequence length, $t$ is the current position, $\theta$ is the model state,
    and $J$ is mean negative log-likelihood in nats. If correct-token probabilities are
    0.5 and 0.25:

    $$
    J=\frac{-\log(0.5)-\log(0.25)}{2}=1.040
    $$

    Perplexity is $\exp(J)=2.828$ for the same tokenization. Perplexity is not directly
    comparable across different tokenizers because the prediction unit changes.

    A common first-order dense-Transformer estimate is:

    $$
    C\approx6ND
    $$

    where $N$ is non-embedding/model parameter scale depending on the convention, $D$
    is training tokens processed, and $C$ is FLOPs. It omits architecture details,
    sequence-length effects, optimizer overhead, hardware utilization, communication,
    failed runs, and data preparation. State the convention with the estimate.
    """),

    code(r"""
    true_probabilities = np.array([0.5, 0.25])
    mean_loss = float(-np.log(true_probabilities).mean())
    perplexity = math.exp(mean_loss)
    parameter_count = 1_000_000
    processed_tokens = 10_000_000
    estimated_flops = 6 * parameter_count * processed_tokens
    print("mean loss:", mean_loss)
    print("perplexity:", perplexity)
    print("rough training FLOPs:", f"{estimated_flops:.2e}")
    """),

    md(r"""
    ## 12 · Mixtures are measured in tokens

    “50% source A” is ambiguous unless the unit is stated. Pretraining mixtures are
    commonly sampled by tokens or examples, with caps or temperature smoothing so a
    huge source does not dominate and a tiny source is not repeated into memorization.

    Track raw tokens, retained tokens, sampled tokens, unique documents, repeated
    epochs, and loss by source. A mixture weight is a training policy, not the natural
    frequency of truth.
    """),

    code(r"""
    source_tokens = {"general": 8_000_000, "astronomy": 2_000_000}
    desired_fraction = {"general": 0.7, "astronomy": 0.3}
    training_budget = 12_000_000
    mixture_rows = []
    for source, fraction in desired_fraction.items():
        sampled = int(training_budget * fraction)
        mixture_rows.append({
            "source": source,
            "available tokens": source_tokens[source],
            "sampled tokens": sampled,
            "effective passes": sampled / source_tokens[source],
        })
    display(pd.DataFrame(mixture_rows))
    """),

    md(r"""
    ## 13 · A measured continued-pretraining experiment

    The local lab:

    1. trains a tiny base model on general text;
    2. copies the base checkpoint;
    3. continues only the copy on curated astronomy text;
    4. measures held-out astronomy loss and held-out base-text retention loss.

    Domain development data may guide whether to continue. Retention is a separate
    constraint. Do not select a checkpoint on a final test set, and do not hide a
    retention regression because the domain metric improved.
    """),

    code(r"""
    roots = [Path.cwd(), *Path.cwd().parents]
    repository = next(path for path in roots if (path / "projects/language_model_adaptation").exists())
    sys.path[:0] = [
        str(repository / "projects/language_model_adaptation/src"),
        str(repository / "projects/tiny_language_model/src"),
    ]
    from language_model_adaptation.lab import run_adaptation_lab

    adaptation_report = run_adaptation_lab(seed=42)
    curation_report = adaptation_report["data_pipeline"]
    continued = adaptation_report["continued_pretraining"]
    print("curation:", curation_report)
    print("continued-pretraining evidence:", continued)

    assert continued["domain_loss_after"] < continued["domain_loss_before"]
    assert continued["base_retention_loss_after"] > continued["base_retention_loss_before"]
    """),

    code(r"""
    labels = ["domain", "base retention"]
    before = [continued["domain_loss_before"], continued["base_retention_loss_before"]]
    after = [continued["domain_loss_after"], continued["base_retention_loss_after"]]
    positions = np.arange(2)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(positions - 0.18, before, 0.36, label="before adaptation")
    axis.bar(positions + 0.18, after, 0.36, label="after adaptation")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("held-out token loss")
    axis.set_title("Domain gain and retention cost must be read together")
    axis.legend()
    plt.show()
    """),

    md(r"""
    The correct conclusion is narrow: this run adapted to the tiny astronomy
    distribution and damaged performance on its tiny base-retention distribution. It
    does not establish astronomy expertise. A next experiment could mix 10%, 20%, and
    40% base replay, hold update count constant, and compare a domain-versus-retention
    frontier across several seeds.
    """),

    md(r"""
    ## 14 · A checkpoint is more than model weights

    An exact resume needs:

    - model weights and architecture configuration;
    - tokenizer files and special-token IDs;
    - optimizer and learning-rate scheduler state;
    - gradient scaler state when mixed precision uses one;
    - global step, tokens seen, epoch/shard and data-cursor state;
    - Python, NumPy, CPU and accelerator RNG states;
    - data manifest, mixture configuration, code revision, and environment;
    - best-metric record and checkpoint-selection rule.

    Write checkpoints atomically: write a temporary artifact, validate it, then rename.
    Keep periodic and best checkpoints under a retention policy. A “resume test” should
    compare an uninterrupted run with an interrupted-and-restored run under deterministic
    conditions; successful loading alone is insufficient.
    """),

    code(r"""
    # Minimal state shape for one optimizer step. Real data-loader cursor state must
    # also be captured by the input pipeline.
    resume_manifest = {
        "model_config_sha256": "...",
        "tokenizer_sha256": "...",
        "data_manifest_sha256": "...",
        "global_step": 1200,
        "tokens_seen": 98_304_000,
        "optimizer_state": "optimizer.pt",
        "scheduler_state": "scheduler.pt",
        "rng_states": ["python", "numpy", "torch_cpu", "accelerator"],
        "data_cursor": {"shard": 17, "sample_offset": 4096},
    }
    print(json.dumps(resume_manifest, indent=2))
    """),

    md(r"""
    ## 15 · Failure modes

    | Symptom | Likely cause | Evidence | Repair |
    |---|---|---|---|
    | suspicious validation gain | train/evaluation contamination | overlap clusters and source IDs | rebuild split before training |
    | loss falls mainly on repeated text | duplicate oversampling | cluster frequency slices | deduplicate/cap repetition |
    | checkpoint loads but output changes | tokenizer/config mismatch | artifact hashes and old-text regression | restore paired revision |
    | resumed loss diverges immediately | optimizer/RNG/data cursor missing | uninterrupted-vs-resumed comparison | save complete state |
    | domain improves, base worsens | catastrophic forgetting | retention set by source | replay mixture, fewer/smaller updates |
    | one source dominates | mixture counted by documents, not tokens | sampled token accounting | define token-based policy |
    | document transitions look learned | packing crossed boundaries blindly | inspect EOS and masks | boundary-aware packing |
    | clean average hides weak sources | only aggregate validation | source/language/length slices | release gates per risk slice |

    Exact matching misses paraphrases; near-duplicate matching creates false positives;
    quality classifiers can encode bias; and filters themselves drift. Version and
    evaluate the pipeline, not only the model.
    """),

    md(r"""
    ## 16 · Student check and practice

    Answer without notes:

    1. Why deduplicate before splitting?
    2. What does Jaccard shingle similarity detect and miss?
    3. Why can a tokenizer refit corrupt an otherwise shape-compatible checkpoint?
    4. Why insert EOS between packed documents?
    5. Which unit defines a source mixture here?
    6. What does $6ND$ estimate, and what does it omit?
    7. Why are domain and retention sets both required?
    8. Which states are needed for an exact resume?
    9. When is RAG preferable to continued pretraining?
    10. Why is lower next-token loss not proof of domain expertise?

    **Beginner:** manually normalize and hash two whitespace/case variants. Calculate
    loss and perplexity for probabilities `[0.8,0.4,0.2]`.

    **Intermediate:** create duplicate groups before a source-level split. Pack the
    training documents with EOS and verify no evaluation ID appears in any window.

    **Challenge:** implement base replay fractions of 10%, 20%, and 40% under a fixed
    token/update budget. Run three seeds, plot domain loss against retention loss, and
    choose no checkpoint using final-test metrics.
    """),

    md(r"""
    ## 17 · Mastery checkpoint and summary

    Complete `projects/language_model_adaptation/PRETRAINING_CHECKPOINT.md`. Your
    submission needs a versioned data manifest, filter report, duplicate and
    contamination audit, frozen split, tokenizer compatibility record, packing tests,
    token/FLOP accounting, domain and retention curves, and an interrupted-resume test.

    Continued pretraining is not “put text in a trainer.” It is a governed experiment
    whose data decisions determine what the objective rewards.

    **Memory aid:** *Authorize, deduplicate, protect the test, freeze IDs, preserve
    boundaries, count tokens, and measure both learning and forgetting.*

    NLP-07 comes next because instruction tuning changes the supervision contract from
    predicting every corpus token to learning from curated prompt–response behavior.
    """),
]


build("05_nlp_and_llms/03_llm_pretraining_and_data_pipeline.ipynb", cells)
