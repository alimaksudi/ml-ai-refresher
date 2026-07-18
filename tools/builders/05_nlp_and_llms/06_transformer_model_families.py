"""Builder for NLP-06 — Transformer Model Families."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # NLP-06 · Transformer Model Families
    ### Section 05 — From one decoder to GPT, BERT, and T5

    DL-08 built and trained a decoder-only language model. This lesson changes one
    question: **which positions and sequences should each token be allowed to read?**
    That decision produces three major families: GPT-style causal decoders, BERT-style
    bidirectional encoders, and T5-style encoder-decoders with cross-attention.

    **Prerequisites:** DL-07, DL-08, NLP-01, and the tiny-language-model checkpoint.
    **Estimated time:** 5–8 hours including the project mastery assessment.
    """),
    md(r"""
    ## 1 · Learning Objectives

    - Reuse one attention mechanism for causal, bidirectional, and cross-attention.
    - Trace GPT, BERT, and T5 tensor shapes without hiding behind `nn.Transformer`.
    - Train real next-token, masked-token, classification, and source-to-target tasks.
    - Explain why architecture follows information visibility and output structure.
    - Choose a simpler baseline when a Transformer family is unnecessary.
    """),
    md(r"""
    ## 2 · Historical Motivation

    The original [Transformer paper](https://arxiv.org/abs/1706.03762) used an encoder
    and decoder for translation. [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
    showed that a causal decoder could pretrain with next-token prediction.
    [BERT](https://arxiv.org/abs/1810.04805) used a bidirectional encoder and masked
    tokens for representation learning. [T5](https://arxiv.org/abs/1910.10683) framed
    many NLP tasks as text-to-text with an encoder-decoder. The important progression
    is not brand names; it is how masks and objectives match the task.
    """),
    md(r"""
    ## 3 · Intuition and Visual Understanding

    | Family | What one position may read | Natural output |
    |---|---|---|
    | GPT decoder | Current and earlier target tokens | Continue a sequence |
    | BERT encoder | All unpadded input tokens | Contextual representation or label |
    | T5 encoder-decoder | Encoder reads source; decoder reads earlier target and all source | Transform one sequence into another |

    Think of GPT as writing while covering the unwritten page, BERT as reviewing a
    complete document, and T5 as a writer consulting a separate source document. The
    analogy stops at optimization: all three calculate differentiable vector operations,
    not human reading or writing.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    **GPT causal mask.** Before softmax, score $S_{ij}$ becomes
    $S_{ij}+M_{ij}$, where $M_{ij}=0$ when key position $j\le i$ and
    $M_{ij}=-\infty$ when $j>i$. Read aloud: query position $i$ may use only keys at
    or before itself. For positions `0,1,2`, row 1 allows keys `0,1` and blocks key 2.
    It applies to autoregressive targets; forgetting it leaks answers from the future.

    **Masked-token loss.** For masked positions $R$,
    $$J_{MLM}=-\frac{1}{|R|}\sum_{t\in R}\log p(x_t\mid x_{\setminus R}).$$
    Here $R$ is the set of hidden positions, $x_t$ is the original token, and
    $x_{\setminus R}$ is the visible corrupted sequence. If one token has predicted
    probability `0.8`, its loss is $-\log(0.8)=0.223$. This objective learns contextual
    reconstruction; it does not directly train left-to-right generation.

    **T5 cross-attention.** The decoder supplies queries while the encoder supplies keys
    and values:
    $$A=\operatorname{softmax}(Q_{dec}K_{enc}^{\top}/\sqrt D)V_{enc}.$$
    If decoder length is 6 and source length is 5, attention scores have shape
    `(B,H,6,5)`. Cross-attention connects separate sequences; using decoder keys here
    would remove the source-to-target bridge.
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    The project uses direct PyTorch tensor operations and linear layers. One shared
    `MultiHeadAttention` accepts query states, optional context states, a padding mask,
    and a causal flag. The surrounding blocks decide which family is constructed.
    """),
    code(r"""
    import sys
    from pathlib import Path
    import torch

    candidates = [Path.cwd(), *Path.cwd().parents]
    repo_root = next(path for path in candidates if (path / "projects/transformer_families").exists())
    project_root = repo_root / "projects" / "transformer_families"
    sys.path.insert(0, str(project_root / "src"))

    from transformer_families.models import (
        DecoderOnlyModel, EncoderOnlyModel, EncoderDecoderModel, FamilyConfig
    )
    from transformer_families.training import run_family_lab

    config = FamilyConfig()
    tokens = torch.tensor([[4, 5, 6, 7, 8]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    gpt = DecoderOnlyModel(config)
    bert = EncoderOnlyModel(config)
    t5 = EncoderDecoderModel(config)
    print("GPT logits:", tuple(gpt(tokens, valid).shape))
    print("BERT MLM logits:", tuple(bert(tokens, valid).shape))
    print("BERT class logits:", tuple(bert.classify(tokens, valid).shape))
    print("T5 logits:", tuple(t5(tokens, tokens, valid, valid).shape))
    """),
    code(r"""
    report = run_family_lab(seed=42, steps=120)
    for family, evidence in report.items():
        if isinstance(evidence, dict) and "parameters" in evidence:
            print(f"{family}: {evidence}")
    assert report["gpt_decoder_only"]["final_loss"] < report["gpt_decoder_only"]["initial_loss"]
    assert report["bert_encoder_only"]["mlm_final_loss"] < report["bert_encoder_only"]["mlm_initial_loss"]
    assert report["t5_encoder_decoder"]["final_loss"] < report["t5_encoder_decoder"]["initial_loss"]
    """),
    md(r"""
    ## 6 · Visualization

    Draw masks before reading metrics. White means visible and black means blocked.
    T5 cross-attention is rectangular because target and source lengths may differ.
    """),
    code(r"""
    import matplotlib.pyplot as plt
    import numpy as np

    length = 6
    causal = np.tril(np.ones((length, length)))
    bidirectional = np.ones((length, length))
    cross = np.ones((length, 5))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, matrix, title in zip(
        axes,
        [causal, bidirectional, cross],
        ["GPT / T5 decoder causal", "BERT / T5 encoder bidirectional", "T5 cross-attention"],
    ):
        axis.imshow(matrix, vmin=0, vmax=1, cmap="gray")
        axis.set_title(title); axis.set_xlabel("key position"); axis.set_ylabel("query position")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    | Symptom | Likely cause | Evidence | Scoped fix |
    |---|---|---|---|
    | GPT training looks impossibly easy | Future-token leakage | Change future input and inspect earlier logits | Restore causal mask |
    | BERT cannot use right context | Causal mask copied from GPT | Perturb later visible token | Remove causal mask; keep padding mask |
    | T5 ignores source | Missing/miswired cross-attention | Change source under fixed target | Use decoder Q with encoder K/V |
    | Padding changes valid outputs | Padding keys remain visible | Alter only padded token IDs | Mask padding before softmax |
    | High training accuracy presented as mastery | Synthetic memorization | No independent distribution | Label diagnostic limits; add held-out task |
    """),
    md(r"""
    ## 8 · Library or Tool Implementation

    After the scratch path, local Hugging Face classes map to the same families:
    `AutoModelForCausalLM`, `AutoModelForMaskedLM` or sequence classification, and
    `AutoModelForSeq2SeqLM`. A production comparison must pin a model revision, inspect
    its tokenizer and configuration, run locally when possible, and measure on one
    shared dataset. No hosted API is required for this lesson.
    """),
    md(r"""
    ## 9 · Realistic Case Study

    A support platform needs three capabilities. Drafting a response from prior tokens
    fits a decoder. Classifying ticket intent from the complete ticket fits an encoder
    and may be solved more cheaply by TF-IDF first. Translating a ticket while keeping
    source and output separate fits an encoder-decoder. Architecture selection follows
    information flow and constraints, not which model family is newest.
    """),
    md(r"""
    ## 10 · Production and Learning Considerations

    Compare data license, tokenizer, context, latency, memory, adaptation method, and
    evaluation—not only architecture labels. Decoder-only models can perform many tasks
    through prompting, but capability does not prove they are the simplest deployment.
    Encoder models often provide efficient representations; encoder-decoders make source
    conditioning explicit. This project omits distributed training and modern variants.
    """),
    md(r"""
    ## 11 · Tradeoff Analysis

    | Need | First candidate | Why | Simpler alternative |
    |---|---|---|---|
    | Open-ended continuation | Decoder-only | Objective matches generation | N-gram/template for constrained text |
    | Classification/embedding | Encoder-only | Full input context | TF-IDF + linear model |
    | Explicit source-to-target mapping | Encoder-decoder | Separate source and output flows | Rules for deterministic mapping |
    | One general interface | Decoder-only or text-to-text | Flexible task formatting | Separate specialized models |
    """),
    md(r"""
    ## 12 · Readiness and Interview Preparation

    You are ready when you can derive every mask, identify the sources of Q/K/V in
    cross-attention, trace score shapes, and choose a family from task requirements.
    A strong answer names the baseline and evidence that would justify extra capacity.
    """),
    md(r"""
    ## 13 · Teach-Back

    Explain GPT, BERT, and T5 without brand vocabulary: state what each position can
    see, what target supplies its loss, how output is produced, and one task for which
    the architecture is unnecessary. Then explain why changing only a mask can change
    the meaning of the learned representation.
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    1. **Guided (20 min):** draw all visibility matrices for source length 4 and target
       length 3. Self-check: cross-attention scores are `(B,H,3,4)`.
    2. **Beginner (25 min):** change a future GPT token and confirm prior logits remain
       equal; remove the mask and prove the test fails.
    3. **Intermediate (40 min):** mask two BERT tokens and compute loss only at those
       positions. Explain why padding and MLM masks have different jobs.
    4. **Intermediate (45 min):** change the T5 source under a fixed decoder prefix and
       measure which decoder logits change.
    5. **Challenge (60 min):** create held-out sequence patterns for all three tasks.
       Separate memorization from generalization and recommend the simplest model.

    Complete `projects/transformer_families/MASTERY_CHECKPOINT.md`; passing requires
    automated invariants, real loss reduction, a broken-mask diagnosis, and 17/20.
    """),
]

build("05_nlp_and_llms/06_transformer_model_families.ipynb", cells)
