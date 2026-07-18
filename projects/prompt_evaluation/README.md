# Local Prompt Evaluation Lab

This project trains one tiny local instruction model and freezes it before comparing
three versioned sentiment prompts. The development set selects a prompt using a
predeclared rule; the final test set is opened only for the baseline and selected
version. Greedy decoding, tokenizer, seed, data hash, prompt hashes, per-case outputs,
schema validity, response loss, and paired uncertainty are recorded.

The experiment demonstrates prompt-evaluation mechanics. It does not establish that
one prompting technique is universally better, and it does not use a hosted API.

Run `make prompt-evaluation-test` and `make prompt-evaluation-run`, then complete
`MASTERY_CHECKPOINT.md`.
