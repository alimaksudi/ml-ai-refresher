# Language Model Adaptation Learning Lab

This project continues the real decoder-only model from the tiny-language-model phase
through three learning stages and one evaluation stage without a hosted API:

1. data curation and continued pretraining;
2. response-masked SFT with full tuning and LoRA;
3. preference optimization with DPO and a frozen reference policy.
4. paired held-out evaluation with explicit retention guardrails.

The experiment intentionally exposes tradeoffs. Continued pretraining improves domain
loss while damaging the original-domain retention loss. Full SFT nearly memorizes the
small training set but generalizes worse than the lower-capacity LoRA candidate. DPO
reverses held-out preference accuracy while changing SFT retention behavior.

Run `make language-model-adaptation-test` and `make language-model-adaptation-train`,
then complete `EVALUATION_CHECKPOINT.md` before prompt engineering.
The dataset is a diagnostic authored for this curriculum, not evidence of broad model
quality, safety, factuality, or production readiness.
