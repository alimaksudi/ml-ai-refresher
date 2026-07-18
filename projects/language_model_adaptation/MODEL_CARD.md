# Model Card

The model is a one-layer, 32-dimensional causal Transformer with 96-token context,
reused from `projects/tiny_language_model`. Full SFT updates all 16,928 parameters.
The LoRA candidate freezes the base and adds rank-4 adapters to attention input/output
projections, training 768 parameters. DPO starts from full SFT and compares response
sequence log probabilities with a frozen reference copy.

The model is too small and the data too narrow for deployment. PPO/RLHF is not executed
and is never presented as measured evidence in this project.
