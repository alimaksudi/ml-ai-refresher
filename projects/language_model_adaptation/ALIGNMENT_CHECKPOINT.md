# Preference Learning and Alignment Checkpoint

Score ten questions 0–2; pass at 17/20.

1. What information is contained in a preference triple?
2. What assumptions does the Bradley–Terry model make?
3. Why does DPO need a frozen reference policy?
4. What sequence positions contribute to a response log probability?
5. What does beta control in DPO?
6. Why can response length confound summed log probabilities?
7. Distinguish the KL-regularized RL objective from PPO's clipped surrogate.
8. Why does a very negative logistic DPO margin have a large, not vanishing, gradient?
9. What retention regression occurred after DPO in this experiment?
10. Why does preference accuracy not prove truth, helpfulness, or safety?

Practical repair: reverse one chosen/rejected label, identify the contradictory pair,
measure its effect, and define a preference-data review rule.
