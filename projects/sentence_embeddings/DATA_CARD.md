# Data Card

The dataset contains eight customer-support intents. Each intent has three training
queries, two training documents, one held-out query, one held-out document, and an
explicit confusable intent used as a hard negative.

Exact text overlap across training and evaluation is forbidden. Evaluation queries are
not used to fit the word vocabulary. Evaluation documents are included in vocabulary
construction because a deployed local index must be tokenizable before queries arrive.

The text was manually curated for learning. It is not representative of real traffic,
languages, class imbalance, ambiguity, policy changes, or harmful-content patterns.
