# Data Card

The project uses small curriculum-authored base text, astronomy-domain text,
instruction-response pairs, and chosen-rejected preference pairs. Visible curation
rules remove one exact duplicate, one short low-quality document, and one exact
evaluation-substring contamination case before continued pretraining.

The exact checker cannot detect paraphrases or semantic contamination. Preference data
encodes one narrow behavior—preferring verification language—and has no human agreement
measurement. All reported results are diagnostic mechanics, not population estimates.
