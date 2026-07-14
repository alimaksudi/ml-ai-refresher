.PHONY: build validate test execute-foundations execute-capstone execute-all capstone-train capstone-test capstone-serve mastery-checkpoint deep-learning-train deep-learning-test deep-learning-checkpoint foundation-gate rag-foundations-evaluate grounded-rag-evaluate vector-store-evaluate hybrid-rag-evaluate rag-system-evaluate rag-foundations-test rag-foundations-checkpoint grounded-rag-checkpoint vector-store-checkpoint hybrid-rag-checkpoint rag-system-checkpoint

build:
	python3 tools/build_all.py

validate:
	python3 tools/validate_curriculum.py

test: validate

execute-foundations:
	python3 tools/validate_curriculum.py --execute foundations

foundation-gate: execute-foundations
	@echo "Runtime gate passed. Complete Gates A–C in docs/STUDENT_MASTERY_PATH.md before continuing."

execute-capstone:
	python3 tools/validate_curriculum.py --execute capstone

execute-all:
	python3 tools/validate_curriculum.py --execute all

capstone-train:
	PYTHONPATH=projects/wine_classifier/src python3 -m wine_classifier.training --artifact-dir projects/wine_classifier/artifacts

capstone-test:
	PYTHONPATH=projects/wine_classifier/src python3 -m pytest projects/wine_classifier/tests -q

mastery-checkpoint: validate capstone-test
	@echo "Automated checkpoint passed. Complete the scored teach-back in projects/wine_classifier/MASTERY_CHECKPOINT.md."

deep-learning-train:
	PYTHONPATH=projects/digit_classifier/src python3 -m digit_classifier.training --output-dir projects/digit_classifier/artifacts

deep-learning-test:
	PYTHONPATH=projects/digit_classifier/src python3 -m pytest projects/digit_classifier/tests -q

deep-learning-checkpoint: validate deep-learning-test
	@echo "Automated checkpoint passed. Complete projects/digit_classifier/MASTERY_CHECKPOINT.md."

rag-foundations-evaluate:
	PYTHONPATH=projects/rag_foundations/src python3 -m rag_foundations.cli --data-dir projects/rag_foundations/data --output projects/rag_foundations/artifacts/evaluation.json

grounded-rag-evaluate:
	PYTHONPATH=projects/rag_foundations/src python3 -m rag_foundations.grounded_cli --data-dir projects/rag_foundations/data --output projects/rag_foundations/artifacts/grounded_evaluation.json

vector-store-evaluate:
	PYTHONPATH=projects/rag_foundations/src python3 -m rag_foundations.vector_store_cli --data-dir projects/rag_foundations/data --index-dir projects/rag_foundations/.local/vector_index --output projects/rag_foundations/artifacts/vector_store_evaluation.json

hybrid-rag-evaluate:
	PYTHONPATH=projects/rag_foundations/src python3 -m rag_foundations.hybrid_cli --data-dir projects/rag_foundations/data --output projects/rag_foundations/artifacts/hybrid_evaluation.json

rag-system-evaluate:
	PYTHONPATH=projects/rag_foundations/src python3 -m rag_foundations.rag_evaluation_cli --data-dir projects/rag_foundations/data --output projects/rag_foundations/artifacts/rag_system_evaluation.json

rag-foundations-test:
	PYTHONPATH=projects/rag_foundations/src python3 -m pytest projects/rag_foundations/tests -q

rag-foundations-checkpoint: validate rag-foundations-test
	@echo "Automated retrieval gate passed. Complete projects/rag_foundations/MASTERY_CHECKPOINT.md."

grounded-rag-checkpoint: validate rag-foundations-test grounded-rag-evaluate
	@echo "Automated grounded-answer gate passed. Complete the grounded-answer extension in projects/rag_foundations/MASTERY_CHECKPOINT.md."

vector-store-checkpoint: validate rag-foundations-test vector-store-evaluate
	@echo "Automated vector-store gate passed. Complete the vector-store extension in projects/rag_foundations/MASTERY_CHECKPOINT.md."

hybrid-rag-checkpoint: validate rag-foundations-test hybrid-rag-evaluate
	@echo "Automated hybrid-search gate passed. Complete the hybrid-search extension in projects/rag_foundations/MASTERY_CHECKPOINT.md."

rag-system-checkpoint: validate rag-foundations-test rag-system-evaluate
	@echo "Automated RAG evaluation gate passed. Complete the EVAL-03 extension in projects/rag_foundations/MASTERY_CHECKPOINT.md."

capstone-serve:
	PYTHONPATH=projects/wine_classifier/src uvicorn wine_classifier.app:app --host 127.0.0.1 --port 8000
