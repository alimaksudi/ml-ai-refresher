import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "projects" / "prompt_evaluation" / "src"),
    str(ROOT / "projects" / "language_model_adaptation" / "src"),
    str(ROOT / "projects" / "tiny_language_model" / "src"),
]
