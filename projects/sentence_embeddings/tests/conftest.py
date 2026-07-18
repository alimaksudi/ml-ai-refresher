from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "projects" / "sentence_embeddings" / "src"))
sys.path.insert(0, str(REPO_ROOT / "projects" / "transformer_families" / "src"))
