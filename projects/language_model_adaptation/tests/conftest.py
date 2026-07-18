from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "language_model_adaptation" / "src"))
sys.path.insert(0, str(ROOT / "projects" / "tiny_language_model" / "src"))
