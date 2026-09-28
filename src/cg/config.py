from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
SEED = 42
K = 50
DENSE_MODEL = "intfloat/multilingual-e5-small"
