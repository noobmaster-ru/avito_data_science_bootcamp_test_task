from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
SEED = 42
K = 50
DENSE_MODEL = "intfloat/multilingual-e5-small"
RELEASE = ROOT / "release"          # небольшие артефакты, которые лежат в git (веса реранкера)
FT_MODEL = str(ART / "models" / "e5s_ft_v1")
