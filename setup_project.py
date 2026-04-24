# setup_project.py

from pathlib import Path

ROOT = Path(__file__).resolve().parent

directories = [
    "data/raw",
    "data/processed",
    "data/predictions",
    "src/data",
    "src/features",
    "src/models",
    "src/risk",
    "src/explain",
    "src/services",
    "src/api",
    "app",
    "artifacts",
    "tests",
    "notebooks",
]

gitkeep_files = [
    "data/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/predictions/.gitkeep",
    "app/.gitkeep",
    "artifacts/.gitkeep",
    "tests/.gitkeep",
    "notebooks/.gitkeep",
]


for directory in directories:
    path = ROOT / directory
    path.mkdir(parents=True, exist_ok=True)

for file_path in gitkeep_files:
    (ROOT / file_path).touch(exist_ok=True)


src_subdirs = [
    "src/data",
    "src/features",
    "src/models",
    "src/risk",
    "src/explain",
    "src/services",
    "src/api",
]

for subdir in src_subdirs:
    init_file = ROOT / subdir / "__init__.py"
    init_file.touch(exist_ok=True)

requirements = """pandas
numpy
scikit-learn
fastapi
uvicorn
streamlit
joblib
python-dotenv
nba_api
"""

(ROOT / "requirements.txt").write_text(requirements)

gitignore = """# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
venv/
.venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env

# macOS
.DS_Store

# Data and artifacts
data/*
!data/.gitkeep
!data/raw/
!data/raw/.gitkeep
!data/processed/
!data/processed/.gitkeep
!data/predictions/
!data/predictions/.gitkeep

artifacts/*
!artifacts/.gitkeep
# IDE
.vscode/
.idea/
"""

(ROOT / ".gitignore").write_text(gitignore)

config = '''from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"

# App / source directories
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
'''

(ROOT / "config.py").write_text(config)

(ROOT / "data" / ".gitkeep").touch(exist_ok=True)
(ROOT / "artifacts" / ".gitkeep").touch(exist_ok=True)

readme = """# BetSki

BetSki is an NBA betting risk analyzer that predicts game outcomes, estimates win probability, assigns a 1-10 risk score, and explains predictions using historical NBA data.

## Tech Stack

- Python
- pandas
- scikit-learn
- FastAPI
- Streamlit
- nba_api
"""

(ROOT / "README.md").write_text(readme)

print("BetSki project environment initialized successfully.")