from functools import partial
from pathlib import Path
import warnings

from dotenv import load_dotenv
from loguru import logger
from tqdm import TqdmExperimentalWarning

# Load environment variables from .env file if it exists
load_dotenv()

# Ignore tqdm.rich warnings
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

ATLASET_DATASET = EXTERNAL_DATA_DIR / "atlasia_Atlaset"

MODELS_DIR = PROJ_ROOT / "models"
PRETRAINED_MODEL = MODELS_DIR / "pretrained"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
	from tqdm.rich import tqdm

	logger.remove(0)
	logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
	pass
