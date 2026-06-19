from pathlib import Path
import pandas as pd
import yaml
from box import ConfigBox
from .logger import configLogger

logger = configLogger("file_utils", "file_utils.log")


def loadFile(filePath: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(filePath)

    except FileNotFoundError:
        logger.exception("No file found at %s", filePath)
        raise
    
    except pd.errors.EmptyDataError:
        logger.exception("The CSV file at %s is empty or corrupted", filePath)
        raise

    except Exception as e:
        logger.exception("Unexpected error while loading file at %s : %s", filePath, e)
        raise


def saveFile(filePath: str | Path, data: pd.DataFrame) -> None:
    try:
        filePath = Path(filePath)
        filePath.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(filePath, index=False)
        logger.debug("File saved successfully at %s", filePath)

    except Exception as e:
        logger.exception("Unexpected error while saving file at %s : %s", filePath, e)
        raise

def loadYaml(path: Path) -> ConfigBox:
    try:
        with open(path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info("YAML file loaded successfully from: %s", path)
            return ConfigBox(content)
        
    except Exception as e:
        logger.exception("Failed to load YAML file: %s", e)
        raise