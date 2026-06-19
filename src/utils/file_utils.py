from pathlib import Path
import pandas as pd
from .logger import configLogger

logger = configLogger("file_utils", "file_utils.log")


def loadFile(filePath: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(filePath)

    except FileNotFoundError:
        logger.exception("No file found at %s", filePath)
        raise

    except Exception:
        logger.exception("Unexpected error while loading file at %s", filePath)
        raise


def saveFile(filePath: str | Path, data: pd.DataFrame) -> None:
    try:
        filePath = Path(filePath)
        filePath.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(filePath, index=False)
        logger.debug("File saved successfully at %s", filePath)

    except Exception:
        logger.exception("Unexpected error while saving file at %s", filePath)
        raise