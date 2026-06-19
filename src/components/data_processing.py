from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import configLogger, loadFile, saveFile

logger = configLogger("data_processing", "data_processing.log")

def processing(data : pd.DataFrame, testsize : float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        data = data.drop(columns=['Unnamed: 0','path','diag','Class'])
        train_data, test_data = train_test_split(data, test_size=testsize, random_state=27)
        logger.debug("Raw data splitted into train and test data. Test Size : %s", testsize)
        return train_data, test_data
    
    except Exception as e:
        logger.exception("Unexpected error while processing : %s", e)
        raise

def main():
    raw_path = Path("data/raw/kidneyData.csv")
    train_path = Path("data/final/train.csv")
    test_path = Path("data/final/test.csv")

    data = loadFile(raw_path)
    train_data, test_data = processing(data,0.2)

    saveFile(train_path, train_data)
    saveFile(test_path, test_data)
    logger.info("Train and Test data saved.")


if __name__ == "__main__":
    main()