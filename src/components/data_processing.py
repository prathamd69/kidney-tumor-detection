from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import configLogger, loadFile, saveFile, loadYaml

logger = configLogger("data_processing", "data_processing.log")

def processing(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        columns_to_drop = ['Unnamed: 0', 'path', 'diag', 'Class']
        data = data.drop(columns=columns_to_drop, errors='ignore')

        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.debug("Raw data split into train and test data. Test Size: %s, Random State: %s", test_size, random_state)
        return train_data, test_data
    
    except Exception as e:
        logger.exception("Unexpected error while processing: %s", e)
        raise

def main():

    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    raw_path = Path(config.data_processing.raw_data_path)
    train_path = Path(config.data_processing.train_data_path)
    test_path = Path(config.data_processing.test_data_path)
    
    test_size = float(params.data_processing.test_size)
    random_state = int(params.data_processing.random_state)

    data = loadFile(raw_path)
    train_data, test_data = processing(data, test_size, random_state)

    saveFile(train_path, train_data)
    saveFile(test_path, test_data)
    logger.info("Train and Test data saved successfully based on configuration metrics.")

if __name__ == "__main__":
    main()