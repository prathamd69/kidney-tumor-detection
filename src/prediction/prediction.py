import numpy as np
from pathlib import Path
import tensorflow as tf
from src.utils import configLogger
from src.utils import loadYaml

logger = configLogger("prediction","prediction.log")

class Predictor:
    def __init__(self) -> None:
        try:
            self.config = loadYaml(Path("config/config.yaml"))
            self.params = loadYaml(Path("params.yaml"))

            self.model_path = Path(self.config.model_training.trained_model_path)
            self.image_height = int(self.params.model_training.image_height)
            self.image_width = int(self.params.model_training.image_width)

            logger.info("Loading serialized Keras model weights from %s", self.model_path)
            self.model = tf.keras.models.load_model(str(self.model_path))
        
        except Exception as e:
            logger.exception("Failed to initialize the prediction pipeline : %s", e)
            raise
