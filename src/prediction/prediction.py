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
            self.image_height = int(self.params.model_training.img_height)
            self.image_width = int(self.params.model_training.img_width)

            logger.info("Loading serialized Keras model weights from %s", self.model_path)
            self.model = tf.keras.models.load_model(str(self.model_path))
        
        except Exception as e:
            logger.exception("Failed to initialize the prediction pipeline : %s", e)
            raise
    
    def process_image(self, imagebytes : bytes) -> tf.Tensor:
        try:
            image = tf.image.decode_jpeg(imagebytes, channels=3)
            image = tf.image.resize(image, (self.image_height, self.image_width))
            image = image/255.0
            image = tf.expand_dims(image, axis=0)
            return image
        
        except Exception as e:
            logger.error("Error occurred during in-memory image preprocessing: %s", e)
            raise

    def predict(self, imagebytes : bytes) -> dict:
        try:
            processed_img = self.process_image(imagebytes)
            prediction_probab = float(self.model.predict(processed_img)[0][0])

            prediction_class = "Tumor" if prediction_probab > 0.5 else "Normal"

            confidence = prediction_probab if prediction_class == "Tumor" else (1.0 - prediction_probab)

            logger.info("Inference successful. Predicted: %s with confidence %4f", prediction_class, confidence)
            return {
                "status": "success",
                "prediction": prediction_class,
                "confidence": round(confidence, 4),
                "raw_probability": round(prediction_probab, 4)
            }
        
        except Exception as e:
            logger.exception("Prediction layer failed execution : %s", e)
            return {"status": "error", "message": str(e)}
