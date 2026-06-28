import numpy as np
from pathlib import Path
import tensorflow as tf
from src.utils import configLogger
from src.utils import loadYaml

logger = configLogger("prediction", "prediction.log")

class ImageProcessor:
    """Handles configuration loading and transforms raw image bytes into a reusable tensor once."""
    def __init__(self) -> None:
        try:
            self.params = loadYaml(Path("params.yaml"))
            self.image_height = int(self.params.basic_model_params.img_height)
            self.image_width = int(self.params.basic_model_params.img_width)
            
        except Exception as e:
            logger.exception("Failed to load configurations for ImageProcessor: %s", e)
            raise

    def process(self, imagebytes: bytes) -> tf.Tensor:
        try:
            image = tf.image.decode_jpeg(imagebytes, channels=3)
            image = tf.image.resize(image, (self.image_height, self.image_width))
            image = image / 255.0
            image = tf.expand_dims(image, axis=0)
            return image
        
        except Exception as e:
            logger.error("Error occurred during standalone image preprocessing: %s", e)
            raise


class BinaryPredictor:
    def __init__(self) -> None:
        try:
            config = loadYaml(Path("config/config.yaml"))
            self.model_path = Path(config.model_paths.trained_binary_model_path)
            
            logger.info("Loading serialized Keras binary model from %s", self.model_path)
            self.model = tf.keras.models.load_model(str(self.model_path))
            
        except Exception as e:
            logger.exception("Failed to initialize the BinaryPredictor: %s", e)
            raise

    def predict(self, processed_img: tf.Tensor) -> dict:
        try:
            prediction_probab = float(self.model.predict(processed_img)[0][0])

            prediction_class = "Tumor" if prediction_probab > 0.5 else "Normal"
            confidence = prediction_probab if prediction_class == "Tumor" else (1.0 - prediction_probab)

            logger.info("Inference successful (Binary) - Predicted: %s with confidence %.4f", prediction_class, confidence)
            
            return {
                "status": "success",
                "prediction": prediction_class,
                "confidence": round(confidence, 4),
                "raw_probability": round(prediction_probab, 4)
            }
        
        except Exception as e:
            logger.exception("Binary Prediction layer failed execution: %s", e)
            return {"status": "error", "message": str(e)}


class MultiPredictor:
    def __init__(self) -> None:
        try:
            config = loadYaml(Path("config/config.yaml"))
            params = loadYaml(Path("params.yaml"))
            
            self.model_path = Path(config.model_paths.trained_multiclass_model_path)
            self.relation_map = dict(params.data_processing.relation_map)
            
            logger.info("Loading serialized Keras multi-class model from %s", self.model_path)
            self.model = tf.keras.models.load_model(str(self.model_path))
            
        except Exception as e:
            logger.exception("Failed to initialize the MultiPredictor: %s", e)
            raise

    def predict(self, processed_img: tf.Tensor) -> dict:
        try:
            prediction_probabs = self.model.predict(processed_img)[0]
            
            prediction_idx = int(np.argmax(prediction_probabs))
            prediction_class = self.relation_map[prediction_idx]
            confidence = float(prediction_probabs[prediction_idx])

            raw_probs_dict = {
                self.relation_map[idx]: round(float(prob), 4) 
                for idx, prob in enumerate(prediction_probabs)
            }

            logger.info("Inference successful (Multi) - Predicted: %s with confidence %.4f", prediction_class, confidence)
            
            return {
                "status": "success",
                "prediction": prediction_class,
                "confidence": round(confidence, 4),
                "raw_probability": raw_probs_dict
            }
        
        except Exception as e:
            logger.exception("Multi Prediction layer failed execution: %s", e)
            return {"status": "error", "message": str(e)}