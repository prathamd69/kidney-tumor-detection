import numpy as np
import io
from PIL import Image
from pathlib import Path
from src.utils import configLogger, loadYaml 

try:
    # 1. Primary: Tries to load the lightweight engine (For Docker Deployment)
    import tflite_runtime.interpreter as tflite
except ImportError:
    # 2. Fallback: Loads the full TF engine (For Local Windows Testing)
    from tensorflow import lite as tflite
    
logger = configLogger("prediction", "prediction.log")

class ImageProcessor:

    @staticmethod
    def process_image(imagebytes: bytes, image_height: int, image_width: int) -> np.ndarray:
        try:
            image = io.BytesIO(imagebytes)
            image = Image.open(image).convert("RGB")
            image = image.resize((image_width, image_height))

            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0 

            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        
        except Exception as e:
            logger.error("Error occurred during standalone image preprocessing: %s", e)
            raise

class BinaryPredictor:
    def __init__(self) -> None:
        try:
            self.config = loadYaml(Path("config/config.yaml"))
            self.modelpath = self.config.artifacts_paths.litebinarymodel

            self.interpreter = tflite.Interpreter(model_path=str(self.modelpath))
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_shape = self.input_details[0]['shape'] 
            self.img_height = self.input_shape[1]
            self.img_width = self.input_shape[2]
            
            logger.info("BinaryPredictor initialized and TFLite tensors allocated successfully.")
            
        except Exception as e:
            logger.error("Failed to initialize BinaryPredictor: %s", e)
            raise

    def predict(self, imagebytes: bytes) -> dict:
        try:
            image_array = ImageProcessor.process_image(imagebytes, self.img_height, self.img_width)

            self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])

            prediction_probab = float(predictions[0][0])

            prediction_class = "Tumor" if prediction_probab > 0.5 else "Normal"
            confidence = prediction_probab if prediction_probab > 0.5 else (1.0 - prediction_probab)

            return {
                "status": "success",
                "prediction": prediction_class,
                "confidence": round(confidence, 4),
                "raw_probability": round(prediction_probab, 4)
            }
        
        except Exception as e:
            logger.exception("Binary Prediction layer failed execution: %s", e)
            return {"status": "error", "message": str(e)}
        

