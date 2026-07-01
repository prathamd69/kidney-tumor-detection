import tensorflow as tf
from pathlib import Path
from box import ConfigBox
from src.utils import configLogger, loadYaml
from src.components import ModelPipelineFactory

logger = configLogger("model_optimization", "model_optimization.log")

class ModelOptimizer:
    def __init__(self, config: ConfigBox, params: ConfigBox, is_binaryClassification: bool):
        try:
            self.pipeline = ModelPipelineFactory(config, params, is_binaryClassification)
            self.weights = Path(self.pipeline.weights_path)
            logger.info(f"Loading weights and configuring model from {self.weights}")
            
            self.model = self.pipeline.model_config()
            self.model.load_weights(str(self.weights))

            # Setting up return path
            if is_binaryClassification:
                self.litemodel = Path(config.artifacts_paths.litebinarymodel)    
            else:
                self.litemodel = Path(config.artifacts_paths.litemultimodel)
                
        except Exception as e:
            logger.error(f"Failed to initialize ModelOptimizer or load weights: {e}")
            raise


    def make_tflite(self):
        try:
            logger.info("Initializing TFLite Converter with default optimizations")
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Apply quantization to shrink the file footprint
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            logger.info("Converting model...")
            tflite_model = converter.convert()

            self.litemodel.parent.mkdir(parents=True, exist_ok=True)

            with open(self.litemodel, "wb") as f:
                f.write(tflite_model)
                
            logger.info(f"Optimized TFLite model successfully saved to {self.litemodel}")
            
        except tf.errors.OpError as e:
            logger.error(f"TensorFlow conversion failed: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error during model conversion or saving: {e}")
            raise
    
    @property
    def litemodelpath(self)-> Path:
        return self.litemodel


def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    try:
        binarypipeline = ModelOptimizer(config=config, params=params, is_binaryClassification=True)
        binarypipeline.make_tflite()
   
        multipipeline = ModelOptimizer(config=config, params=params, is_binaryClassification=False)
        multipipeline.make_tflite()
    
    except Exception as e:
        logger.exception("Error while optimizing tf models : %s", e)
        raise
    
if __name__ == "__main__":
    main()
