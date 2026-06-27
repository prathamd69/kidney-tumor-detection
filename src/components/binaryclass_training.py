from pathlib import Path
import pandas as pd
import tensorflow as tf
import mlflow
from src.utils import configLogger
from src.utils import loadFile, loadYaml
from src.utils import create_dataset

logger = configLogger("model_training", "model_training.log")
    
def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    is_binaryClassification = bool(params.model_training.is_binaryClassification)

    base_binaryclass_model_path = Path(config.model_training.base_binaryclass_model_path)
    trained_binaryclass_model_path = Path(config.model_training.trained_binaryclass_model_path)

    train_df = loadFile(Path(config.model_training.train_data_path))
    images_dir = Path(config.model_training.images_dir)
    
    img_shape = (int(params.model_training.img_height), int(params.model_training.img_width))
    batch_size = int(params.model_training.batch_size)
    epochs = int(params.model_training.epochs)

    train_dataset = create_dataset(train_df, images_dir, batch_size, img_shape, is_binaryClassification)
    
    # Configuring MLflow Tracking Experiments
    experiment_name = str(config.experiment_tracking.experiment_name)
    run_name = str(config.experiment_tracking.run_name)

    mlflow.set_experiment(experiment_name=experiment_name)
    
    logger.info("Starting active MLflow tracking session...")
    with mlflow.start_run(run_name=run_name):
        params_dict = dict(params.model_training)
        mlflow.log_params(params_dict)

        logger.info("Loading un-trained compiled model structure from %s", base_binaryclass_model_path)
        model = tf.keras.models.load_model(str(base_binaryclass_model_path))

        logger.info("Executing network execution layers across %s training epochs...", epochs)
        model.fit(
            train_dataset, 
            epochs=epochs,
            verbose=1
        )


        trained_binaryclass_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(trained_binaryclass_model_path))
        mlflow.tensorflow.log_model(model=model, artifact_path="binaryclassmodels") # type:ignore
        logger.info("Binaryclass trained architecture safely exported to local storage: %s", trained_binaryclass_model_path)

if __name__ == "__main__":
    main()