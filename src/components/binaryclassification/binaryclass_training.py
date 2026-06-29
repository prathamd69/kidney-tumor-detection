from pathlib import Path
import pandas as pd
import tensorflow as tf
import mlflow
from src.utils import configLogger
from src.utils import loadFile, loadYaml
from src.utils import create_training_dataset

logger = configLogger("binarymodel_training", "binarymodel_training.log")
    
def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    is_binaryClassification = bool(params.binaryclass_model_params.is_binaryClassification)

    base_binaryclass_model_path = Path(config.model_paths.base_binaryclass_model_path)
    trained_binaryclass_model_path = Path(config.model_paths.trained_binaryclass_model_path)

    train_df = loadFile(Path(config.data_paths.train_data_path))
    images_dir = Path(config.data_paths.images_dir)
    
    img_shape = (int(params.basic_model_params.img_height), int(params.basic_model_params.img_width))
    batch_size = int(params.binaryclass_model_params.batch_size)
    epochs = int(params.binaryclass_model_params.epochs)

    train_dataset, val_dataset = create_training_dataset(train_df, images_dir, batch_size, img_shape, is_binaryClassification)
    
    # Configuring MLflow Tracking Experiments
    experiment_name = str(params.binaryclass_model_params.experiment_name)
    run_name = str(params.binaryclass_model_params.run_name)

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True)

    mlflow.set_experiment(experiment_name=experiment_name)
    
    logger.info("Starting active MLflow tracking session...")
    with mlflow.start_run(run_name=run_name):
        params_dict = dict(params.binaryclass_model_params)
        mlflow.log_params(params_dict)

        logger.info("Loading un-trained compiled model structure from %s", base_binaryclass_model_path)
        model = tf.keras.models.load_model(str(base_binaryclass_model_path))

        logger.info("Executing network execution layers across %s training epochs...", epochs)
        model.fit(
            train_dataset, 
            epochs=epochs,
            validation_data = val_dataset,
            callbacks = [early_stopping],
            verbose=1
        )

        trained_binaryclass_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(trained_binaryclass_model_path))
        mlflow.tensorflow.log_model(model=model, artifact_path="binaryclassmodels") # type:ignore
        logger.info("Binaryclass trained architecture safely exported to local storage: %s", trained_binaryclass_model_path)

if __name__ == "__main__":
    main()