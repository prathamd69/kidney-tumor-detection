from pathlib import Path
import pandas as pd
import tensorflow as tf
import mlflow
from src.utils import configLogger
from src.utils import loadFile, loadYaml
from src.components.multiclassification import training

logger = configLogger("multimodel_training", "multimodel_training.log")
    
def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    #Loading all the paths
    trained_multiweights_path = Path(config.model_paths.trained_multiweights_path)
    
    # Configuring MLflow Tracking Experiments
    experiment_name = str(params.multiclass_model_params.experiment_name)
    run_name = str(params.multiclass_model_params.run_name)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Configuring early stopping, epochs
    epochs = params.multiclass_model_params.epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    min_delta=0.01,
    restore_best_weights=True)

    train_dataset, val_dataset, model = training(config=config, params=params)
    logger.info("Starting active MLflow tracking session...")
    with mlflow.start_run(run_name=run_name):
        params_dict = dict(params.multiclass_model_params)
        mlflow.log_params(params_dict)

        logger.info("Executing network execution layers across %s training epochs...", epochs)
        model.fit(
            train_dataset, 
            epochs=epochs,
            validation_data = val_dataset,
            callbacks = [early_stopping],
            verbose=1
        )

        weights_path = Path(config.model_paths.trained_multiweights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(weights_path))
        logger.info("Weights saved at: %s", weights_path)

if __name__ == "__main__":
    main()