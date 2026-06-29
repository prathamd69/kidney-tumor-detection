import tensorflow as tf
import mlflow
from pathlib import Path
from src.utils import configLogger, loadYaml
from src.components.binaryclassification import training

logger = configLogger("binarymodel_training", "binarymodel_training.log")

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    trained_binaryweights_path = Path(config.model_paths.trained_binaryweights_path)
    
    experiment_name = str(params.binaryclass_model_params.experiment_name)
    run_name = str(params.binaryclass_model_params.run_name)
    mlflow.set_experiment(experiment_name=experiment_name)

    train_ds, val_ds, model = training(config=config, params=params)

    logger.info("Starting MLflow tracking session for Binary Classification...")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(dict(params.binaryclass_model_params))

        history = model.fit(
            train_ds, 
            epochs=int(params.binaryclass_model_params.epochs),
            validation_data=val_ds,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                min_delta=0.01,
                restore_best_weights=True
            )],
            verbose=1
        )

        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])

        trained_binaryweights_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(trained_binaryweights_path))

        mlflow.log_artifact(str(trained_binaryweights_path), artifact_path="binary_weights")
        logger.info("Binary model weights saved and logged successfully.")

if __name__ == "__main__":
    main()