import tensorflow as tf
import mlflow
from pathlib import Path
from src.utils import configLogger, loadYaml, ModelPipelineFactory

logger = configLogger("binarymodel_training", "binarymodel_training.log")

def main():
    logger.info("Loading configuration files...")
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))
    
    # centralized pipeline factory
    pipeline = ModelPipelineFactory(config, params, is_binaryClassification=True)

    logger.info("Building model architecture and data streams...")
    train_ds, val_ds, model = pipeline.training_components()
    logger.info("Assets ready. Model compiled. Training dataset initialized.")

    binaryweights = pipeline.weights_path
    experiment_name, run_name = pipeline.experiment_meta
    modelparams = pipeline.getparams

    mlflow.set_experiment(experiment_name=experiment_name)    

    logger.info("Starting MLflow tracking session for Run: %s", run_name)
    with mlflow.start_run(run_name=run_name):
        # Casting lists to strings so MLflow doesn't reject them
        flat_params = {k: str(v) if isinstance(v, list) else v for k, v in dict(modelparams).items()}
        mlflow.log_params(flat_params)

        logger.info("Commencing model training loop for %d epochs.", pipeline.epochs)
        history = model.fit(
            train_ds, 
            epochs=pipeline.epochs,
            validation_data=val_ds,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                min_delta=0.01,
                restore_best_weights=True
            )],
            verbose=1
        )
        
        final_val_acc = history.history['val_accuracy'][-1]
        logger.info("Training loop finalized. Best Val Accuracy achieved: %.4f", final_val_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)

        logger.info("Exporting weights locally to: %s", binaryweights)
        binaryweights.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(binaryweights))

        logger.info("Uploading weights artifact to MLflow server...")
        mlflow.log_artifact(str(binaryweights), artifact_path="binary_weights")
        logger.info("Binary pipeline execution completed successfully.")

if __name__ == "__main__":
    main()