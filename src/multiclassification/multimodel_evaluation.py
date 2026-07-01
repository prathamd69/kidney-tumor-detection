import json
import mlflow
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.utils import configLogger, loadYaml
from src.components import ModelPipelineFactory

logger = configLogger("multimodel_evaluation", "multimodel_evaluation.log")


def main():
    logger.info("Loading configuration files...")
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    # centralized pipeline factory for Multiclassification
    pipeline = ModelPipelineFactory(config, params, is_binaryClassification=False)

    # Pulling metadata and paths directly from factory properties
    trained_multiweights_path = pipeline.weights_path
    experiment_name, run_name = pipeline.experiment_meta
    metrics_path = pipeline.metrics_path

    mlflow.set_experiment(experiment_name=experiment_name)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Manufacturing the data stream and network graph using the factory
    logger.info("Building evaluation data streaming pipeline and model architecture...")
    eval_dataset, labels, model = pipeline.testing_components()

    logger.info("Loading trained weights from: %s", trained_multiweights_path)
    model.load_weights(str(trained_multiweights_path))

    logger.info("Generating predictions over evaluation dataset...")
    y_probs = model.predict(eval_dataset, verbose=1)
    
    # Process multi-class probabilities using argmax
    y_pred = np.argmax(y_probs, axis=1)
    y_true = labels
    
    logger.info("Starting active MLflow evaluation tracking session for Run: %s", run_name)
    with mlflow.start_run(run_name=run_name):

        # Calculate metrics using Macro averaging for multiclass layout
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average='macro'))
        rec = float(recall_score(y_true, y_pred, average='macro'))
        f1 = float(f1_score(y_true, y_pred, average='macro'))
        auc = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro'))  
        
        cm = confusion_matrix(y_true, y_pred)
        cm_payload = cm.tolist()

        logger.info("Metrics computed -> Acc: %.4f | Prec (Macro): %.4f | Rec (Macro): %.4f | F1 (Macro): %.4f", acc, prec, rec, f1)

        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_precision", prec)
        mlflow.log_metric("eval_recall", rec)
        mlflow.log_metric("eval_f1_score", f1)
        mlflow.log_metric("eval_roc_auc", auc)
        
        logger.info("Metrics successfully tracked to MLflow Experiment Dashboard.")

        metrics_payload = {
            "run_name": run_name,
            "accuracy": acc,
            "precision": prec,
            "recall_sensitivity": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm_payload
        }


        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics_history = json.load(f)
        else:
            metrics_history = []

        metrics_history.append(metrics_payload)

        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
                
        logger.info("Metrics report appended locally to %s", metrics_path)

        logger.info("Uploading metrics report artifact to MLflow server...")
        mlflow.log_artifact(str(metrics_path), artifact_path="multi_evaluation_reports")
        logger.info("Multiclass pipeline evaluation execution completed successfully.")


if __name__ == "__main__":
    main()