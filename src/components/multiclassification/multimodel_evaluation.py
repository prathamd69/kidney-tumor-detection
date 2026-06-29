from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (confusion_matrix,
                             roc_auc_score, 
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)
import mlflow
import json
from src.utils import configLogger
from src.utils import loadFile, loadYaml
from src.components.multiclassification import testing

logger = configLogger("multimodel_evaluation", "multimodel_evaluation.log")

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    trained_multiweights_path = Path(config.model_paths.trained_multiweights_path )

    experiment_name = str(params.multiclass_model_params.experiment_name)
    run_name = str(params.multiclass_model_params.run_name)
    mlflow.set_experiment(experiment_name=experiment_name)

    metrics_path = Path(config.report_paths.multimodel_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building evaluation data streaming pipeline...")
    logger.info("Loading trained weights from %s", trained_multiweights_path)
    
    eval_dataset, labels, model = testing(config=config, params=params)
    model.load_weights(trained_multiweights_path)

    logger.info("Generating predictions over evaluation dataset...")
    y_probs = model.predict(eval_dataset, verbose=1)
    
    y_pred = np.argmax(y_probs, axis=1)
    y_true = labels
    
    logger.info("Starting active MLflow evaluation tracking session...")
    with mlflow.start_run(run_name=run_name):

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average='macro'))
        rec = float(recall_score(y_true, y_pred, average='macro'))
        f1 = float(f1_score(y_true, y_pred, average='macro'))
        auc = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro'))  
        cm = confusion_matrix(y_true, y_pred)

        cm_payload = cm.tolist()

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
                
        logger.info("Metrics dictionary successfully saved locally to %s", metrics_path)
        mlflow.log_artifact(str(metrics_path), artifact_path="multi_evaluation_reports")

if __name__ == "__main__":
    main()