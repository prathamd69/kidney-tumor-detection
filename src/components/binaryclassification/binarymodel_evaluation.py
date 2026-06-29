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
from src.components.binaryclassification import testing

logger = configLogger("binarymodel_evaluation", "binarymodel_evaluation.log")

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    trained_binaryweights_path = Path(config.model_paths.trained_binaryweights_path )

    experiment_name = str(params.binaryclass_model_params.experiment_name)
    run_name = str(params.binaryclass_model_params.run_name)
    mlflow.set_experiment(experiment_name=experiment_name)

    metrics_path = Path(config.report_paths.binarymodel_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building evaluation data streaming pipeline...")
    logger.info("Loading trained weights from %s", trained_binaryweights_path)
    
    eval_dataset, labels, model = testing(config=config, params=params)
    model.load_weights(trained_binaryweights_path)

    logger.info("Generating predictions over evaluation dataset...")
    y_probs = model.predict(eval_dataset, verbose=1).flatten()
    
    # Convert sigmoid probabilities to binary 0 or 1 classes (threshold = 0.5)
    y_pred = (y_probs > 0.5).astype(int)
    y_true = labels
    
    logger.info("Starting active MLflow evaluation tracking session...")
    with mlflow.start_run(run_name=run_name):

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred))
        rec = float(recall_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, y_probs))
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = map(int, cm.ravel())

        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_precision", prec)
        mlflow.log_metric("eval_recall", rec)
        mlflow.log_metric("eval_f1_score", f1)
        mlflow.log_metric("eval_roc_auc", auc)
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("false_negatives", fn)
        mlflow.log_metric("true_positives", tp)
        
        logger.info("Metrics successfully tracked to MLflow Experiment Dashboard.")

        metrics_payload = {
            "run_name": run_name,
            "accuracy": acc,
            "precision": prec,
            "recall_sensitivity": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": {
                "true_normal_tn": tn,
                "false_tumor_fp": fp,
                "false_normal_fn": fn,
                "true_tumor_tp": tp
                }
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
        mlflow.log_artifact(str(metrics_path), artifact_path="binary_evaluation_reports")

if __name__ == "__main__":
    main()