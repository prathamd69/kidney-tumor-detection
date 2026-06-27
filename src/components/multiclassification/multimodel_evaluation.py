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
from src.utils import create_dataset

logger = configLogger("binarymodel_evaluation", "binarymodel_evaluation.log")

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    test_df = loadFile(Path(config.data_paths.test_data_path)) 
    images_dir = Path(config.data_paths.images_dir)
    trained_multiclass_model_path = Path(config.model_paths.trained_multiclass_model_path)

    img_shape = (int(params.model_training.img_height), int(params.model_training.img_width))
    batch_size = int(params.model_training.batch_size)
    is_binaryClassification = bool(params.model_training.is_binaryClassification)

    experiment_name = str(config.experiment_paths.multiclass_experiment_name)
    run_name = str(config.experiment_paths.multiclass_run_name)
    mlflow.set_experiment(experiment_name=experiment_name)

    metrics_path = Path(config.reports.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building evaluation data streaming pipeline...")
    eval_dataset = create_dataset(test_df, images_dir, batch_size, img_shape, is_binaryClassification)

    logger.info("Loading trained model from %s", trained_multiclass_model_path)
    model = tf.keras.models.load_model(str(trained_multiclass_model_path))

    logger.info("Generating predictions over evaluation dataset...")
    y_probs = model.predict(eval_dataset, verbose=1)

    # Using np.argmax to extract classes from softmax values
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_df['target'].astype('int32').to_numpy()
    
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
