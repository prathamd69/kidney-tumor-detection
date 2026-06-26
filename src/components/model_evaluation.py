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
import os
import json
from src.utils import configLogger, loadFile, loadYaml

logger = configLogger("model_evaluation", "model_evaluation.log")

def parse_image(file_path: str, label: int, img_shape: tuple) -> tuple:
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_shape)
        return image, label
    except Exception as e:
        logger.error("Failed to parse image file at %s: %s", file_path, e)
        raise

def create_eval_dataset(df: pd.DataFrame, images_dir: Path, batch_size: int, img_shape: tuple, is_binaryClassification : bool) -> tf.data.Dataset:
    """Assembles a streaming evaluation pipeline (NO shuffling needed for evaluation)."""
    file_paths = df['image_id'].apply(lambda x: str(images_dir / f"{x}.jpg")).astype(str).to_numpy()
    
    if is_binaryClassification:
        labels = df['binarytarget'].astype('int32').to_numpy()
    
    else:
        labels = df['target'].astype('int32').to_numpy()
    
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(file_paths, dtype=tf.string), 
                                                  tf.constant(labels, dtype=tf.int32)))
    
    dataset = dataset.map(lambda fp, lbl: parse_image(fp, lbl, img_shape), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    test_df = loadFile(Path(config.model_evaluation.test_data_path)) 
    images_dir = Path(config.model_training.images_dir)
    trained_binaryclass_model_path = Path(config.model_training.trained_binaryclass_model_path)
    trained_multiclass_model_path = Path(config.model_training.trained_multiclass_model_path)
    
    img_shape = (int(params.model_training.img_height), int(params.model_training.img_width))
    batch_size = int(params.model_training.batch_size)
    is_binaryClassification = bool(params.model_training.is_binaryClassification)

    experiment_name = str(config.experiment_tracking.experiment_name)
    run_name = str(config.experiment_tracking.run_name)
    mlflow.set_experiment(experiment_name=experiment_name)

    metrics_path = Path(config.reports.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building evaluation data streaming pipeline...")
    eval_dataset = create_eval_dataset(test_df, images_dir, batch_size, img_shape, is_binaryClassification)

    if is_binaryClassification:        

        logger.info("Loading trained model from %s", trained_binaryclass_model_path)
        model = tf.keras.models.load_model(str(trained_binaryclass_model_path))

        logger.info("Generating predictions over evaluation dataset...")
        y_probs = model.predict(eval_dataset, verbose=1).flatten()
    
        # Convert sigmoid probabilities to binary 0 or 1 classes (threshold = 0.5)
        y_pred = (y_probs > 0.5).astype(int)
        y_true = test_df['binarytarget'].astype('int32').to_numpy()
    
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

    else:

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