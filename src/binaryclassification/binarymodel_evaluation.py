import json
import mlflow
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

logger = configLogger("binarymodel_evaluation", "binarymodel_evaluation.log")


def main():
    logger.info("Loading configuration files...")
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    # centralized pipeline factory
    pipeline = ModelPipelineFactory(config, params, is_binaryClassification=True)

    # Pulling metadata and paths directly from factory properties
    binaryweights = pipeline.weights_path
    experiment_name, run_name = pipeline.experiment_meta

    mlflow.set_experiment(experiment_name=experiment_name)

    metrics_path = pipeline.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Manufacturing the data stream and network graph using the factory
    logger.info("Building evaluation data streaming pipeline and model architecture...")
    eval_dataset, labels, model = pipeline.testing_components()

    logger.info("Loading trained weights from: %s", binaryweights)
    model.load_weights(str(binaryweights))

    logger.info("Generating predictions over evaluation dataset...")
    y_probs = model.predict(eval_dataset, verbose=1).flatten()

    # Convert sigmoid probabilities to binary 0 or 1 classes (threshold = 0.5)
    y_pred = (y_probs > 0.5).astype(int)
    y_true = labels

    logger.info("Starting active MLflow evaluation tracking session for Run: %s", run_name)
    with mlflow.start_run(run_name=run_name):

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred))
        rec = float(recall_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, y_probs))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = map(int, cm.ravel())

        logger.info("Metrics computed -> Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f", acc, prec, rec, f1)

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
                "true_tumor_tp": tp,
            },
        }

        # Handling metrics compilation history append logic
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
        mlflow.log_artifact(str(metrics_path), artifact_path="binary_evaluation_reports")
        logger.info("Binary pipeline evaluation execution completed successfully.")


if __name__ == "__main__":
    main()