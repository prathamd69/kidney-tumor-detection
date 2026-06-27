# Kidney Tumor Detection — MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) [![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-orange.svg)](https://dvc.org/) ![MLflow](https://img.shields.io/badge/MLflow-yellow.svg)

A production-ready, end-to-end Machine Learning Operations (MLOps) pipeline for automated detection and classification of kidney tumors from CT imagery. Built with TensorFlow/Keras, the project implements two strictly isolated neural workflows: a binary classification pipeline for tumor presence detection and a multiclass pipeline for fine-grained tumor categorization. DVC orchestrates independent pipeline DAGs so experiments are reproducible and artifact collisions are avoided; MLflow tracks experiments, metrics, and artifacts. Trained .keras artifacts are consumed by a prediction module and served via a lightweight web front-end.

Table of contents
- Overview
- Key features
- Repo layout
- Quickstart
- Configuration and parameters
- DVC pipeline (stages)
- Components (what each script does)
  - Data processing
  - Binary classification: base / training / evaluation
  - Multi-class classification: base / training / evaluation
  - Prediction & web app
- Logging & reproducibility
- Example evaluation metrics
- Contributing / License / Contact

Overview
--------
This repository is designed for safe, concurrent experiments and production readiness:
- Two isolated model workflows (binary vs multiclass) so changing params for one model does not invalidate cached artifacts for the other.
- Memory-efficient tf.data streaming that shuffles on file paths, decodes images on-demand and uses AUTOTUNE to avoid OOMs.
- DVC for dataset/artifact versioning and stage orchestration.
- MLflow integration for experiment, metrics and artifact logging.
- Prediction module and a simple web UI to serve results to users.

Key features
------------
- ResNet50V2 backbone (pretrained weights) used as the base encoder for both pipelines.
- Binary classification pipeline: detects tumor presence quickly and reliably.
- Multiclass pipeline: categorizes input scans into clinical classes (e.g., Cyst, Normal, Stone, Tumor).
- Strict separation of base model building, training and evaluation stages for each workflow.
- Centralized logging and robust exception handling.
- Reproducible, DVC-orchestrated experiments with MLflow tracking.

Repository layout
-----------------
```
kidney-tumor-detection/
├── config/
│   └── config.yaml                 # Environment: local paths, model/report locations
├── data/
│   └── raw/                        # Raw imagery and csv
├── dvc.yaml                         # DVC pipeline stages
├── params.yaml                      # Experiment hyperparameters
├── src/
│   ├── components/
│   │   ├── data_processing.py
│   │   ├── binaryclassification/
│   │   │   ├── binarybase_model.py
│   │   │   ├── binaryclass_training.py
│   │   │   └── binarymodel_evaluation.py
│   │   └── multiclassification/
│   │       ├── multibase_model.py
│   │       ├── multiclass_training.py
│   │       └── multimodel_evaluation.py
│   ├── prediction/
│   │   └── prediction.py
│   └── utils/
│       ├── logger.py
│       └── file_utils.py
├── models/                          # saved .keras artifacts
├── reports/                         # evaluation JSONs and history
├── app.py                           # small web server to render templates/index.html
├── templates/
│   └── index.html                   # front-end for prediction results
├── requirements.txt
└── README.md
```

Quickstart
----------
1. Clone and install:
```bash
git clone https://github.com/prathamd69/kidney-tumor-detection.git
cd kidney-tumor-detection
pip install -r requirements.txt
```

2. Configure:
- Edit `config/config.yaml` to set local paths for data, models, and reports.
- Review `params.yaml` to set hyperparameters, batch sizes, epochs and mapping values.

3. Run pipeline (recommended with DVC):
```bash
# execute full, reproducible pipeline
dvc repro
```

4. Start MLflow to inspect runs:
```bash
mlflow server
# open http://localhost:5000 to view experiments and logged artifacts
```

Configuration and parameters
----------------------------
- `config/config.yaml`: centralizes file-system paths used across stages (data paths, model paths, report paths).
- `params.yaml`: controls hyperparameters and pipeline behavior (test split, model hyperparams, optimizer, learning rate, batch size, epochs, experiment names and run names).

Example snippet (from params.yaml):
```yaml
binaryclass_model_params:
  is_binaryClassification: True
  loss: "binary_crossentropy"
  optimizer: "Adam"
  batch_size: 32
  epochs: 5
  learning_rate: 0.005
  experiment_name: binaryclass_CNNs
  run_name: RESNET50V2_Evaluation_1

multiclass_model_params:
  is_binaryClassification: False
  loss: "sparse_categorical_crossentropy"
  optimizer: "Adam"
  num_classes: 4
  batch_size: 32
  epochs: 10
  learning_rate: 0.005
  experiment_name: multiclass_CNNs
  run_name: RESNET50V2_Evaluation_1
```

DVC pipeline (high level)
--------------------------
The `dvc.yaml` defines independent stages for binary and multiclass workflows plus shared preprocessing:
- data_processing: creates train/test CSVs from raw tabular inputs
- binary_base_model_building → binaryclass_training → binaryclass_evaluation
- multi_base_model_building → multiclass_training → multiclass_evaluation

Each model's base-building, training and evaluation stages have discrete outs (e.g., `models/binaryclass/base_binary_model.keras` and `models/binaryclass/trained_binary_model.keras`) so caching and parallel experiments are safe.

Components & what they do
-------------------------
- src/components/data_processing.py
  - Reads raw tabular dataset, applies column filtering, maps multi-class labels to binary (as configured), and writes `data/final/train.csv` and `data/final/test.csv`.
  - Uses sklearn's train_test_split. Logging via centralized logger.

- src/components/binaryclassification/*
  - binarybase_model.py: builds and compiles a ResNet50V2-based binary model (frozen backbone) and saves base `.keras` artifact.
  - binaryclass_training.py: loads base model, constructs a tf.data dataset (low-RAM streaming), trains using hyperparameters in `params.yaml`, logs params and model to MLflow, and saves trained artifact.
  - binarymodel_evaluation.py: creates evaluation dataset, produces prediction probabilities, thresholds to classes, computes accuracy/precision/recall/f1/roc_auc and confusion matrix, logs metrics to MLflow and appends to a local evaluation JSON under `reports/`.

- src/components/multiclassification/*
  - multibase_model.py: builds compiled ResNet50V2 multiclass base (softmax output) and saves base artifact.
  - multiclass_training.py / multimodel_evaluation.py: training and evaluation analogous to binary pipeline but for multi-class labels (uses categorical/sparse categorical loss as configured).

- src/utils/logger.py
  - Centralized logger that auto-creates `logs/` and standardizes file and console handlers. Used across components.

- src/utils/file_utils.py (utility functions)
  - Utilities for reading/writing CSVs, YAML, and building tf.data datasets with on-demand decoding & augmentation (used by training/eval scripts).

- src/prediction/prediction.py
  - Loads trained `.keras` artifacts, applies the same preprocessing pipeline, and returns prediction probabilities/labels. Intended to be used by `app.py` to serve predictions via `templates/index.html`.

Logging & reproducibility
------------------------
- All components write dedicated logs to `logs/` (e.g., `binarybase_model.log`, `binarymodel_training.log`).
- DVC tracks data and model files; change one pipeline's params while preserving cache for the other.
- MLflow captures experiment params, metrics and artifacts for every run and evaluation.

Example evaluation metrics (JSON)
---------------------------------
Evaluation runs are appended to `reports/metrics.json`. Example entry created by the binary evaluation stage:
```json
{
    "run_name": "RESNET50V2_Evaluation_1",
    "accuracy": 0.9787917737789203,
    "precision": 0.9027552674230146,
    "recall_sensitivity": 0.9893428063943162,
    "f1_score": 0.9440677966101695,
    "roc_auc": 0.998217529668933,
    "confusion_matrix": {
        "true_normal_tn": 2489,
        "false_tumor_fp": 60,
        "false_normal_fn": 6,
        "true_tumor_tp": 557
    }
}
```

Running individual stages (examples)
------------------------------------
- Run only data processing:
```bash
python -m src.components.data_processing
# or via dvc
dvc repro data_processing
```

- Build binary base model:
```bash
python -m src.components.binaryclassification.binarybase_model
```

- Train binary classifier:
```bash
python -m src.components.binaryclassification.binaryclass_training
```

- Evaluate binary classifier:
```bash
python -m src.components.binaryclassification.binarymodel_evaluation
```

- Full pipeline:
```bash
dvc repro
```

Serving predictions (quick)
---------------------------
- The prediction module (`src/prediction/prediction.py`) consumes `.keras` artifacts and the same preprocessing pipeline used for training.
- A simple app (`app.py`) wires prediction to `templates/index.html` to provide a minimal UI for uploading/scoring images.

Best practices & tips
---------------------
- Use DVC to pull the exact dataset version associated with a run before evaluating reproductions.
- Keep `params.yaml` and `config/config.yaml` under version control (but do not commit large raw image data).
- Start MLflow server before running long training jobs to capture runs reliably.

Contributing
------------
Contributions, bug reports and feature requests are welcome. Please:
1. Open an issue describing the change or bug.
2. Fork the repo, create a feature branch, add tests if applicable.
3. Open a PR and reference the issue.

Contact
-------
Repo owner: prathamd69 (GitHub).
For questions about reproduction, DVC usage or MLflow experiments, please open an issue in this repository.

Acknowledgements
----------------
Built with TensorFlow/Keras, DVC and MLflow. Inspired by MLOps best practices to enable reproducible, production-ready imaging models.