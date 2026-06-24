# Kidney Tumor Detection Pipeline


[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-orange.svg)](https://dvc.org/)
![MLflow](https://img.shields.io/badge/MLflow-yellow.svg)

A highly optimized, production-ready deep learning pipeline designed to detect kidney tumors from medical imaging (CT scans) using a pre-trained **ResNet50V2** backbone. 

This repository leverages **TensorFlow / Keras** for core model execution, utilizes a custom-built low-RAM streaming data pipeline (`tf.data.Dataset`), and implements full experiment lifecycle tracking with **MLflow**.

---

## Project Architecture

The repository is built following modular, industry-standard MLOps design principles:

```text
kidney-tumor-detection/
├── config/
│   └── config.yaml           # Local path management variables
├── logs/                     # Component execution runtime logs
├── models/                   # Finalized serialized model artifacts (.h5/.keras)
├── reports/                  # Evaluation JSON scorecards
├── data/
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_processing.py
│   │   ├── model_building.py
│   │   ├── model_training.py
│   │   └── model_evaluation.py
│   ├── utils/
│   |    ├── __init__.py
│   |    └── logger.py         # Shared logger configs and YAML parsing utilities
|   |    └── file_utils.py
|   └── prediction/
|        ├── __init__.py
|        └── prediction.py
├── params.yaml               # Model training hyperparameter configurations
├── dvc.yaml   
├── dvc.lock  
├── requirements.txt          # Python dependency specifications
├── .gitignore
├── .dvcignore    
└── README.md
```

## Performance Optimization Highlights
-----------------------------------

### 1\. Zero-RAM Latent Streaming Pipeline

Medical imaging files can quickly saturate local system memory. To combat out-of-memory crashes (Segmentation fault (core dumped)), this pipeline executes a lazy-loading streaming mechanism:

*   **Pre-Map Shuffling:** Shuffling operations are run directly on tiny text-based string file paths inside memory rather than raw decoded image objects.
    
*   **On-Demand Decoding:** Images are dynamically read from disk and resized to target inputs only when a specific execution batch step requests them.
    
*   **Autotuned Prefetching:** Utilizes tf.data.AUTOTUNE to fetch and prepare batch $N+1$ on CPU threads while batch $N$ is actively calculating gradients.
    

### 2\. Multi-Tier Custom Exceptions & Logging

Every module within src/components/ is tightly bound to a dedicated component log matrix file (logs/). The runtime isolates unexpected structural schema errors or corrupted image headers via contextual CustomProjectExceptions without cleanly masking or swallowing errors blindly up the stack.

Quick Start Guide
--------------------

### 1\. Installation Environment

Clone the repository and install dependencies inside a local Python environment - 

`git clone https://github.com/prathamd69/kidney-tumor-detection.git`

`cd kidney-tumor-detection`

`pip install -r requirements.txt`

### 2\. Configure Your Parameters

Adjust network dimensions and batch properties inside params.yaml:

YAML

```model_training:```   

`batch_size: 32`
`img_height: 224`
`img_width: 224`
`epochs: 10`

### 3\. Run Pipeline Execution

Execute model training and track analytics via MLflow:

` dvc repro `
`mlflow server`

Navigate to http://localhost:5000 in your web browser to examine your tracking session run charts.

📊 Evaluation Metrics Tracked
-----------------------------

Final outputs compile directly into an isolated evaluation run within your MLflow tracking workspace and dump locally to reports/metrics.json. Key monitored metrics include:

*   **Recall (Sensitivity):** Critical for healthcare validation to guarantee false-negative profiles stay minimized - 0.9893
    
*   **Precision:** Minimizes false alarm benchmarks - 0.9027
    
*   **ROC-AUC Score:** Evaluates how cleanly the model separates abnormal tumor variants from standard normal scans - 0.9982

```{
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
