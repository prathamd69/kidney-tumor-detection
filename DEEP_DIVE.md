From 1.2GB to 4.6MB: Engineering a Production-Ready ML Pipeline for Kidney Tumor Detection
==========================================================================================

Training a deep learning model to 97.9% accuracy is an incredible milestone. But as I quickly learned, a Jupyter Notebook is not a production environment.

This project started as a computer vision challenge: classifying CT kidney scans using an **EfficientNetB0** architecture. After training on a dataset of 12,446 images, the metrics were phenomenal—**97.9% accuracy, 94.4% F1-score, and a 0.998 ROC-AUC**. The model successfully isolated binary and multiclass workflows for concurrent evaluation.

But getting those weights out of a notebook and into a live, automated, and containerized web API exposed the massive gap between Machine Learning and ML Engineering. Here is the breakdown of the critical bottlenecks encountered during deployment and the architectural shifts required to solve them.

### Bottleneck 1: The 1.2GB TensorFlow Anchor

The initial deployment strategy was standard: build a FastAPI wrapper, list the dependencies in requirements.txt, and containerize it using Docker.

The build immediately ground to a halt. TensorFlow is a massive, heavy framework designed for training, pulling in gigabytes of CUDA dependencies and graph execution logic. Pushing a multi-gigabyte Docker image just to run inference on a single web endpoint is incredibly inefficient and costly for cloud scaling.

**The Fix:** Post-Training Quantization.Instead of bringing the training framework to production, I decoupled them. I converted the trained EfficientNetB0 weights into a **TensorFlow Lite (.tflite)** artifact. This dropped the deployment model size down to a staggering **4.6MB**. Consequently, TensorFlow was completely wiped from the production requirements.txt, resulting in a lightning-fast, ultra-lightweight Docker container.

### Bottleneck 2: The "Dev vs. Prod" Boundary

In an MLOps repository, your DVC pipelines, training scripts, and dataset builders live in the same space as your web server logic. Initially, the Docker build was copying everything into the production container.

**The Fix:** Strict .dockerignore Implementation.By aggressively filtering the repository at the Docker level, I established a clear security and optimization boundary. The training components (src/components, src/binaryclassification) were blocked. The only artifacts sent to the live server were the web app (app.py), the inference engine, and the 4.6MB TFLite weights. The repository remained the brain; the Docker container became the muscle.

### Bottleneck 3: The FastAPI Lifespan Trap in Pytest

With CI/CD decoupled, I wrote a pytest suite to fire dummy images at the /predict/binary and /predict/detailed endpoints before deployment.

The pipeline immediately failed with a 500 Internal Server Error. The logs revealed a bizarre issue: AttributeError: 'State' object has no attribute 'binary\_pred'.

In production, the FastAPI application utilized an asynchronous @asynccontextmanager lifespan function to load the TFLite models into memory _only_ when the server booted up. But when pytest initialized the TestClient(app) globally, the startup event never fired. The server was trying to predict on models that didn't exist in memory.

**The Fix:** Pytest Context Fixtures.The testing architecture was rewritten to force FastAPI through its full startup and shutdown sequence. By wrapping the TestClient inside a @pytest.fixture using a Python with block, the models successfully loaded into the application state before the dummy images were processed:

```python
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
```


### The Final Architecture

Today, the system runs as a completely automated, zero-touch deployment pipeline.

When a new model is trained or a new feature is added on the dev branch, opening a Pull Request spins up a GitHub Actions runner. It executes the Pytest suite, simulating image uploads to verify endpoint integrity and JSON schemas. Once a human reviews the code and clicks "Merge", the system automatically builds a highly optimized, TensorFlow-free Docker image and ships it to Docker Hub.

The gap between data science and production has been completely closed.