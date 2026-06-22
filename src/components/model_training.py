from pathlib import Path
import pandas as pd
import tensorflow as tf
import mlflow, mlflow.keras
from src.utils import configLogger, loadFile, loadYaml

logger = configLogger("model_training", "model_training.log")

def parse_image(file_path: str, label: int, img_shape: tuple) -> tuple:
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_shape)
        return image, label
    
    except Exception as e:
        logger.error("Failed to parse image file at %s: %s", file_path, e)
        raise

def create_dataset(df: pd.DataFrame, images_dir: Path, batch_size: int, img_shape: tuple, is_training: bool = True) -> tf.data.Dataset:
    """Assembles a highly optimized, prefetched streaming data pipeline."""
    file_paths = df['image_id'].apply(lambda x: str(images_dir / f"{x}.jpg")).astype(str).to_numpy()
    labels = df['finaltarget'].astype('int32').to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(file_paths, dtype=tf.string), 
                                                  tf.constant(labels, dtype=tf.int32)))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.map(lambda fp, lbl: parse_image(fp, lbl, img_shape), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    train_df = loadFile(Path(config.model_training.train_data_path))
    images_dir = Path(config.model_training.images_dir)
    base_model_path = Path(config.model_training.base_model_path)
    trained_model_path = Path(config.model_training.trained_model_path)
    
    img_shape = (int(params.model_training.img_height), int(params.model_training.img_width))
    batch_size = int(params.model_training.batch_size)
    epochs = int(params.model_training.epochs)

    train_dataset = create_dataset(train_df, images_dir, batch_size, img_shape, is_training=True)
    
    # Configuring MLflow Tracking Experiments
    experiment_name = str(config.experiment_tracking.experiment_name)
    run_name = str(config.experiment_tracking.run_name)

    mlflow.set_experiment(experiment_name=experiment_name)
    
    logger.info("Starting active MLflow tracking session...")
    with mlflow.start_run(run_name=run_name):
        params = dict(params.model_training)
        mlflow.log_params(params)
        mlflow.log_model_params

        logger.info("Loading un-trained compiled model structure from %s", base_model_path)
        model = tf.keras.models.load_model(str(base_model_path))

        logger.info("Executing network execution layers across %s training epochs...", epochs)
        model.fit(
            train_dataset, 
            epochs=epochs,
            verbose=1
        )

        trained_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(trained_model_path))

        mlflow.keras.log_model(model=model, artifact_path="models") #type:ignore
        logger.info("Trained architecture safely exported to local storage: %s", trained_model_path)

if __name__ == "__main__":
    main()