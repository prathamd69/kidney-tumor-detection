from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

def create_eval_dataset(df: pd.DataFrame, images_dir: Path, batch_size: int, img_shape: tuple) -> tf.data.Dataset:
    """Assembles a streaming evaluation pipeline (NO shuffling needed for evaluation)."""
    file_paths = df['image_id'].apply(lambda x: str(images_dir / f"{x}.jpg")).astype(str).to_numpy()
    labels = df['finaltarget'].astype('int32').to_numpy()
    
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(file_paths, dtype=tf.string), 
                                                  tf.constant(labels, dtype=tf.int32)))
    
    dataset = dataset.map(lambda fp, lbl: parse_image(fp, lbl, img_shape), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
