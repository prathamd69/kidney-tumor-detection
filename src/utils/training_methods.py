import tensorflow as tf
import pandas as pd
from pathlib import Path
from src.utils import configLogger

logger = configLogger("training_methods", "training_methods.log")

def parse_image(file_path: str, label: int, img_shape: tuple) -> tuple:
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_shape)
        return image, label
    
    except Exception as e:
        logger.error("Failed to parse image file at %s: %s", file_path, e)
        raise

def create_dataset(df: pd.DataFrame, images_dir: Path, batch_size: int, img_shape: tuple, is_binaryClassfication: bool) -> tf.data.Dataset:
    """Assembles a highly optimized, prefetched streaming data pipeline."""

    try:
        file_paths = df['image_id'].apply(lambda x: str(images_dir / f"{x}.jpg")).astype(str).to_numpy()

        if is_binaryClassfication:
            labels = df['binarytarget'].astype('int32').to_numpy()
        
        else:
            labels = df['target'].astype('int32').to_numpy()

        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(file_paths, dtype=tf.string), 
                                                    tf.constant(labels, dtype=tf.int32)))

        dataset = dataset.shuffle(buffer_size=len(df))

        dataset = dataset.map(lambda fp, lbl: parse_image(fp, lbl, img_shape), 
                            num_parallel_calls=tf.data.AUTOTUNE)
        
        logger.info("Dataset created successfully : is_binary = %s", is_binaryClassfication)
        return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    except Exception as e:
        logger.exception("Error while making dataset : %s", e)
        raise
    