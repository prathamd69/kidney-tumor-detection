import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np
from src.utils import configLogger, loadYaml

logger = configLogger("Build_Dataset", "Build_Dataset.log")

def parse_image(file_path: str, label: int, img_shape: tuple) -> tuple:
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_shape)
        return image, label
    
    except Exception as e:
        logger.error("Failed to parse image file at %s: %s", file_path, e)
        raise

def create_training_dataset(df: pd.DataFrame, images_dir: Path,
                            batch_size: int, img_shape: tuple, 
                            is_binaryClassfication: bool) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Assembles highly optimized, leak-proof training and validation streaming pipelines."""

    try:
        file_paths = df['image_id'].apply(lambda x: str(images_dir / f"{x}.jpg")).astype(str).to_numpy()

        if is_binaryClassfication:
            labels = df['binarytarget'].astype('int32').to_numpy()
        else:
            labels = df['target'].astype('int32').to_numpy()

        base_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(file_paths, dtype=tf.string), 
                                                           tf.constant(labels, dtype=tf.int32)))

        total_samples = len(df)
        val_size = int(total_samples * 0.2)
        
        val_dataset = base_dataset.take(val_size)
        train_dataset = base_dataset.skip(val_size)

        train_dataset = train_dataset.shuffle(buffer_size=total_samples, reshuffle_each_iteration=True)

        train_dataset = train_dataset.map(lambda fp, lbl: parse_image(fp, lbl, img_shape), 
                                          num_parallel_calls=tf.data.AUTOTUNE)
        
        val_dataset = val_dataset.map(lambda fp, lbl: parse_image(fp, lbl, img_shape), 
                                        num_parallel_calls=tf.data.AUTOTUNE)

        train_pipeline = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_pipeline = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.info("Train and Validation datasets created successfully: is_binary = %s", is_binaryClassfication)

        return train_pipeline, val_pipeline
    
    except Exception as e:
        logger.exception("Error while bulding train and validation dataset : %s", e)
        raise
    
def create_testing_dataset(df: pd.DataFrame, images_dir: Path,
                           batch_size: int, img_shape: tuple, 
                           is_binaryClassfication: bool) -> tuple[tf.data.Dataset, np.ndarray]:
    """Assembles a highly optimized, sequential testing data pipeline.
    
    No shuffling or splitting is applied to guarantee stable evaluation order.
    """

    try:
        file_paths = df['image_id'].apply(lambda x: str(images_dir / f"{x}.jpg")).astype(str).to_numpy()

        if is_binaryClassfication:
            labels = df['binarytarget'].astype('int32').to_numpy()
        else:
            labels = df['target'].astype('int32').to_numpy()

        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(file_paths, dtype=tf.string), 
                                                      tf.constant(labels, dtype=tf.int32)))

        dataset = dataset.map(lambda fp, lbl: parse_image(fp, lbl, img_shape), 
                              num_parallel_calls=tf.data.AUTOTUNE)
        
        test_pipeline = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.info("Testing dataset created successfully : is_binary = %s", is_binaryClassfication)
        logger.info("Testing labels generated successfully")
        return test_pipeline, labels
    
    except Exception as e:
        logger.exception("Error while making testing dataset and labels : %s", e)
        raise