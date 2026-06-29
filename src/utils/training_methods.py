import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np
from tensorflow.keras import layers #type:ignore
from tensorflow.keras import applications # type:ignore
from tensorflow.keras.layers import(Dense, Dropout, BatchNormalization) # type:ignore
from tensorflow.keras.models import Model # type:ignore
from src.utils import configLogger, loadYaml

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

        logger.info("Train and Val datasets created successfully: is_binary = %s", is_binaryClassfication)

        return train_pipeline, val_pipeline
    
    except Exception as e:
        logger.exception("Error while making dataset : %s", e)
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
        return test_pipeline, labels
    
    except Exception as e:
        logger.exception("Error while making testing dataset : %s", e)
        raise

def build_base_model(
    img_shape: tuple, 
    lr: float, 
    loss: str, 
    metrics: list, 
    optimizer: str,
    is_binaryClassification : bool,
    num_classes: int) -> Model:

    params = loadYaml(Path('params.yaml'))

    base_model = str(params.model_architecture.base_model)
    include_top = bool(params.model_architecture.include_top)
    pooling = str(params.model_architecture.pooling)
    dense_units = int(params.model_architecture.dense_units)
    dropout_rate = float(params.model_architecture.dropout_rate)
    activation = str(params.model_architecture.activation)

    base_model_cls = getattr(applications, base_model)
    base_model = base_model_cls(
        weights='imagenet', 
        include_top=include_top, 
        input_shape=(*img_shape, 3), 
        pooling=pooling
    )

    base_model.trainable = False 
    inputs = layers.Input(shape=(*img_shape, 3))
    x = base_model(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    if is_binaryClassification:
        output = Dense(1, activation='sigmoid')(x)

    else:
        output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)

    OPTIMIZER_MAP= {
        "Adam": tf.keras.optimizers.Adam,
        "SGD": tf.keras.optimizers.SGD,
        "RMSprop": tf.keras.optimizers.RMSprop,
        "Adamax" : tf.keras.optimizers.Adamax
    }

    optimizer_cls = OPTIMIZER_MAP.get(optimizer, tf.keras.optimizers.Adam)

    model.compile(
        optimizer=optimizer_cls(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )
    
    return model