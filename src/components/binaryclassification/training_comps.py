from pathlib import Path
from box import ConfigBox
import tensorflow as tf
from typing import Tuple
import numpy as np
from src.utils import build_base_model, create_training_dataset, create_testing_dataset, loadFile


def _get_model_binary(params: ConfigBox) -> tf.keras.Model:
    return build_base_model(
        img_shape= (params.basic_model_params.img_height, params.basic_model_params.img_width),
        lr=float(params.binaryclass_model_params.learning_rate),
        loss=str(params.binaryclass_model_params.loss),
        metrics=list(params.binaryclass_model_params.metrics),
        optimizer=str(params.binaryclass_model_params.optimizer),
        is_binaryClassification=bool(params.binaryclass_model_params.is_binaryClassification),
        num_classes=int(params.binaryclass_model_params.num_classes)
    )

def training(config: ConfigBox, params: ConfigBox) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.keras.Model]:
    
    img_shape = (params.basic_model_params.img_height, params.basic_model_params.img_width)
    
    train_df = loadFile(Path(config.data_paths.train_data_path))
    
    train_dataset, val_dataset = create_training_dataset(
        train_df, 
        Path(config.data_paths.images_dir), 
        params.binaryclass_model_params.batch_size, 
        img_shape, 
        params.binaryclass_model_params.is_binaryClassification
    )

    model = _get_model_binary(params)

    return train_dataset, val_dataset, model

def testing(config: ConfigBox, params: ConfigBox) -> Tuple[tf.data.Dataset, np.ndarray, tf.keras.Model]:

    test_df = loadFile(Path(config.data_paths.test_data_path))
    img_shape = (params.basic_model_params.img_height, params.basic_model_params.img_width)
    
    test_dataset, labels = create_testing_dataset(
        test_df, 
        Path(config.data_paths.images_dir), 
        params.binaryclass_model_params.batch_size, 
        img_shape, 
        params.binaryclass_model_params.is_binaryClassification
    )

    model = _get_model_binary(params)

    return test_dataset, labels, model