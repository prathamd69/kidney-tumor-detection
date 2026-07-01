from pathlib import Path
from box import ConfigBox
import tensorflow as tf
from typing import Tuple
import numpy as np
from src.components import (build_base_model, 
                       create_training_dataset, 
                       create_testing_dataset)
from src.utils import (loadFile,
                       configLogger)

logger = configLogger('ModelFactory', 'ModelFactory.log')

class ModelPipelineFactory:
    def __init__(self, config: ConfigBox, params: ConfigBox, is_binaryClassification: bool):
        
        self.is_binaryClassification = is_binaryClassification

        # setting up common attributes
        self._artifactspaths = ConfigBox(config.artifacts_paths)
        self._basicparams = ConfigBox(params.basic_model_params)
        self._datapaths = ConfigBox(config.data_paths)

        self.img_shape= (int(self._basicparams.img_height), int(self._basicparams.img_width))

        # setting up data paths
        self.trainpath = Path(self._datapaths.train_data_path)
        self.testpath = Path(self._datapaths.test_data_path)
        self.imagesdir = Path(self._datapaths.images_dir)

        # setting up model attributes and paths
        if self.is_binaryClassification:
            self._modelparams = ConfigBox(params.binaryclass_model_params)
              
        else:
            self._modelparams = ConfigBox(params.multiclass_model_params)

        # model attributes
        self.lr=float(self._modelparams.learning_rate)
        self.loss=str(self._modelparams.loss)
        self.metrics=list(self._modelparams.metrics)
        self.optimizer=str(self._modelparams.optimizer)
        self.num_classes=int(self._modelparams.num_classes)
        self.batch_size=int(self._modelparams.batch_size)
        self.epochs=int(self._modelparams.epochs)

        logger.info("ModelPipelineFactory object created (is_binary = %s)", is_binaryClassification)

    # function to setup the basic model architecture for training/testing/prediction
    def model_config(self) -> tf.keras.Model:

        return build_base_model(
            img_shape= self.img_shape,
            lr=self.lr,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            is_binaryClassification=self.is_binaryClassification,
            num_classes=self.num_classes
        )

    def training_components(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.keras.Model]:
        
        logger.info("Loading training dataframe from %s", self.trainpath)
        train_df = loadFile(self.trainpath)
        
        train_dataset, val_dataset = create_training_dataset(
                                        train_df, 
                                        self.imagesdir, 
                                        self.batch_size, 
                                        self.img_shape, 
                                        self.is_binaryClassification)

        model = self.model_config()

        logger.info("Training and validation data streaming pipelines initialized.")
        return train_dataset, val_dataset, model

    def testing_components(self) -> Tuple[tf.data.Dataset, np.ndarray, tf.keras.Model]:

        logger.info("Loading testing dataframe from %s", self.testpath)
        test_df = loadFile(self.testpath)
        
        test_dataset, labels = create_testing_dataset(
            test_df, 
            self.imagesdir, 
            self.batch_size, 
            self.img_shape, 
            self.is_binaryClassification
        )

        model = self.model_config()

        logger.info("Testing dataset pipeline created. Extracted %d true ground labels.", len(labels))
        return test_dataset, labels, model
    
    @property
    def weights_path(self) -> Path:
        if self.is_binaryClassification:
            return Path(self._artifactspaths.binaryweights)
        return Path(self._artifactspaths.multiweights)

    @property
    def experiment_meta(self) -> tuple[str, str]:
        return str(self._modelparams.experiment_name), str(self._modelparams.run_name)
    
    @property
    def metrics_path(self) -> Path:
        if self.is_binaryClassification:
            return Path(self._artifactspaths.binarymetrics)
        return Path(self._artifactspaths.multimetrics)

    @property
    def getparams(self) -> ConfigBox:
        params = {
            'learning_rate' : self.lr,
            'loss' : self.loss,
            'metrics' : self.metrics,
            'optimizer' : self.optimizer,
            'num_classes' : self.num_classes,
            'batch_size' : self.batch_size,
            'epochs' : self.epochs
            }
        params = ConfigBox(params)
        return params