import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers #type:ignore
from tensorflow.keras import applications # type:ignore
from tensorflow.keras.layers import(Dense, Dropout, BatchNormalization) # type:ignore
from tensorflow.keras.models import Model # type:ignore
from src.utils import configLogger, loadYaml

logger = configLogger("training_methods", "training_methods.log")


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