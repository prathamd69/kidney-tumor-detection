from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2 # type:ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type:ignore
from tensorflow.keras.models import Model # type:ignore
from src.utils import configLogger, loadYaml

logger = configLogger("model_building", "model_building.log")

def build_binary_model(
    img_shape: tuple, 
    lr: float, 
    loss: str, 
    metrics: list, 
    optimizer_cls: type[tf.keras.optimizers.Optimizer]) -> Model:
    
    base_model = ResNet50V2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_shape[0], img_shape[1], 3)
    )
    base_model.trainable = False  
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    model.compile(
        optimizer=optimizer_cls(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_multi_model(
    img_shape: tuple, 
    lr: float, 
    loss: str, 
    metrics: list, 
    optimizer_cls: type[tf.keras.optimizers.Optimizer],
    num_classes: int = 4) -> Model:

    base_model = ResNet50V2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_shape[0], img_shape[1], 3)
    )
    base_model.trainable = False  
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    model.compile(
        optimizer=optimizer_cls(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )
    
    return model

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    img_shape = (int(params.model_building.img_height), int(params.model_building.img_width))

    binary_lr = float(params.binaryclass_model_training.learning_rate)
    multi_lr = float(params.multiclass_model_training.learning_rate)

    binaryloss = str(params.model_building.binary_loss)
    multiloss = str(params.model_building.multi_loss)

    metrics = list(params.model_building.metrics)

    OPTIMIZER_MAP= {
        "Adam": tf.keras.optimizers.Adam,
        "SGD": tf.keras.optimizers.SGD,
        "RMSprop": tf.keras.optimizers.RMSprop
    }

    opt_name = params.model_building.optimizer
    optimizer_class = OPTIMIZER_MAP.get(opt_name, tf.keras.optimizers.Adam)

    base_binaryclass_model_path = Path(config.model_paths.base_binaryclass_model_path)
    base_multiclass_model_path = Path(config.model_paths.base_multiclass_model_path)
    
    logger.info("Building compiled ResNet50V2 baseline model architecture")
    binarymodel = build_binary_model(img_shape, binary_lr, binaryloss, metrics, optimizer_class)
    multimodel = build_multi_model(img_shape, multi_lr, multiloss, metrics, optimizer_class)
    
    base_binaryclass_model_path.parent.mkdir(parents=True, exist_ok=True)
    binarymodel.save(str(base_binaryclass_model_path))
    multimodel.save(str(base_multiclass_model_path))
    logger.info("Base binary untargeted model saved at: %s", base_binaryclass_model_path)
    logger.info("Base multiclass untargeted model saved at: %s", base_multiclass_model_path)

if __name__ == "__main__":
    main()