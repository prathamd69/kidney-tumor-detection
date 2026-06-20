from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2 # type:ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type:ignore
from tensorflow.keras.models import Model # type:ignore
from src.utils import configLogger, loadYaml

logger = configLogger("model_building", "model_building.log")

def build_model(img_shape: tuple) -> Model:
    base_model = ResNet50V2(weights='imagenet', 
                            include_top=False, 
                            input_shape=(img_shape[0], img_shape[1], 3))
    base_model.trainable = False  
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=base_model.input, outputs=output_layer)

def main():
    config = loadYaml(Path("config/config.yaml"))
    params = loadYaml(Path("params.yaml"))

    img_shape = (int(params.model_training.img_height), int(params.model_training.img_width))
    lr = float(params.model_training.learning_rate)
    base_model_path = Path(config.model_building.base_model_path)
    
    logger.info("Building compiled ResNet50V2 baseline model architecture")
    model = build_model(img_shape)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    base_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(base_model_path))
    logger.info("Base untargeted model saved at: %s", base_model_path)

if __name__ == "__main__":
    main()