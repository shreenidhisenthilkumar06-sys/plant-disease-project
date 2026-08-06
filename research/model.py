"""
  - InceptionV3 base (ImageNet weights, include_top=False)
  - GlobalAveragePooling2D  (GAP — replaces fully connected layers)
  - Dropout(0.5)
  - Dense(4, activation='softmax')

Phase 1: base frozen    → RMSprop(1e-3)
Phase 2: all unfrozen   → RMSprop(1e-4)
"""

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2


def build_pitlid_model(num_classes=4, phase=1):
    """
    Build PiTLiD model based on Inception-V3 + Transfer Learning.
        basic_model = InceptionV3(include_top=False, weights='imagenet')
        x = GlobalAveragePooling2D()(x)
        x = Dropout(.5)(x)
        x = Activation('relu')(x)
        x = Dense(num_classes, activation='softmax')(x)

    Args:
        num_classes (int): Number of disease classes (4 for corn)
        phase (int):
            1 = base frozen 
            2 = all unfrozen 

    Returns:
        model: Keras Model (not yet compiled)
    """

    # Base model — ImageNet pretrained, no top FC layers
    base_model = InceptionV3(
        weights     = 'imagenet',
        include_top = False,          # Remove 1000-class FC head
        input_shape = (299, 299, 3)   # Inception-V3 standard input
    )

    # Custom classification head 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)   # GAP: MxNxC → 1xC 
    x = Dropout(0.5)(x)               # Dropout prevents overfitting
    x = Activation('relu')(x)         # add relu before final dense
    predictions = Dense(
        num_classes,
        activation='softmax'          # Softmax for multi-class
    )(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if phase == 1:
        # for layer in base_model.layers: layer.trainable = False
        for layer in base_model.layers:
            layer.trainable = False
        print("Phase 1: Base FROZEN — training classification head only")

    elif phase == 2:
        # for layer in base_model.layers: layer.trainable = True
        for layer in base_model.layers:
            layer.trainable = True
        print("Phase 2: All layers UNFROZEN — full fine-tuning (None_frozen)")

    trainable   = sum(1 for l in model.layers if l.trainable)
    total       = len(model.layers)
    print(f"Trainable layers: {trainable} / {total}")

    return model


def compile_model(model, lr=1e-3):
    """
    Compile model with RMSprop 
      Phase 1: RMSprop(1e-3)
      Phase 2: RMSprop(lr=0.0001)
    """
    model.compile(
        optimizer = RMSprop(learning_rate=lr),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )
    return model


if __name__ == '__main__':
    # Quick test
    model = build_pitlid_model(num_classes=4, phase=1)
    model = compile_model(model, lr=1e-3)
    model.summary()
