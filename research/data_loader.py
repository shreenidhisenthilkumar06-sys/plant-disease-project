
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE    = (299, 299)   # Inception-V3 required input
BATCH_TRAIN = 32           # batch_size=32
BATCH_VAL   = 16           # batch_size=16
NUM_CLASSES = 4

CLASS_NAMES = CLASS_NAMES = ['Gray_leaf_spot','Common_rust', 'Healthy','Leaf_blight']


def get_train_datagen():
    """
        rescale=1./255
        rotation_range=90       → random rotate [0,90] degrees
        width_shift_range=0.3   → random horizontal shift
        height_shift_range=0.3  → random vertical shift
        shear_range=0.3         → random shear
        zoom_range=0.3          → random zoom
        vertical_flip=True      → random vertical flip
        horizontal_flip=True    → random horizontal flip
        fill_mode='nearest'
    """
    return ImageDataGenerator(
        rescale            = 1.0 / 255,
        rotation_range     = 90,
        width_shift_range  = 0.3,
        height_shift_range = 0.3,
        shear_range        = 0.3,
        zoom_range         = 0.3,
        vertical_flip      = True,
        horizontal_flip    = True,
        fill_mode          = 'nearest'
    )


def get_val_datagen():

    return ImageDataGenerator(rescale=1.0 / 255)


def load_data_from_directory(data_dir, batch_size=BATCH_TRAIN,
                              augment=True, shuffle=True):
    
    datagen = get_train_datagen() if augment else get_val_datagen()

    generator = datagen.flow_from_directory(
        data_dir,
        target_size = IMG_SIZE,       # 299x299 for Inception-V3
        batch_size  = batch_size,
        class_mode  = 'categorical',
        classes     = CLASS_NAMES,
        shuffle     = shuffle
    )

    print(f"Loaded {generator.samples} images from '{data_dir}'")
    print(f"  Classes: {generator.class_indices}")
    return generator


def visualize_augmentation(data_dir, save_path='augmentation_examples.png'):
    """
    Visualize the 6 augmentation techniques applied to a sample image.
    Reproduces Fig. 4B in the paper.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # Find first image
    first_class = CLASS_NAMES[0]
    class_dir   = os.path.join(data_dir, first_class)
    if not os.path.exists(class_dir):
        print(f"Could not find {class_dir} for visualization.")
        return

    img_file = os.listdir(class_dir)[0]
    img_path = os.path.join(class_dir, img_file)

    img     = load_img(img_path, target_size=IMG_SIZE)
    img_arr = img_to_array(img).reshape((1,) + img_to_array(img).shape)

    augmentations = {
        'Original' : ImageDataGenerator(rescale=1./255),
        'Flip'     : ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True),
        'Rotation' : ImageDataGenerator(rescale=1./255, rotation_range=90),
        'Shift'    : ImageDataGenerator(rescale=1./255, width_shift_range=0.3, height_shift_range=0.3),
        'Shear'    : ImageDataGenerator(rescale=1./255, shear_range=0.3),
        'Zoom'     : ImageDataGenerator(rescale=1./255, zoom_range=0.3),
    }

    fig, axes = plt.subplots(1, len(augmentations), figsize=(18, 3))
    fig.suptitle(f'Data Augmentation — {first_class}', fontsize=13)

    for ax, (name, gen) in zip(axes, augmentations.items()):
        batch = next(gen.flow(img_arr, batch_size=1))
        ax.imshow(batch[0])
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Augmentation visualization saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    DATA_DIR = './data_corn/train'
    if os.path.exists(DATA_DIR):
        gen = load_data_from_directory(DATA_DIR)
        visualize_augmentation(DATA_DIR)
    else:
        print("Set DATA_DIR to your dataset path to test.")
