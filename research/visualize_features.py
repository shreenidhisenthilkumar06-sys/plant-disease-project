"""
visualize_features.py - Feature Map Visualization for PiTLiD
Paper: "PiTLiD: Identification of Plant Disease From Leaf Images Based on CNN"

Matches author's featurevisual.py — visualizes outputs of:
  Conv2d_1 to Conv2d_5  (first 5 convolutional layers)
  Mixed0, Mixed5, Mixed10 (inception modules — paper Fig. 4C)

Usage:
    python visualize_features.py --image_path ./data/test/Black_rot/image.jpg
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pylab import axis
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

IMG_SIZE    = (299, 299)
CLASS_NAMES = ['Black_rot', 'Cedar_grape_rust', 'Healthy', 'Scab']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./saved_model/pitlid_best.h5')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to a leaf image')
    parser.add_argument('--layer',      type=str, default='mixed10',
                        help='Layer to visualize (default: mixed10)')
    parser.add_argument('--save_dir',   type=str, default='./visual')
    return parser.parse_args()


def get_row_col(num_pic):
    """
    Matches author's get_row_col() in featurevisual.py.
    Calculates grid layout for feature map visualization.
    """
    squr = num_pic ** 0.6
    row  = round(squr)
    col  = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch, save_dir, layer_name):
    """
    Matches author's visualize_feature_map() in featurevisual.py.
    Shows all feature maps in a grid + their sum overlay.
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_map = img_batch
    print(f"Feature map : {feature_map.}")

    feature_map_combination = []
    num_pic = feature_map.[2]
    row, col = get_row_col(num_pic)

    plt.figure(figsize=(20, 20))
    for i in range(num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split, cmap='viridis')
        axis('off')

    save_path = os.path.join(save_dir, f'{layer_name}_feature_maps.png')
    plt.suptitle(f'Feature Maps — {layer_name}', fontsize=14)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Feature maps saved to {save_path}")
    plt.close()

    # Sum overlay (matches author's feature_map_sum)
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map_sum, cmap='viridis')
    plt.title(f'Feature Map Sum — {layer_name}')
    plt.axis('off')
    sum_path = os.path.join(save_dir, f'{layer_name}_feature_map_sum.png')
    plt.savefig(sum_path, dpi=150, bbox_inches='tight')
    print(f"Feature map sum saved to {sum_path}")
    plt.close()


def visualize_all_layers(image_path, save_dir='./visual'):
    """
    Visualize key layers matching paper Fig. 4C:
      Conv2d_1 to Conv2d_5 — first 5 convolutional layers
      Mixed0, Mixed5, Mixed10 — inception modules
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load base Inception-V3 (matches author's featurevisual.py)
    base_model = inception_v3.InceptionV3(
        weights     = 'imagenet',
        include_top = False
    )

    # Layers to visualize (paper Fig. 4C)
    target_layers = [
        'conv2d',    # Conv2d_1
        'conv2d_1',  # Conv2d_2
        'conv2d_2',  # Conv2d_3
        'conv2d_3',  # Conv2d_4
        'conv2d_4',  # Conv2d_5
        'mixed0',    # Mixed0  (256 × 29×29)
        'mixed5',    # Mixed5  (768 × 14×14)
        'mixed10',   # Mixed10 (2048 × 6×6)
    ]

    # Load and preprocess image
    img   = load_img(image_path, target_size=IMG_SIZE)
    x     = img_to_array(img)
    x     = np.expand_dims(x, axis=0)
    x     = inception_v3.preprocess_input(x)

    print(f"\nVisualizing feature maps for: {image_path}")
    print(f"Saving to: {save_dir}/\n")

    for layer_name in target_layers:
        try:
            layer  = base_model.get_layer(layer_name)
            vis_model = Model(inputs=base_model.input, outputs=layer.output)
            features  = vis_model.predict(x, verbose=0)
            print(f"  {layer_name}: ={features.}")
            feature   = features.reshape(features.shape[1:])
            visualize_feature_map(feature, save_dir, layer_name)
        except Exception as e:
            print(f"  Skipping {layer_name}: {e}")


def predict_single_image(model_path, image_path):
    """
    Predict disease class for a single leaf image and show confidence.
    """
    model = tf.keras.models.load_model(model_path)

    img   = load_img(image_path, target_size=IMG_SIZE)
    arr   = img_to_array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    pred  = np.argmax(probs)

    print(f"\n{'='*45}")
    print(f"  Prediction for: {os.path.basename(image_path)}")
    print(f"{'='*45}")
    for cls, prob in zip(CLASS_NAMES, probs):
        bar = '█' * int(prob * 30)
        print(f"  {cls:25s}: {prob*100:6.2f}%  {bar}")
    print(f"\n  → Predicted: {CLASS_NAMES[pred]} "
          f"({probs[pred]*100:.2f}% confidence)")
    print(f"{'='*45}")
    return CLASS_NAMES[pred], probs


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Visualize feature maps
    visualize_all_layers(args.image_path, args.save_dir)

    # Predict
    if os.path.exists(args.model_path):
        predict_single_image(args.model_path, args.image_path)
    else:
        print(f"Model not found at {args.model_path}. Train first.")
