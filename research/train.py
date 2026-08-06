"""
  Phase 1: steps=100, epochs=10  (head only, base frozen)
  Phase 2: steps=200, epochs=50  (all layers unfrozen, initial_epoch=10)
"""

import os
import sys
 
# ── Fix segmentation fault on Mac (must be before any TF import) ─────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']       = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
 
import numpy as np
 
import tensorflow as tf
 
# Prevent TF from grabbing all memory at once (fixes seg fault on Mac)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU found: {gpus}")
else:
    print("No GPU found — using CPU")
 
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents display crashes on Mac
import matplotlib.pyplot as plt


from model import build_pitlid_model, compile_model
from data_loader import load_data_from_directory, BATCH_TRAIN, BATCH_VAL
from cyclical_lr import get_callbacks


TRAIN_DIR = './data_corn/train'
VAL_DIR   = './data_corn/val'
SAVE_PATH = './saved_model/pitlid_corn_best.h5'
NUM_CLASSES    = 4

PHASE1_STEPS   = 100    #  steps_per_epoch=100
PHASE1_EPOCHS  = 10     #  epochs=10

PHASE2_STEPS   = 200    #  steps_per_epoch=200
PHASE2_EPOCHS  = 50     #  epochs=50, initial_epoch=10 → runs 40 more epochs


def train():
    os.makedirs('./saved_model', exist_ok=True)

    print("\n" + "="*55)
    print("  PiTLiD Training ")
    print("="*55)
    print(f"  Phase 1 : steps={PHASE1_STEPS}, epochs={PHASE1_EPOCHS} (head only)")
    print(f"  Phase 2 : steps={PHASE2_STEPS}, epochs={PHASE2_EPOCHS} (full fine-tune)")
    print("="*55)

    # ── Load data 
    print("\n[1/4] Loading data...")
    train_gen = load_data_from_directory(
        TRAIN_DIR, batch_size=BATCH_TRAIN, augment=True, shuffle=True)
    val_gen = load_data_from_directory(
        VAL_DIR, batch_size=BATCH_VAL, augment=False, shuffle=False)

    # ── PHASE 1: Freeze base, train head only ─────────────────────────────────
    #   for layer in base_model.layers: layer.trainable = False
    #   model.compile(optimizer=RMSprop(1e-3), ...)
    #   model.fit_generator(steps_per_epoch=100, epochs=10)
    print(f"\n[2/4] Phase 1: Head-only training ({PHASE1_EPOCHS} epochs, "
          f"steps={PHASE1_STEPS}, LR=1e-3)...")

    model = build_pitlid_model(num_classes=NUM_CLASSES, phase=1)
    model = compile_model(model, lr=1e-3)

    callbacks_p1, _ = get_callbacks(
        clr_base_lr         = 5e-4,
        clr_max_lr          = 1e-3,
        step_size           = PHASE1_STEPS,
        early_stop_patience = 5,
        model_save_path     = './saved_model/phase1_best.h5'
    )

    history1 = model.fit(
        train_gen,
        steps_per_epoch  = PHASE1_STEPS,
        epochs           = PHASE1_EPOCHS,
        validation_data  = val_gen,
        validation_steps = len(val_gen),
        callbacks        = callbacks_p1,
        verbose          = 1
    )

    # Load best phase 1 weights
    if os.path.exists('./saved_model/phase1_best.h5'):
        model.load_weights('./saved_model/phase1_best.h5')
        print("  Loaded best Phase 1 weights.")

    best_p1_acc = max(history1.history['val_accuracy'])
    print(f"\n  Phase 1 best val_accuracy: {best_p1_acc*100:.2f}%")

    # ── PHASE 2: Unfreeze all, full fine-tune ─────────────────────────────────
    #   for layer in base_model.layers: layer.trainable = True
    #   model.compile(optimizer=RMSprop(lr=0.0001), ...)
    #   model.fit_generator(steps_per_epoch=200, epochs=50, initial_epoch=10)
    print(f"\n[3/4] Phase 2: Full fine-tuning ({PHASE2_EPOCHS} epochs, "
          f"steps={PHASE2_STEPS}, LR=1e-4)...")

    # Unfreeze ALL layers 
    for layer in model.layers:
        layer.trainable = True

    # uses RMSprop(lr=0.0001) for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_p2, clr_cb = get_callbacks(
        clr_base_lr         = 1e-5,
        clr_max_lr          = 1e-4,   # CLR within fine-tune range
        step_size           = PHASE2_STEPS,
        early_stop_patience = 14,     
        model_save_path     = SAVE_PATH
    )

    # initial_epoch=PHASE1_EPOCHS 
    history2 = model.fit(
        train_gen,
        steps_per_epoch  = PHASE2_STEPS,
        epochs           = PHASE2_EPOCHS,
        initial_epoch    = PHASE1_EPOCHS,   # continue epoch count from 10
        validation_data  = val_gen,
        validation_steps = len(val_gen),
        callbacks        = callbacks_p2,
        verbose          = 1
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving plots...")
    plot_combined_history(history1, history2)
    clr_cb.plot_lr_schedule('clr_schedule.png')

    best_p2_acc = max(history2.history['val_accuracy'])
    print(f"\n{'='*55}")
    print(f"  Phase 1 best val_accuracy : {best_p1_acc*100:.2f}%")
    print(f"  Phase 2 best val_accuracy : {best_p2_acc*100:.2f}%")
    print(f"  Best model saved to       : {SAVE_PATH}")
    print(f"{'='*55}")

    return model


def plot_combined_history(h1, h2, save_path='training_history.png'):
    """
    Plot accuracy and loss across both phases.
    Red dashed line marks where Phase 2 begins (epoch 10).
    Green dotted line marks paper's target accuracy (99.45%).
    """
    acc1  = h1.history['val_accuracy']
    acc2  = h2.history['val_accuracy']
    loss1 = h1.history['val_loss']
    loss2 = h2.history['val_loss']

    all_acc  = acc1 + acc2
    all_loss = loss1 + loss2
    epochs   = range(1, len(all_acc) + 1)
    split    = len(acc1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PiTLiD Training History', fontsize=13)

    # Accuracy plot
    axes[0].plot(epochs, all_acc, 'b-', linewidth=1.5, label='Val Accuracy')
    axes[0].axvline(x=split, color='red', linestyle='--',
                    alpha=0.7, label=f'Phase 2 starts (epoch {split})')
    axes[0].axhline(y=0.9945, color='green', linestyle=':',
                    alpha=0.8, label='Paper target: 99.45%')
    axes[0].set_title('Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(epochs, all_loss, color='orange', linewidth=1.5, label='Val Loss')
    axes[1].axvline(x=split, color='red', linestyle='--',
                    alpha=0.7, label=f'Phase 2 starts (epoch {split})')
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Training history saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    train()
