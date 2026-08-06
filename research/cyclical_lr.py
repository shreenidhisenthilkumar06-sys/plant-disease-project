"""
cyclical_lr.py - Cyclical Learning Rate Callback for PiTLiD
three policies:
  1. triangular  — linear variation
  2. triangular2 — max_lr halves each cycle (BEST, used here)
  3. exp_range   — exponential decay of max_lr

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class CyclicalLearningRate(Callback):
    """
    Cyclical Learning Rate — oscillates between base_lr and max_lr
    to escape saddle points and converge faster

    Args:
        base_lr  : Lower bound LR (0.001)
        max_lr   : Upper bound LR (0.006)
        step_size: Half-cycle length in iterations
        policy   : 'triangular', 'triangular2', or 'exp_range'
        gamma    : Decay for exp_range (default: 0.99994)
    """

    def __init__(self, base_lr=0.001, max_lr=0.006,
                 step_size=200, policy='triangular2', gamma=0.99994):
        super().__init__()
        self.base_lr        = base_lr
        self.max_lr         = max_lr
        self.step_size      = step_size
        self.policy         = policy
        self.gamma          = gamma
        self.clr_iterations = 0
        self.history        = {}

    def _compute_lr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x     = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)

        if self.policy == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)

        elif self.policy == 'triangular2':
            # Paper Fig 3G: max_lr halves each full cycle
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) / (2 ** (cycle - 1))

        elif self.policy == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * (self.gamma ** self.clr_iterations)

        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        return lr

    def _set_lr(self, lr):
        """Compatible with both old and new Keras versions."""
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate.assign(lr)
        elif hasattr(optimizer, 'lr'):
            tf.keras.backend.set_value(optimizer.lr, lr)
        else:
            optimizer._set_hyper('learning_rate', lr)

    def on_train_begin(self, logs=None):
        self._set_lr(self.base_lr)

    def on_batch_end(self, batch, logs=None):
        self.clr_iterations += 1
        lr = self._compute_lr()
        self._set_lr(lr)
        self.history.setdefault('lr', []).append(lr)
        self.history.setdefault('iterations', []).append(self.clr_iterations)

    def plot_lr_schedule(self, save_path='clr_schedule.png'):
        if not self.history.get('lr'):
            print("No LR history. Train first.")
            return
        plt.figure(figsize=(10, 4))
        plt.plot(self.history['iterations'], self.history['lr'], 'b-', linewidth=1.5)
        plt.axhline(y=self.max_lr,  color='r', linestyle='--', label=f'Max LR = {self.max_lr}')
        plt.axhline(y=self.base_lr, color='g', linestyle='--', label=f'Base LR = {self.base_lr}')
        plt.xlabel('Training Iterations')
        plt.ylabel('Learning Rate')
        plt.title(f"CLR — '{self.policy}' policy ")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CLR schedule saved to {save_path}")
        plt.close()


def get_callbacks(clr_base_lr=0.001, clr_max_lr=0.006, step_size=200,
                  early_stop_patience=14, model_save_path='best_model.h5'):
    """
    Returns all training callbacks 
      1. CyclicalLearningRate (triangular2)
      2. EarlyStopping        (patience=14 )
      3. ModelCheckpoint      (save best val_accuracy)
      4. CSVLogger            (training log)
    """
    clr = CyclicalLearningRate(
        base_lr   = clr_base_lr,
        max_lr    = clr_max_lr,
        step_size = step_size,
        policy    = 'triangular2'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',
        patience             = early_stop_patience,
        restore_best_weights = True,
        verbose              = 1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath          = model_save_path,
        monitor           = 'val_accuracy',
        save_best_only    = True,
        save_weights_only = False,
        verbose           = 1
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        'training_log.csv',
        append = True    # append so both phases are in one file
    )

    return [clr, early_stop, checkpoint, csv_logger], clr


if __name__ == '__main__':
    # Preview CLR for 8000 iterations
    clr = CyclicalLearningRate(base_lr=0.001, max_lr=0.006,
                                step_size=200, policy='triangular2')
    for i in range(8000):
        clr.clr_iterations = i
        clr.history.setdefault('lr', []).append(clr._compute_lr())
        clr.history.setdefault('iterations', []).append(i)
    clr.plot_lr_schedule('clr_preview.png')
    print("CLR preview saved.")
