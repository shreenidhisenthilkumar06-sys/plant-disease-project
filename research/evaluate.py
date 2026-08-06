

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from data_loader import load_data_from_directory, BATCH_VAL, CLASS_NAMES


TEST_DIR    = './data_corn/test'
MODEL_PATH  = './saved_model/pitlid_corn_best.h5'
N_RUNS      = 10


# ── Metrics  ───────────────────────────────────
def compute_metrics(y_true, y_pred):
    """
    Four metrics used in paper:
      SEN = TP / (TP + FN)          — Sensitivity
      PRC = TP / (TP + FP)          — Precision
      F1  = 2*TP / (2*TP + FP + FN) — F1 Score
      ACC = (TP + TN) / Total        — Accuracy
    """
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, average='macro', zero_division=0)
    prc = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {'ACC': acc, 'SEN': sen, 'PRC': prc, 'F1': f1}


# ── Single run ────────────────────────────────────────────────────────────────
def evaluate_once(model, test_dir):
    test_gen = load_data_from_directory(
        test_dir, batch_size=BATCH_VAL, augment=False, shuffle=False)
    test_gen.reset()
    y_pred_prob = model.predict(test_gen, steps=len(test_gen), verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_gen.classes[:len(y_pred)]
    return y_true, y_pred


# ── 10-run stability test ─────────────────────────────────────
def evaluate_n_runs(model, test_dir, n_runs=10):
    
    results = {'ACC': [], 'SEN': [], 'PRC': [], 'F1': []}

    print(f"\n{'='*50}")
    print(f"  {n_runs}-Run Stability Evaluation")
    print(f"{'='*50}")

    for run in range(1, n_runs + 1):
        y_true, y_pred = evaluate_once(model, test_dir)
        m = compute_metrics(y_true, y_pred)
        for k in results:
            results[k].append(m[k])
        print(f"  Run {run:2d}: ACC={m['ACC']*100:.2f}%  "
              f"SEN={m['SEN']*100:.2f}%  "
              f"PRC={m['PRC']*100:.2f}%  "
              f"F1={m['F1']*100:.2f}%")

    print(f"\n{'='*50}")
    print("  Summary (mean):")
    for k in results:
        arr = np.array(results[k]) * 100
        print(f"    {k}: {arr.mean():.2f}%")
    print(f"{'='*50}")
    plot_n_runs(results, n_runs)
    return results


# ── Confusion matrix  ───────────────────────────────
def plot_confusion_matrix(y_true, y_pred,
                          class_names=CLASS_NAMES,
                          title='Confusion Matrix (CORN Disease)',
                          save_path='confusion_matrix.png'):
    """
    Normalized confusion matrix matching author's matrix.py style.
    Ideal: 1.00 on diagonal, 0.00 everywhere else.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16, pad=20, fontweight='bold')

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=15)
    plt.yticks(tick_marks, class_names)

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()

    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm_norm[i, j]),
                 horizontalalignment='center',
                 color='white' if cm_norm[i, j] > thresh else 'black',
                 fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('True label', labelpad=10, fontsize=12, fontstyle='oblique')
    plt.xlabel('Predicted label', labelpad=15, fontsize=12, fontstyle='oblique')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


# ── 10-run bar chart ────────────────────────────────────
def plot_n_runs(results, n_runs=10, save_path='ten_runs_evaluation.png'):
    runs    = np.arange(1, n_runs + 1)
    metrics = ['ACC', 'SEN', 'PRC', 'F1']
    colors  = ['steelblue', 'tomato', 'mediumseagreen', 'mediumpurple']
    width   = 0.2

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_runs)

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = np.array(results[metric]) * 100
        ax.bar(x + i * width, vals, width, label=metric,
               color=color, alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Run')
    ax.set_ylabel('Score (%)')
    ax.set_title('10-Run Evaluation: ACC, SEN, PRC, F1 (Paper Fig. 5C)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([str(i) for i in range(1, n_runs + 1)])
    ax.set_ylim(85, 101)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"10-run chart saved to {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\nLoading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Single run + confusion matrix
    print("\n--- Single Run ---")
    y_true, y_pred = evaluate_once(model, TEST_DIR)
    m = compute_metrics(y_true, y_pred)
    print(f"  ACC: {m['ACC']*100:.2f}%")
    print(f"  SEN: {m['SEN']*100:.2f}%")
    print(f"  PRC: {m['PRC']*100:.2f}%")
    print(f"  F1 : {m['F1']*100:.2f}%")

    plot_confusion_matrix(y_true, y_pred)

    # 10-run stability
    evaluate_n_runs(model, TEST_DIR, n_runs=N_RUNS)
