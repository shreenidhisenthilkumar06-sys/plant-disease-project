
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import shutil

from model import build_pitlid_model, compile_model
from data_loader import get_train_datagen, get_val_datagen, IMG_SIZE, BATCH_TRAIN, BATCH_VAL
from cyclical_lr import get_callbacks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ── Settings ──────────────────────────────────────────────────────────────────
RESULTS_DIR  = './novelty_results/novelty1'
TEMP_DIR     = './temp_optimal'

# Image counts to test — includes below AND above paper's 30
IMAGE_COUNTS = [5, 10, 15, 20, 25, 30, 40, 50]

# Reduced training for speed — same architecture as paper
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20
PHASE1_STEPS  = 50
PHASE2_STEPS  = 100

# Species configuration
SPECIES_CONFIG = {
    'Peach': {
        'train_dir' : './data_peach/train',
        'val_dir'   : './data_peach/val',
        'test_dir'  : './data_peach/test',
        'classes'   : ['Bacterial_spot', 'Healthy'],
        'accuracy_obt' : 89.0,
        'color'     : '#fb8c00'
    },
    'Corn': {
        'train_dir' : './data_corn/train',
        'val_dir'   : './data_corn/val',
        'test_dir'  : './data_corn/test',
        'classes'   : ['Gray_leaf_spot', 'Common_rust', 'Healthy', 'Leaf_blight'],
        'accuracy_obt' : 95.0,
        'color'     : '#8e24aa'
    }

}


# ── Dataset Utilities ─────────────────────────────────────────────────────────

def create_subset(n_images, source_train_dir, output_dir, class_names):
    """
    Create temporary training folder with exactly n_images per class.
    Randomly selects n_images from full training set.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for cls in class_names:
        src = os.path.join(source_train_dir, cls)
        dst = os.path.join(output_dir, cls)
        os.makedirs(dst, exist_ok=True)

        if not os.path.exists(src):
            print(f"  WARNING: {src} not found")
            continue

        all_images = [
            f for f in os.listdir(src)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        random.seed(42)
        selected = random.sample(
            all_images, min(n_images, len(all_images)))

        for img in selected:
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    return output_dir


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(train_dir, val_dir, test_dir,
                        class_names, n_images, species):
    """
    Train PiTLiD on n_images per class and return test accuracy.
    Uses same two-phase training as original paper.
    """
    train_datagen = get_train_datagen()
    val_datagen   = get_val_datagen()

    batch = min(BATCH_TRAIN, n_images * len(class_names))

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size = IMG_SIZE,
        batch_size  = batch,
        class_mode  = 'categorical',
        classes     = class_names,
        shuffle     = True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size = IMG_SIZE,
        batch_size  = BATCH_VAL,
        class_mode  = 'categorical',
        classes     = class_names,
        shuffle     = False
    )

    test_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size = IMG_SIZE,
        batch_size  = BATCH_VAL,
        class_mode  = 'categorical',
        classes     = class_names,
        shuffle     = False
    )

    os.makedirs(TEMP_DIR, exist_ok=True)
    save_path = os.path.join(
        TEMP_DIR, f'{species}_{n_images}_best.h5')

    # Phase 1 — frozen base
    model = build_pitlid_model(
        num_classes = len(class_names), phase=1)
    model = compile_model(model, lr=1e-3)

    cb1, _ = get_callbacks(
        clr_base_lr         = 5e-4,
        clr_max_lr          = 1e-3,
        step_size           = PHASE1_STEPS,
        early_stop_patience = 3,
        model_save_path     = save_path.replace('.h5', '_p1.h5')
    )

    model.fit(
        train_gen,
        steps_per_epoch  = PHASE1_STEPS,
        epochs           = PHASE1_EPOCHS,
        validation_data  = val_gen,
        validation_steps = len(val_gen),
        callbacks        = cb1,
        verbose          = 0
    )

    # Phase 2 — all unfrozen
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    cb2, _ = get_callbacks(
        clr_base_lr         = 1e-5,
        clr_max_lr          = 1e-4,
        step_size           = PHASE2_STEPS,
        early_stop_patience = 5,
        model_save_path     = save_path
    )

    model.fit(
        train_gen,
        steps_per_epoch  = PHASE2_STEPS,
        epochs           = PHASE2_EPOCHS,
        initial_epoch    = PHASE1_EPOCHS,
        validation_data  = val_gen,
        validation_steps = len(val_gen),
        callbacks        = cb2,
        verbose          = 0
    )

    # Load best and evaluate
    if os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path)

    test_gen.reset()
    results  = model.evaluate(test_gen, verbose=0)
    accuracy = results[1] * 100

    tf.keras.backend.clear_session()
    return accuracy


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_individual_species(species, image_counts, accuracies,
                             accuracy_obt, color):
    """
    Plot learning curve for one species.
    Shows accuracy vs training images with paper's 30-image result marked.
    """
    save_dir = os.path.join(RESULTS_DIR, species.lower())
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'{species} — Training Sample Size Study',
        fontsize=13, fontweight='bold'
    )

    # Plot 1 — Learning curve
    axes[0].plot(image_counts, accuracies, 'o-',
                 color=color, linewidth=2,
                 markersize=8, label=f'{species} accuracy')

    # Mark 30-image point
    if 30 in image_counts:
        idx30 = image_counts.index(30)
        axes[0].scatter([30], [accuracies[idx30]],
                        color='red', s=150, zorder=5,
                        label=f'Paper used 30 images ({accuracies[idx30]:.1f}%)')
        axes[0].axvline(x=30, color='red', linestyle='--',
                        alpha=0.5)

    axes[0].axhline(y=90, color='orange', linestyle=':',
                    alpha=0.7, label='90% threshold')
    axes[0].axhline(y=95, color='green', linestyle=':',
                    alpha=0.7, label='95% threshold')

    axes[0].set_xlabel('Training Images Per Class')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Learning Curve')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    # Plot 2 — Bar chart with color coding
    bar_colors = []
    for acc in accuracies:
        if acc >= 95:
            bar_colors.append('#388e3c')
        elif acc >= 90:
            bar_colors.append('#f57c00')
        elif acc >= 80:
            bar_colors.append('#fbc02d')
        else:
            bar_colors.append('#d32f2f')

    bars = axes[1].bar(
        [str(n) for n in image_counts],
        accuracies,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.5
    )

    # Highlight 30-image bar
    if 30 in image_counts:
        idx30 = image_counts.index(30)
        bars[idx30].set_edgecolor('red')
        bars[idx30].set_linewidth(2.5)

    axes[1].set_xlabel('Training Images Per Class')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Accuracy by Training Size\n(Red border = paper\'s choice)')
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    save_path = os.path.join(
        save_dir, f'{species.lower()}_sample_size_study.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {save_path}")
    plt.close()


def plot_cross_species_comparison(all_results):
    """
    Key figure: Compare learning curves across all 3 species.
    Shows which species needs more images and why.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Cross-Species Training Sample Size Comparison',
        fontsize=14, fontweight='bold'
    )

    # Plot 1 — All species learning curves together
    for species, data in all_results.items():
        config = SPECIES_CONFIG[species]
        axes[0].plot(
            data['counts'], data['accuracies'],
            'o-',
            color   = config['color'],
            linewidth = 2,
            markersize = 7,
            label   = f'{species} (baseline: {config["accuracy_obt"]}%)'
        )

    # Mark 30-image line
    axes[0].axvline(x=30, color='black', linestyle='--',
                    alpha=0.6, label='Paper used 30 images')
    axes[0].axhline(y=90, color='gray', linestyle=':',
                    alpha=0.5, label='90% threshold')

    axes[0].set_xlabel('Training Images Per Class')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Learning Curves Across Species')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    # Plot 2 — Accuracy at each count grouped by species
    x         = np.arange(len(IMAGE_COUNTS))
    width     = 0.25
    species_list = list(all_results.keys())

    for i, species in enumerate(species_list):
        data   = all_results[species]
        config = SPECIES_CONFIG[species]
        axes[1].bar(
            x + i * width,
            data['accuracies'],
            width,
            label       = species,
            color       = config['color'],
            alpha       = 0.85,
            edgecolor   = 'black',
            linewidth   = 0.5
        )

    axes[1].set_xlabel('Training Images Per Class')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Accuracy Comparison Across Species')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([str(n) for n in IMAGE_COUNTS])
    axes[1].legend()
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        RESULTS_DIR, 'cross_species_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCross-species comparison saved to {save_path}")
    plt.close()


# ── Summary ───────────────────────────────────────────────────────────────────

def find_optimal(image_counts, accuracies, threshold=0.5):
    """
    Find optimal image count where accuracy plateaus.
    Plateau = less than threshold% improvement from adding more images.
    """
    for i in range(1, len(accuracies)):
        improvement = accuracies[i] - accuracies[i-1]
        if improvement < threshold and accuracies[i] >= 90:
            return image_counts[i-1]
    return image_counts[-1]


def print_summary(all_results):
    """Print complete summary across all species."""
    print("\n" + "="*65)
    print("  RESULTS: SPECIES-SPECIFIC TRAINING DATA REQUIREMENTS")
    print("="*65)

    for species, data in all_results.items():
        config  = SPECIES_CONFIG[species]
        optimal = find_optimal(data['counts'], data['accuracies'])
        acc_at_30 = None
        if 30 in data['counts']:
            acc_at_30 = data['accuracies'][data['counts'].index(30)]
        acc_at_50 = data['accuracies'][-1]

        print(f"\n  {species.upper()}")
        print(f"  {'─'*40}")
        print(f"  {'Images':8} {'Accuracy':12} {'Status'}")
        print(f"  {'─'*40}")

        for n, acc in zip(data['counts'], data['accuracies']):
            marker = " ← paper's choice" if n == 30 else ""
            status = ("Good" if acc >= 95
                     else "Ok" if acc >= 85
                     else "Poor")
            print(f"  {n:<8} {acc:.2f}%       {status}{marker}")

        print(f"\n  Optimal point    : {optimal} images")
        if acc_at_30:
            print(f"  Accuracy at 30   : {acc_at_30:.2f}%")
        print(f"  Accuracy at 50   : {acc_at_50:.2f}%")
        gain = acc_at_50 - (acc_at_30 if acc_at_30 else 0)
        print(f"  Gain 30→50 images: {gain:+.2f}%")

    print("\n" + "="*65)
    print("  KEY FINDING:")

    findings = []
    for species, data in all_results.items():
        optimal  = find_optimal(data['counts'], data['accuracies'])
        findings.append(f"{species}={optimal}")

    print(f"  Optimal images per class: {', '.join(findings)}")
    print("\n  Is 30 images optimal for all species?")

    optima = [find_optimal(d['counts'], d['accuracies'])
              for d in all_results.values()]

    if len(set(optima)) == 1:
        print(f"  YES — all species reach optimal at {optima[0]} images")
        print("  Paper's choice of 30 is validated across all species")
    else:
        print("  NO — optimal size varies by species:")
        for species, opt in zip(all_results.keys(), optima):
            print(f"    {species}: {opt} images")
        print("  Species-specific training requirements confirmed")

    print("="*65)


def save_results(all_results):
    """Save numerical results to text file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, 'results.txt')

    with open(results_file, 'w') as f:
        f.write("Species-Specific Training Data Requirements Study\n")
        f.write("="*50 + "\n\n")

        for species, data in all_results.items():
            f.write(f"{species}\n")
            f.write("-"*30 + "\n")
            for n, acc in zip(data['counts'], data['accuracies']):
                f.write(f"  {n} images: {acc:.2f}%\n")
            optimal = find_optimal(data['counts'], data['accuracies'])
            f.write(f"  Optimal: {optimal} images\n\n")

    print(f"Results saved to {results_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_novelty1():
    print("\n" + "="*65)
    print("  NOVELTY 1: SPECIES-SPECIFIC TRAINING DATA REQUIREMENTS")
    print(f"  Testing: {IMAGE_COUNTS} images per class")
    #print(f"  Species: Apple, Grape, Peach")
    print(f"  Species: Peach,Grape")
    print("="*65)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for species, config in SPECIES_CONFIG.items():

        # Check if data exists
        if not os.path.exists(config['train_dir']):
            print(f"\nSkipping {species} — {config['train_dir']} not found")
            continue

        print(f"\n{'─'*50}")
        print(f"  SPECIES: {species.upper()}")
        print(f"{'─'*50}")

        accuracies = []

        for n in IMAGE_COUNTS:
            print(f"\n  [{species}] Testing {n} images per class...")

            # Create subset
            subset_dir = os.path.join(TEMP_DIR, f'{species}_{n}')
            create_subset(n, config['train_dir'],
                         subset_dir, config['classes'])

            # Train and evaluate
            acc = train_and_evaluate(
                subset_dir,
                config['val_dir'],
                config['test_dir'],
                config['classes'],
                n, species
            )
            accuracies.append(acc)

        # Store results
        all_results[species] = {
            'counts'    : IMAGE_COUNTS,
            'accuracies': accuracies
        }

        # Plot individual species
        plot_individual_species(
            species,
            IMAGE_COUNTS,
            accuracies,
            config['accuracy_obt'],
            config['color']
        )

    # Cross-species comparison
    if len(all_results) > 1:
        plot_cross_species_comparison(all_results)

    # Summary
    print_summary(all_results)
    save_results(all_results)

    # Cleanup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Folder structure:")
    print("  novelty_results/novelty1/")
    """
    print("    apple/_sample_size_study.png")
    print("    grape/grape_sample_size_study.png")
    """
    print("    peach/peach_sample_size_study.png")
    print("    corn/corn_sample_size_study.png")
    print("    cross_species_comparison.png  ← KEY FIGURE")
    print("    results.txt")

    return all_results


if __name__ == '__main__':
    run_novelty1()
