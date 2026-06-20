"""
================================================
 MNIST Handwritten Digit Recognizer — CNN Model
================================================
Author  : You
Dataset : MNIST (60,000 train / 10,000 test images)
Target  : ≥ 98% test accuracy

Run with:
    python model/train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Reproducibility ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_cnn.h5")
PLOT_DIR   = os.path.join(os.path.dirname(__file__), "..", "app", "static")
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD & PREPROCESS DATA
# =============================================================================

def load_data():
    """Load MNIST, normalise pixel values and reshape for CNN input."""

    # Keras bundles MNIST; downloads ~11 MB on first run
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Shape: (samples, 28, 28) → (samples, 28, 28, 1)
    # CNNs expect a channel dimension (1 = grayscale)
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test  = X_test .reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # One-hot encode labels: 3 → [0,0,0,1,0,0,0,0,0,0]
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh  = keras.utils.to_categorical(y_test,  10)

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return (X_train, y_train_oh, y_train), (X_test, y_test_oh, y_test)


# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================

def build_model():
    """
    A compact but powerful CNN for MNIST.

    Architecture:
        INPUT  28×28×1
        ↓ Conv2D(32) + BatchNorm + ReLU
        ↓ Conv2D(64) + BatchNorm + ReLU
        ↓ MaxPool2D  + Dropout(0.25)
        ↓ Conv2D(64) + BatchNorm + ReLU
        ↓ MaxPool2D  + Dropout(0.25)
        ↓ Flatten
        ↓ Dense(128) + BatchNorm + ReLU + Dropout(0.5)
        ↓ Dense(10,  softmax)
    """
    model = keras.Sequential([
        # ── Block 1 ──────────────────────────────────────────────────────────
        # 32 filters of size 3×3; 'same' padding keeps spatial dims intact
        layers.Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        # Halve spatial resolution → 14×14
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ──────────────────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        # Halve again → 7×7
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Classifier head ──────────────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),

        # 10 output neurons with softmax → probability over each digit
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")

    return model


# =============================================================================
# 3. COMPILE & TRAIN
# =============================================================================

def train(model, X_train, y_train_oh, X_test, y_test_oh):
    """Compile model, set up callbacks, then fit."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",   # multi-class CE
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        # Reduce LR by ×0.5 when val_loss stalls for 3 epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        # Stop early if val_loss hasn't improved in 8 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        # Save the best checkpoint automatically
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    # Data augmentation (slight rotations/shifts) — done in-memory via ImageDataGenerator
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train_oh, batch_size=128),
        steps_per_epoch=len(X_train) // 128,
        epochs=40,
        validation_data=(X_test, y_test_oh),
        callbacks=callbacks,
    )
    return history


# =============================================================================
# 4. EVALUATE
# =============================================================================

def evaluate(model, X_test, y_test_oh, y_test_raw):
    """Print accuracy, plot training curves, and draw confusion matrix."""

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\n{'─'*40}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"{'─'*40}\n")

    # Predictions → argmax to get digit class
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("Classification Report:")
    print(classification_report(y_test_raw, y_pred,
                                target_names=[str(i) for i in range(10)]))
    return y_pred


def plot_training_history(history):
    """Save accuracy & loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[metric[0]],  label="Train",      linewidth=2)
        ax.plot(history.history[metric[1]],  label="Validation", linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history → {path}")


def plot_confusion_matrix(y_true, y_pred):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Confusion Matrix — MNIST CNN", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)

    path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix  → {path}")


# =============================================================================
# 5. MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n📦  Loading data …")
    (X_train, y_train_oh, _), (X_test, y_test_oh, y_test_raw) = load_data()

    print("\n🏗️   Building model …")
    model = build_model()

    print("\n🚀  Training …")
    history = train(model, X_train, y_train_oh, X_test, y_test_oh)

    print("\n📊  Evaluating …")
    y_pred = evaluate(model, X_test, y_test_oh, y_test_raw)

    print("\n📈  Saving plots …")
    plot_training_history(history)
    plot_confusion_matrix(y_test_raw, y_pred)

    print(f"\n✅  Model saved to: {MODEL_PATH}")
