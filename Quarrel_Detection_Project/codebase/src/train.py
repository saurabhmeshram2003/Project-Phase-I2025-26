# src/train.py
"""
Training script for Quarrel Detection CNN Model
Uses MobileNetV2 with transfer learning
"""

import os
import sys
import numpy as np
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Import config
from config import (
    DATASET_DIR, MODEL_DIR, QUARREL_MODEL_PATH, IMG_SIZE, 
    BATCH_SIZE, EPOCHS, VAL_SPLIT, LEARNING_RATE, 
    EARLY_STOP_PATIENCE, ROTATION_RANGE, WIDTH_SHIFT_RANGE,
    HEIGHT_SHIFT_RANGE, HORIZONTAL_FLIP, ZOOM_RANGE,
    DENSE_UNITS, DROPOUT_RATE, FREEZE_LAYERS, LOGS_DIR,
    SAVE_TRAINING_PLOT, RANDOM_SEED
)
from utils import (
    plot_training_history, save_training_history, 
    create_timestamp, print_metrics
)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build MobileNetV2-based CNN model for quarrel detection.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes (2: normal, quarrel)
        
    Returns:
        Compiled Keras model
    """
    # Load base MobileNetV2 without top layer
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze most layers (fine-tune only last few)
    for layer in base_model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    
    # Add custom classification head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    """Main training function"""
    print("="*70)
    print("QUARREL DETECTION - MODEL TRAINING")
    print("="*70)
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Model Output: {QUARREL_MODEL_PATH}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Validation Split: {VAL_SPLIT}")
    print("="*70)
    
    # Check if dataset exists
    if not Path(DATASET_DIR).exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("Please run preprocess_dataset.py first!")
        sys.exit(1)
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VAL_SPLIT,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        zoom_range=ZOOM_RANGE,
        fill_mode='nearest'
    )
    
    # Validation generator (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VAL_SPLIT
    )

    print("\n[INFO] Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=RANDOM_SEED
    )

    print("[INFO] Loading validation data...")
    val_gen = val_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=RANDOM_SEED
    )

    print(f"\n[INFO] Class indices: {train_gen.class_indices}")
    print(f"[INFO] Training samples: {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}")
    
    # Check for class imbalance
    if train_gen.samples < 100:
        print("\n[WARNING] Very few training samples! Consider collecting more data.")
    
    # Build model
    print("\n[INFO] Building model...")
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Print model summary
    print("\n[INFO] Model Summary:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        str(QUARREL_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Train
    print("\n[INFO] Starting training...")
    print("="*70)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    print("\n" + "="*70)
    print("[SUCCESS] Training complete!")
    print(f"Model saved at: {QUARREL_MODEL_PATH}")
    print("="*70)
    
    # Print final metrics
    final_metrics = {
        "Final Training Accuracy": history.history['accuracy'][-1],
        "Final Validation Accuracy": history.history['val_accuracy'][-1],
        "Final Training Loss": history.history['loss'][-1],
        "Final Validation Loss": history.history['val_loss'][-1],
        "Best Validation Accuracy": max(history.history['val_accuracy']),
        "Total Epochs Trained": len(history.history['accuracy'])
    }
    print_metrics(final_metrics)
    
    # Save training history
    timestamp = create_timestamp()
    history_path = LOGS_DIR / f"training_history_{timestamp}.json"
    save_training_history(history, history_path)
    
    # Plot and save training curves
    if SAVE_TRAINING_PLOT:
        plot_path = LOGS_DIR / f"training_plot_{timestamp}.png"
        plot_training_history(history, save_path=plot_path)
    
    # Evaluate on validation set and generate confusion matrix
    print("\n[INFO] Generating confusion matrix and performance metrics...")
    evaluate_model(model, val_gen, timestamp)
    
    print("\n[INFO] Next steps:")
    print("  1. Check confusion matrix in logs/ directory")
    print("  2. Test real-time detection: python src/detection.py")
    print("="*70)

def evaluate_model(model, val_gen, timestamp):
    """
    Evaluate model and generate confusion matrix, ROC curve, and metrics
    """
    # Get predictions
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    
    # Class names
    class_names = list(val_gen.class_indices.keys())
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Quarrel Detection Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.2%}', 
             ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
    
    cm_path = LOGS_DIR / f"confusion_matrix_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Confusion matrix saved: {cm_path}")
    plt.close()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(report)
    
    # Save classification report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = LOGS_DIR / f"classification_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"[✓] Classification report saved: {report_path}")
    
    # Performance metrics summary
    metrics = {
        "Overall Accuracy": accuracy,
        "Normal Precision": report_dict['normal']['precision'],
        "Normal Recall": report_dict['normal']['recall'],
        "Normal F1-Score": report_dict['normal']['f1-score'],
        "Quarrel Precision": report_dict['quarrel']['precision'],
        "Quarrel Recall": report_dict['quarrel']['recall'],
        "Quarrel F1-Score": report_dict['quarrel']['f1-score'],
        "Macro Avg F1": report_dict['macro avg']['f1-score'],
        "Weighted Avg F1": report_dict['weighted avg']['f1-score']
    }
    
    # ROC Curve (for binary classification)
    if len(class_names) == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_probs[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Quarrel Detection Model', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        roc_path = LOGS_DIR / f"roc_curve_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"[✓] ROC curve saved: {roc_path}")
        plt.close()
        
        metrics["ROC AUC Score"] = roc_auc
    
    # Save metrics summary
    metrics_path = LOGS_DIR / f"performance_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[✓] Performance metrics saved: {metrics_path}")
    
    # Print metrics summary
    print("\n" + "="*70)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*70)
    for key, value in metrics.items():
        print(f"{key:.<50} {value:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()