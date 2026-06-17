import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# Dataset paths
TRAIN_DIR = 'chest_xray/train'
VAL_DIR = 'chest_xray/val'
TEST_DIR = 'chest_xray/test'

# ==================== DATA GENERATORS ====================
print("Setting up data generators with augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nDataset Statistics:")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Classes: {train_generator.class_indices}")

# ==================== CLASS WEIGHTS ====================
from sklearn.utils import class_weight

class_counts = np.bincount(train_generator.classes)
print(f"\nClass Distribution:")
print(f"Normal: {class_counts[0]} ({class_counts[0]/len(train_generator.classes)*100:.1f}%)")
print(f"Pneumonia: {class_counts[1]} ({class_counts[1]/len(train_generator.classes)*100:.1f}%)")

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ==================== TRANSFER LEARNING MODEL ====================
print("\nBuilding Transfer Learning Model (VGG16)...")

def create_transfer_model(base_model_name='VGG16'):
    """
    Create model using transfer learning
    Options: 'VGG16', 'ResNet50', 'DenseNet121'
    """
    
    # Load pre-trained base model
    if base_model_name == 'VGG16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif base_model_name == 'DenseNet121':
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build complete model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model, base_model

# Create model
model, base_model = create_transfer_model('VGG16')

# Display model architecture
print("\nModel Architecture:")
model.summary()

# ==================== COMPILE MODEL ====================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

# ==================== CALLBACKS ====================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_transfer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ==================== PHASE 1: TRAIN WITH FROZEN BASE ====================
print("\n" + "="*70)
print("PHASE 1: Training with frozen base model")
print("="*70)

history_phase1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ==================== PHASE 2: FINE-TUNING ====================
print("\n" + "="*70)
print("PHASE 2: Fine-tuning - Unfreezing top layers")
print("="*70)

# Unfreeze the last few layers of base model for fine-tuning
base_model.trainable = True

# Freeze all layers except the last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print(f"\nTrainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

# Continue training
history_phase2 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ==================== COMBINE HISTORY ====================
# Combine both training phases
history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'precision': history_phase1.history['precision'] + history_phase2.history['precision'],
    'val_precision': history_phase1.history['val_precision'] + history_phase2.history['val_precision'],
    'recall': history_phase1.history['recall'] + history_phase2.history['recall'],
    'val_recall': history_phase1.history['val_recall'] + history_phase2.history['val_recall']
}

# ==================== SAVE MODEL ====================
model.save('x-ray-classification.h5')
print("\nFinal model saved as 'x-ray-classification.h5'")

# ==================== EVALUATE ====================
print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET")
print("="*70)

test_results = model.evaluate(test_generator, verbose=1)
print(f"\nTest Results:")
print(f"Loss: {test_results[0]:.4f}")
print(f"Accuracy: {test_results[1]:.4f}")
print(f"Precision: {test_results[2]:.4f}")
print(f"Recall: {test_results[3]:.4f}")
print(f"AUC: {test_results[4]:.4f}")

# Predictions
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")

# ==================== VISUALIZATIONS ====================
# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].axvline(x=10, color='r', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].axvline(x=10, color='r', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
axes[1, 0].axvline(x=10, color='r', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
axes[1, 1].axvline(x=10, color='r', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history saved as 'transfer_learning_history.png'")
plt.show()

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16})
plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('transfer_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'transfer_confusion_matrix.png'")
plt.show()

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("1. x-ray-classification.h5 (Final trained model)")
print("2. best_transfer_model.h5 (Best model checkpoint)")
print("3. transfer_learning_history.png (Training metrics)")
print("4. transfer_confusion_matrix.png (Confusion matrix)")
print("\nYou can now use 'x-ray-classification.h5' in your Flask app!")