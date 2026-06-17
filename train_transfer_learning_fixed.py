import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("PNEUMONIA DETECTION MODEL TRAINING - TRANSFER LEARNING")
print("="*70)

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
print("\nSetting up data generators with augmentation...")

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

# ==================== BUILD MODEL ====================
print("\n" + "="*70)
print("Building Transfer Learning Model (VGG16)...")
print("="*70)

# Load VGG16 base model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Build model using Functional API (more compatible)
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print("\n✅ Model created successfully!")
print(f"Total layers: {len(model.layers)}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights])}")

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

print("✅ Model compiled successfully!")

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
        'best_transfer_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ==================== PHASE 1: TRAIN WITH FROZEN BASE ====================
print("\n" + "="*70)
print("PHASE 1: Training with frozen base model (10 epochs)")
print("="*70)
print("This trains only the top layers while keeping VGG16 frozen...")

history_phase1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Phase 1 completed!")

# ==================== PHASE 2: FINE-TUNING ====================
print("\n" + "="*70)
print("PHASE 2: Fine-tuning - Unfreezing top layers (20 epochs)")
print("="*70)

# Unfreeze base model
base_model.trainable = True

# Freeze all layers except the last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

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

print("This fine-tunes the last layers of VGG16...")

# Continue training
history_phase2 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Phase 2 completed!")

# ==================== SAVE MODEL ====================
print("\n" + "="*70)
print("Saving model...")
print("="*70)

model.save('x-ray-classification.h5')
print("✅ Model saved as: x-ray-classification.h5")

# ==================== EVALUATE ====================
print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET")
print("="*70)

test_results = model.evaluate(test_generator, verbose=1)
print(f"\nTest Results:")
print(f"Loss:      {test_results[0]:.4f}")
print(f"Accuracy:  {test_results[1]:.4f}")
print(f"Precision: {test_results[2]:.4f}")
print(f"Recall:    {test_results[3]:.4f}")
print(f"AUC:       {test_results[4]:.4f}")

# Predictions
print("\nGenerating predictions...")
y_pred_prob = model.predict(test_generator, verbose=1)
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

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")

# ==================== COMBINE HISTORY ====================
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

# ==================== VISUALIZATIONS ====================
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2, color='#667eea')
axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#764ba2')
axes[0, 0].axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2, color='#667eea')
axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#764ba2')
axes[0, 1].axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2, color='#667eea')
axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2, color='#764ba2')
axes[1, 0].axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2, color='#667eea')
axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2, color='#764ba2')
axes[1, 1].axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_history.png', dpi=300, bbox_inches='tight')
print("✅ Saved: transfer_learning_history.png")
plt.close()

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
print("✅ Saved: transfer_confusion_matrix.png")
plt.close()

# ==================== FINAL SUMMARY ====================
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)

print("\n📊 Final Test Performance:")
print(f"   Accuracy:  {test_results[1]*100:.2f}%")
print(f"   Precision: {test_results[2]*100:.2f}%")
print(f"   Recall:    {test_results[3]*100:.2f}%")
print(f"   AUC:       {test_results[4]:.4f}")

print("\n📁 Generated Files:")
print("   1. ✅ x-ray-classification.h5 (Use this in your Flask app!)")
print("   2. ✅ best_transfer_model.h5")
print("   3. ✅ transfer_learning_history.png")
print("   4. ✅ transfer_confusion_matrix.png")

print("\n🚀 Next Steps:")
print("   1. Copy 'x-ray-classification.h5' to your Flask project")
print("   2. Run: python app.py")
print("   3. Test with different X-ray images")
print("   4. Your model should now predict correctly!")

print("\n" + "="*70)