import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys

try:
    from scipy import ndimage
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print("\nError: Required packages not found.")
    print("Please run: pip install scipy tensorflow\n")
    sys.exit(1)

print("Starting enhanced training process...")
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Enhanced preprocessing
print("Enhanced preprocessing...")
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Optimized data augmentation
datagen = ImageDataGenerator(
    rotation_range=8,  # Reduced from 10
    width_shift_range=0.15,  # Reduced from 0.2
    height_shift_range=0.15,  # Reduced from 0.2
    shear_range=0.15,  # Reduced from 0.2
    zoom_range=0.15,  # Reduced from 0.2
    fill_mode="nearest",
)

# Build optimized model
model = keras.Sequential(
    [
        # Input
        keras.layers.Input(shape=(28, 28, 1)),
        # First Conv Block
        keras.layers.Conv2D(32, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(32, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((2, 2)),
        # Second Conv Block
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        # Dense Layers
        keras.layers.Flatten(),
        keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

# Compile with optimized settings
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("\nStarting optimized training...")
try:
    # Train with better monitoring
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),  # Smaller batch size
        epochs=20,  # Reduced from 30
        validation_data=(x_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=0.00001, verbose=1
            ),
        ],
        verbose=1,
    )

    # Evaluate
    print("\nFinal evaluation...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save model
    print("\nSaving model...")
    model.save("handwriting_model.h5")
    print("Model saved successfully!")

except Exception as e:
    print(f"\nError during training: {str(e)}")
    print("Please ensure you have installed all required packages:")
    print("pip install tensorflow numpy scipy")
    exit(1)

print("\nTraining complete! You can now run the server.")
