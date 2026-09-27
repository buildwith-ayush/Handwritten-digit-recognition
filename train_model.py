import tensorflow as tf

print("Loading MNIST dataset...")

# Dataset load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize 0-255 -> 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Building model...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# Save model
model.save("digit_model.h5")

print("\nModel saved successfully as digit_model.h5")

# Test accuracy
loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")