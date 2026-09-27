import tensorflow as tf
from tensorflow.keras.datasets import mnist

print("Loading model...")

model = tf.keras.models.load_model("digit_model.h5")

print("Loading test data...")

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n==============================")
print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
print("==============================")