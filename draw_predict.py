from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

print("Model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json["image"]

        # Base64 se image data nikalo
        image_data = data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        # Image ko grayscale me convert karo
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Numpy array
        image = np.array(image)

        # ------------------------------------------------
        # MNIST STYLE PREPROCESSING
        # ------------------------------------------------

        # White pixels find karo
        coords = np.argwhere(image > 30)

        if coords.size == 0:
            return jsonify({
                "error": "Please draw a digit first."
            })

        # Bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Digit crop karo
        cropped = image[y_min:y_max + 1, x_min:x_max + 1]

        # Original size
        h, w = cropped.shape

        # Digit ko maximum 20x20 ke andar fit karo
        if h > w:
            new_h = 20
            new_w = max(1, int(w * 20 / h))
        else:
            new_w = 20
            new_h = max(1, int(h * 20 / w))

        digit = Image.fromarray(cropped)

        # Resize
        digit = digit.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )

        digit = np.array(digit)

        # ------------------------------------------------
        # 28x28 IMAGE
        # ------------------------------------------------

        final_image = np.zeros((28, 28), dtype=np.uint8)

        # Pehle roughly center me place karo
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        final_image[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = digit

        # ------------------------------------------------
        # CENTER OF MASS ALIGNMENT
        # ------------------------------------------------

        total_mass = np.sum(final_image)

        if total_mass > 0:

            y_indices, x_indices = np.indices(final_image.shape)

            center_x = np.sum(
                x_indices * final_image
            ) / total_mass

            center_y = np.sum(
                y_indices * final_image
            ) / total_mass

            # Desired center
            target_x = 13.5
            target_y = 13.5

            shift_x = int(round(target_x - center_x))
            shift_y = int(round(target_y - center_y))

            # Shift image
            shifted = np.zeros_like(final_image)

            source_x_start = max(0, -shift_x)
            source_x_end = min(28, 28 - shift_x)

            source_y_start = max(0, -shift_y)
            source_y_end = min(28, 28 - shift_y)

            dest_x_start = max(0, shift_x)
            dest_x_end = dest_x_start + (
                source_x_end - source_x_start
            )

            dest_y_start = max(0, shift_y)
            dest_y_end = dest_y_start + (
                source_y_end - source_y_start
            )

            if (
                source_x_end > source_x_start
                and source_y_end > source_y_start
            ):
                shifted[
                    dest_y_start:dest_y_end,
                    dest_x_start:dest_x_end
                ] = final_image[
                    source_y_start:source_y_end,
                    source_x_start:source_x_end
                ]

            final_image = shifted

        # ------------------------------------------------
        # NORMALIZE
        # ------------------------------------------------

        final_image = final_image.astype("float32") / 255.0

        # Model input shape: (1, 28, 28, 1)
        final_image = final_image.reshape(
            1, 28, 28, 1
        )

        # ------------------------------------------------
        # PREDICTION
        # ------------------------------------------------

        prediction = model.predict(
            final_image,
            verbose=0
        )

        digit_prediction = int(
            np.argmax(prediction)
        )

        confidence = float(
            np.max(prediction) * 100
        )

        return jsonify({
            "digit": digit_prediction,
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)