from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path
import io

def preprocess_image(image_bytes: bytes) -> tf.Tensor:

    try:
        # Decode image (assuming JPEG format)
        img = tf.image.decode_jpeg(image_bytes, channels=3)
        # Resize image
        img = tf.image.resize(img, (256, 256))
        # Convert image to float32 and normalize
        img = tf.cast(img, tf.float32) / 255.0
        return img
    except tf.errors.InvalidArgumentError:
        raise ValueError("Invalid image data or format")

def image_pipeline(image_file, filetype):
    img = preprocess_image(image_file)
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    model_path = f'{BASE_DIR}\models_image\image_model.keras'
  # Predict using the model
#  try:
    model = load_model(model_path)
    print("loaded the model")
    prediction = model.predict(img)
    # Convert predictions to binary
    binary_predictions = prediction > 0.5
    print(prediction)
    print(type(prediction))
    return binary_predictions, prediction
#  except:
    return (None, "")