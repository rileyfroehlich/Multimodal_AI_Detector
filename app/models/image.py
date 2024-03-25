from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path

def preprocess_image(file):
  # Read image
  img = tf.io.read_file(file)
  # Decode image
  img = tf.image.decode_image(img, channels=3)
  # Resize image
  img = tf.image.resize(img, (256, 256))
  # Normalize image
  img = img / 255.0  # Assuming images are in the range [0, 255]
  return img

def image_pipeline(image_file):
  img = preprocess_image(image_file)
  BASE_DIR = Path(__file__).resolve(strict=True).parent
  model_path = f'{BASE_DIR}\models_image\image_detection_model.h5'
  # Predict using the model
  try:
    model = load_model(model_path)
    prediction = model.predict(img)
    # Convert predictions to binary
    binary_predictions = prediction > 0.5
    return binary_predictions, prediction
  except:
    return (None, "")