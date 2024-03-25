from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> tf.Tensor:

    image = Image.open(io.BytesIO(image_bytes))
    image.verify()  # Verify image data integrity
    image.close()   # Close the image file
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
    model_path = f'{BASE_DIR}\models_image\image_detection_model.h5'
    model_path = 'C:\\Users\\riley\\OneDrive\\Documents\\Code for school\\Capstone\\app\\models\\models_image\\image_detection_model.h5'
  # Predict using the model
#  try:
    model = load_model(model_path)
    prediction = model.predict(img)
    # Convert predictions to binary
    binary_predictions = prediction > 0.5
    print(prediction)
    print(type(prediction))
    return binary_predictions, prediction
#  except:
    return (None, "")