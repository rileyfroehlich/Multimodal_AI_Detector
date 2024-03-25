import pdfplumber
import docx
from tensorflow.keras.models import load_model
import pickle


def text_pipeline(text_file):
  try:
    text = ""
    if text_file.endswith(".pdf"):
      with pdfplumber.open(text_file) as pdf:
        for page in pdf.pages:
          text = text + " " + page.extract_text()
    elif text_file.endswith(".docx"):
      doc = docx.Document(text_file)
      for paragraph in doc.paragraphs:
        text += paragraph.text

    elif text_file.endswith(".txt"):
      with open(text_file, 'r') as f:
        text = f.read()

    with open('PRETRAINED_TOKENIZER_HERE', 'rb') as t:
      tokenizer = pickle.load(t)

    tokenizer.fit_on_texts(text)

    model = load_model("TEXT_MODEL_HERE")
    sequences = tokenizer.texts_to_sequences(df_subset["text"])
    padded_sequences = pad_sequences(sequences, maxlen=1000, padding='post')
    return (model.predict(text) > .5, f"{model.predict(text)}")
  except:
    return (None, "")

  from tensorflow.keras.models import load_model

  def preprocess_image(image_path):
    # Read image
    img = tf.io.read_file(image_path)
    # Decode image
    img = tf.image.decode_image(img, channels=3)
    # Resize image
    img = tf.image.resize(img, (256, 256))
    # Normalize image
    img = img / 255.0  # Assuming images are in the range [0, 255]
    return img

  def predict_with_model(image_file):
    # Predict using the model
    try:
      model = load_model("MODEL_PATH_HERE")
      prediction = model.predict(image_file)
      # Convert predictions to binary
      binary_predictions = prediction > 0.5
      return binary_predictions, prediction
    except:
      return (None, "")