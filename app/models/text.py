import pdfplumber
import docx
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

def text_pipeline(text_file):
  
  #QUESTIONS: what is df_subset refrencing?
  #TODO: load as bytes file
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