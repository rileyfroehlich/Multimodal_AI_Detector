import pdfplumber
import docx
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path

def text_pipeline(text_file, filetype):
  
  #TODO: load as bytes file
  #try:
    text = ""
    if filetype is ("pdf"):
      with pdfplumber.open(text_file) as pdf:
        for page in pdf.pages:
          text = text + " " + page.extract_text()
    elif filetype is ("docx"):
      doc = docx.Document(text_file)
      for paragraph in doc.paragraphs:
        text += paragraph.text

    elif filetype is ("txt"):
      with open(text_file, 'r') as f:
        text = f.read()
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    tokenizer_path = f'{BASE_DIR}\\models_text\\text_pretrained_tokenizer'
    with open(tokenizer_path, 'rb') as t:
      tokenizer = pickle.load(t)

    tokenizer.fit_on_texts(text)
    print('TOKENIZED')
    model_path = f'{BASE_DIR}\\models_text\\text_model.keras'
    model = load_model(model_path)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=1000, padding='post')
    AI_score = model.predict(padded_sequences)
    print('THIS IS A PREDICTION')
    return (AI_score > .5, AI_score)
  #except:
    return (None, "")