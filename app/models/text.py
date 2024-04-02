import pdfplumber
import docx
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import io

def text_pipeline(text_file, filetype):
  
    text = ""
    if filetype == ("txt"):
      text = text_file.decode("utf-8")

    else:
      text_file = io.BytesIO(text_file)

      if filetype == ("pdf"):
        with pdfplumber.open(text_file) as pdf:
          for page in pdf.pages:
            text = text + " " + page.extract_text()
      elif filetype == ("docx"):
        doc = docx.Document(text_file)
        for paragraph in doc.paragraphs:
          text += paragraph.text

    BASE_DIR = Path(__file__).resolve(strict=True).parent
    tokenizer_path = f'{BASE_DIR}/models_text/text_pretrained_tokenizer.pkl'
    with open(tokenizer_path, 'rb') as t:
      tokenizer = pickle.load(t)
    model_path = f'{BASE_DIR}/models_text/text_model_final.keras'
    model = load_model(model_path)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=1000, padding='post')
    confidence = model.predict(padded_sequence)
    AI_bool = confidence > .5
    if not AI_bool:
     confidence = 1 - confidence
    return (AI_bool, float(confidence), text)
