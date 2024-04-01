#Imports
import io
import pandas as pd
import numpy as np
import librosa
import wave
import pickle
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path

#Takes in multi channel audio clip and returns 1 channel
#INPUT: file = .wav file
#RETURNS: A .wav file with 1 channel
def convert_stereo_to_mono(file):
    # Read the input file into a BytesIO object
    file_bytesio = io.BytesIO(file)
    with wave.open(file_bytesio, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        if channels > 1:
            # Create a BytesIO object to store the mono audio
            mono_audio_bytesio = io.BytesIO()

            # Create a new wave file object for writing mono audio
            mono_audio = wave.open(mono_audio_bytesio, 'wb')
            mono_audio.setparams(wav_file.getparams())
            mono_audio.setnchannels(1)

            # Read frames from stereo and write them to mono
            frames = wav_file.readframes(wav_file.getnframes())
            mono_audio.writeframes(frames)

            # Close the mono audio wave file object
            mono_audio.close()

            # Get the mono audio bytes from the BytesIO object
            mono_audio_bytes = mono_audio_bytesio.getvalue()

        else:
            # If already mono, return the original file bytes
            mono_audio_bytes = file_bytesio.getvalue()

    return mono_audio_bytes

#Takes a .mp3 file and converts to a .wav file
#INPUT: the .mp3 file
#RETURN: A .wav file
def convert_mp3_to_wav(file):

    audio_data = AudioSegment.from_mp3(io.BytesIO(file))
    wav_data = io.BytesIO()
    audio_data.export(wav_data, format="wav")
    wav_data.seek(0)  # Reset file pointer for reading
    return wav_data.getvalue()

#Takes a .m4a file and converts to a .wav file
#INPUT: .m4a file
#RETURN: .wav file
def convert_m4a_to_wav(file):
    
    m4a_audio = AudioSegment.from_file(io.BytesIO(file), format="m4a")
    buffer = io.BytesIO()
    m4a_audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer.getvalue()

#Takes a .flac file and converts to a .wav file
#INPUT: .flac file
#RETURN: .wav file
def convert_flac_to_wav(file):

  audio_data = AudioSegment.from_file(io.BytesIO(file), format="flac")
  wav_data = io.BytesIO()
  audio_data.export(wav_data, format="wav")
  wav_data.seek(0)
  return wav_data.getvalue()

#Takes in the file and returns extracted data using librosa
#INPUT: .wav file
#OUTPUT: Pd.dataframe row of features 1x27

def extract_audio_data(file):
  # Use io.BytesIO to create a file-like object from the bytes data
  wav_io = io.BytesIO(file)
  data, samplerate = sf.read(wav_io, dtype='float32')
  mfccs_df = pd.DataFrame()

  #COMPUTE FEATURES
  #MFCCS (ARRAY)
  mfccs_features = librosa.feature.mfcc(y=data, sr=samplerate, n_mfcc=20)
  mfccs_features_scaled = np.mean(mfccs_features.T, axis=0)
  #ROOT MEAN SQUARED
  rms = np.mean(librosa.feature.rms(y=data))
  #SPECTRAL CENTROID
  spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=data, sr=samplerate))
  #SPECTRAL BANDWIDTH
  spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=samplerate))
  #SPECTRAL ROLLOFF
  spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=samplerate))
  #CHROMA STFT
  chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=samplerate))
  #ZERO CROSSING RATE
  zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data))

  #Combine MFCCS and features into one DF
  features_list = [samplerate, rms, spectral_centroids, spectral_bandwidth, spectral_rolloff, chroma_stft, zero_crossing_rate]
  numpy_series = np.concatenate((features_list, mfccs_features_scaled.tolist()))
  mfccs_series = pd.Series(numpy_series)
  mfccs_df = pd.concat([mfccs_df, mfccs_series], axis=1)

  return mfccs_df

#Takes in the file and the filetype and predicts with confidence the probability
#of AI created audio
#INPUT: file - file types .mp3 and .wav files
#INPUT: filetype - the extenstion of the file as "mp3" or "wav"
#OUTPUT: AI Generated flag - False = REAL // True = AI
#OUTPUT: Confidence Percent
def audio_detection(file, filetype):
  #Check file extenstion, change to .wav
  # MAKE THIS AN ASYNC METHOD TO KEEP FILE TYPES THE SAME
  if filetype != 'wav':
    if filetype == 'mp3':
      file = convert_mp3_to_wav(file)
    elif filetype == 'flac':
      file = convert_flac_to_wav(file)
    elif filetype == 'm4a':
      file = convert_m4a_to_wav(file)
    else:
      raise ValueError("Acceptable audio file types are .mp3, .m4a, .flac, or .wav, sorry!")

  #Check Stereo audio, convert to mono
  wav_file = convert_stereo_to_mono(file)
  #Extract audio data
  extracted_audio_df = extract_audio_data(wav_file)
  extracted_audio_df = extracted_audio_df.T

  #Load model
  BASE_DIR = Path(__file__).resolve(strict=True).parent
  model_path = f'{BASE_DIR}/models_audio/audio_random_forest_model.pkl'
  
  with open(model_path, 'rb') as file:
    model = pickle.load(file)

  #Predict with model
  prediction = model.predict(extracted_audio_df)
  confidence = (model.predict_proba(extracted_audio_df))[0][0]
  AI_bool = prediction > .5

  if AI_bool:
     confidence = 1 - confidence

  if confidence == .5:
    confidence =.51

  return AI_bool, float(confidence)