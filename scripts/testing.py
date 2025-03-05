#%%
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.special import softmax
import json

from numpy import ndarray
from typing import List
#%%
duration: int = 1;  # In seconds. Duration of the recording.
fs: int = 16000;  # Frequency of recording, 22050 samples per seconds. Continues -> Discrete.
frames: int = duration * fs;  # Frame count.

categories: List[str] = ["yes", "no", "unknown"]

with open("normalization_values.json", "r") as f:
    norm_data = json.load(f)

mean = np.array(norm_data["mean"], dtype=np.float32)
stddev = np.array(norm_data["stddev"], dtype=np.float32)



#%%
def get_waveform_file(file_name: str) -> ndarray:
    file_path: str = f"./data/{file_name}"
    file_tensor = tf.io.read_file(file_path)
    audio_tensor, _ = tf.audio.decode_wav(file_tensor)
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    return audio_tensor
#%%
print("Speak Now")
recording: ndarray = sd.rec(frames=frames, samplerate=fs, channels=1)
sd.wait(ignore_errors=False)

audio = np.squeeze(recording).astype(np.float32)
spectrogram = tf.signal.stft(audio, frame_length=256, frame_step=64, fft_length=256)

spectrogram = tf.abs(spectrogram)
spectrogram = tf.reshape(spectrogram, (1, 1, 129, 247))

spectrogram = (spectrogram - mean) / stddev

tf_model = tf.lite.Interpreter(model_path="models/model_0.tflite")
input_details = tf_model.get_input_details()
output_details = tf_model.get_output_details()

print("Spectrogram Shape: ", spectrogram.shape)
print("Input Shape: ", input_details[0]["shape"])
#%%
tf_model.allocate_tensors()
tf_model.set_tensor(input_details[0]["index"], spectrogram)
tf_model.invoke()

tf_model_prediction_coefficients = tf_model.get_tensor(output_details[0]["index"])

tf_model_prediction = categories[tf.argmax(tf_model_prediction_coefficients, axis=1).numpy()[0]]

probabilities = softmax(tf_model_prediction_coefficients)

print("Categories: ", categories)
print("Coefficients: ", tf_model_prediction_coefficients)
print("Probabilities: ", probabilities)
print("Model Prediction: ", tf_model_prediction)

print("Finished!")