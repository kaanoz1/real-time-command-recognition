#%% md
# # Project Purpose:
# 
# **The goal of this project is to develop a deep learning model capable of recognizing specific voice commands (“yes” or “no”) during an ordinary conversation. The model will identify these commands with a certain level of confidence and display the results in the console.**
# 
# 
# ## About dataset:
# 
# All dataset files are located in the /data folder. The dataset consists of three categories:
# - All recordings of the word “yes” are stored in /data/yes_data/
# - All recordings of the word “no” are stored in /data/no_data/.
# - Any other sounds, including daily conversation fragments and background noise, fall under the "unknown" category. Since these recordings may be longer than one second, they must be split into 1-second segments before training. The model will then be trained to recognize these segments as "unknown".
#%% md
# ### Importing Modules and types:
#%%
### Packages:
import os
from pydub import AudioSegment
import tensorflow as tf
import gc
from scipy.special import softmax
import numpy as np
from tensorflow.keras.layers import Rescaling, Normalization, TextVectorization, Resizing
from tensorflow.keras import models, layers
import glob
import librosa
import soundfile as sf


### Typings:
from typing import Final, List, Optional
from tensorflow import Tensor
from tensorflow.python.types.data import DatasetV2


#%% md
# ### Variables and functions initialization:
#%%
CATEGORIES: Final[List[str]] = ["yes", "no", "unknown"]
DATASET_FILE_EXTENSIONS: Final[List[str]] = [".wav", ".mp3"]
PATH_FOR_YES_DATASET: Final[str] = "data/yes_data"
PATH_FOR_NO_DATASET: Final[str] = "data/no_data"
PATH_FOR_UNPROCESSED_UNKNOWN_DATASET: Final[str] = "data/unknown_data_unprocessed"
PATH_FOR_PROCESSED_UNKNOWN_DATASET: Final[str] = "data/unknown_data_processed"
AUDIO_CLIP_LENGTH_MS: Final[int] = 1000


def check_audio_file_is_valid(file_path: str) -> (bool, Optional[AudioSegment]):
    is_file_exists: bool = os.path.exists(file_path) or os.path.isfile(file_path)

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension not in DATASET_FILE_EXTENSIONS:
        raise ValueError(f"Invalid file extension: {file_path}. Expected: {DATASET_FILE_EXTENSIONS}")

    if file_extension == ".wav":
        audio = AudioSegment.from_wav(file_path)
    elif file_extension == ".mp3":
        audio = AudioSegment.from_mp3(file_path)

    file_name: str = os.path.basename(file_path)
    duration: int = len(audio)

    if duration < AUDIO_CLIP_LENGTH_MS:
        raise ValueError(f"Invalid audio file duration: {duration} ms, for file {file_name}")

    return True, audio


def check_label_is_valid(label: str) -> bool:
    return label in CATEGORIES


def extract_label(file_path_tensor: Tensor) -> str:
    # Convert tensor to string
    file_path = file_path_tensor.numpy().decode("utf-8")  # Ensure it's a proper string

    if not isinstance(file_path, str):
        raise ValueError(f"Invalid file path format: {file_path}")

    is_valid_file, _ = check_audio_file_is_valid(file_path)

    if not is_valid_file:
        raise ValueError(f"Invalid file path to extract label: {file_path}")

    full_dir_name = os.path.dirname(file_path)
    dir_name = os.path.basename(full_dir_name)
    label = dir_name.split("_")[0]

    if not check_label_is_valid(label):
        raise ValueError(f"Invalid label: {label} for file path: {file_path}")

    return label



def process_unprocessed_unknown_data(unprocessed_dataset_path: str, processed_data_set_path: str) -> None:
    counter: int = 0
    splitter_counter: int = 0
    found_file_counter: int = 0
    TARGET_SAMPLE_RATE: Final[int] = 16000

    for unprocessed_unknown_file_name in os.listdir(unprocessed_dataset_path):
        file_path = os.path.join(unprocessed_dataset_path, unprocessed_unknown_file_name)

        base_name = os.path.splitext(unprocessed_unknown_file_name)[0]
        processed_files = glob.glob(os.path.join(processed_data_set_path, f"{base_name}_part*.wav"))

        if processed_files:
            found_file_counter += 1
            print(f"Skipping {unprocessed_unknown_file_name}, already processed ({len(processed_files)} parts found).")
            continue

        is_valid_file, audio = check_audio_file_is_valid(file_path)

        if not is_valid_file or audio is None:
            raise FileNotFoundError(f"Invalid file for path: {unprocessed_unknown_file_name}")

        y, sr = librosa.load(file_path, sr=None)  # Load with original sample rate

        if sr != TARGET_SAMPLE_RATE:
            print(f"Resampling {unprocessed_unknown_file_name} from {sr} Hz to {TARGET_SAMPLE_RATE} Hz")
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

        duration_ms: int = int((len(y) / TARGET_SAMPLE_RATE) * 1000)  # Convert samples to milliseconds

        for i, start in enumerate(range(0, duration_ms, AUDIO_CLIP_LENGTH_MS)):
            segment_start_sample = int((start / 1000) * TARGET_SAMPLE_RATE)
            segment_end_sample = segment_start_sample + int((AUDIO_CLIP_LENGTH_MS / 1000) * TARGET_SAMPLE_RATE)

            segment = y[segment_start_sample:segment_end_sample]
            if len(segment) < int((AUDIO_CLIP_LENGTH_MS / 1000) * TARGET_SAMPLE_RATE):
                continue

            new_file_name: str = f"{base_name}_part{i}.wav"
            new_file_path: str = os.path.join(processed_data_set_path, new_file_name)

            sf.write(new_file_path, segment, TARGET_SAMPLE_RATE)

            print(f"Saved: {new_file_name}.")
            splitter_counter += 1

        counter += 1

    print(f"Processed {counter} new files, split into {splitter_counter} parts. Found {found_file_counter} already processed files.")


def path_to_tensor_label_map(file_path: Tensor) -> (Tensor, Tensor):
    label: Tensor = tf.py_function(func=extract_label, inp=[file_path], Tout=tf.string)

    label: Tensor = tf.reshape(label, [])

    file_tensor: Tensor = tf.io.read_file(file_path)
    audio_tensor, _ = tf.audio.decode_wav(file_tensor, desired_channels=1, desired_samples=16000)
    audio_tensor: Tensor = tf.squeeze(audio_tensor, axis=-1)

    return audio_tensor, label




def stft(waveform_par: Tensor) -> Tensor:

    spectrogram_var = tf.signal.stft(waveform_par, frame_length=256, frame_step=64, fft_length=256)
    spectrogram_var: Tensor = tf.abs(spectrogram_var)
    return spectrogram_var



def waveforms_to_spectrogram(waveform_par: Tensor, label_par: Tensor) -> (Tensor, Tensor):
    spectrogram: Tensor = stft(waveform_par)
    spectrogram_reshaped: Tensor = tf.reshape(spectrogram, (129, 247))
    spectrogram_expanded: Tensor = tf.expand_dims(spectrogram_reshaped, axis=0)
    label_ind: Tensor = tf.math.argmax(label_par == CATEGORIES)

    return spectrogram_expanded, label_ind


def convert_waveform_to_spectrogram(dataset_waveform: DatasetV2) -> DatasetV2:
    return dataset_waveform.map(waveforms_to_spectrogram)
#%%
## process_unprocessed_unknown_data(PATH_FOR_UNPROCESSED_UNKNOWN_DATASET, PATH_FOR_PROCESSED_UNKNOWN_DATASET)
#%% md
# ### Path collection, shuffle and splitting data:
# 
# 
#%%

YES_LABEL_PATHS: Final[List[str]] = tf.io.gfile.glob(PATH_FOR_YES_DATASET + "/*")
NO_LABEL_PATHS: Final[List[str]] = tf.io.gfile.glob(PATH_FOR_NO_DATASET + "/*")
UNKNOWN_LABEL_PATHS: Final[List[str]] = tf.io.gfile.glob(PATH_FOR_PROCESSED_UNKNOWN_DATASET + "/*")



YES_LABEL_PATHS_SHUFFLED: Final[List[str]] = tf.random.shuffle(YES_LABEL_PATHS)
NO_LABEL_PATHS_SHUFFLED: Final[List[str]] = tf.random.shuffle(NO_LABEL_PATHS)
UNKNOWN_LABEL_PATHS_SHUFFLED: Final[List[str]] = tf.random.shuffle(UNKNOWN_LABEL_PATHS)

YES_LABEL_DATASET_PIPELINE: DatasetV2 = tf.data.Dataset.from_tensor_slices(YES_LABEL_PATHS_SHUFFLED)
NO_LABEL_DATASET_PIPELINE: DatasetV2 = tf.data.Dataset.from_tensor_slices(NO_LABEL_PATHS_SHUFFLED)
UNKNOWN_LABEL_DATASET_PIPELINE: DatasetV2 = tf.data.Dataset.from_tensor_slices(UNKNOWN_LABEL_PATHS_SHUFFLED)

YES_LABELED_DATASET_LABEL_MAPPED: DatasetV2 = YES_LABEL_DATASET_PIPELINE.map(path_to_tensor_label_map)
NO_LABELED_DATASET_LABEL_MAPPED: DatasetV2 = NO_LABEL_DATASET_PIPELINE.map(path_to_tensor_label_map)
UNKNOWN_LABELED_DATASET_LABEL_MAPPED: DatasetV2 = UNKNOWN_LABEL_DATASET_PIPELINE.map(path_to_tensor_label_map)

TRAIN_RATIO: Final[float] = 0.6
TEST_RATIO: Final[float] = 0.2
VALIDATION_RATIO: Final[float] = 0.2
SAMPLE_SIZE_FOR_EACH_CLASS: Final[int] = 2000

WHOLE_SAMPLE_SIZE: Final[int] = len(
    CATEGORIES) * SAMPLE_SIZE_FOR_EACH_CLASS


TRAIN_SIZE: Final[int] = int(TRAIN_RATIO * WHOLE_SAMPLE_SIZE)
TEST_SIZE: Final[int] = int(TEST_RATIO * WHOLE_SAMPLE_SIZE)
VALIDATION_SIZE: Final[int] = int(VALIDATION_RATIO * WHOLE_SAMPLE_SIZE)

# ✅ Use take() instead of slicing [:]
TRAIN_DATASET_FOR_YES_LABEL_IN_WAVEFORM: DatasetV2 = YES_LABELED_DATASET_LABEL_MAPPED.take(TRAIN_SIZE)
TRAIN_DATASET_FOR_NO_LABEL_IN_WAVEFORM: DatasetV2 = NO_LABELED_DATASET_LABEL_MAPPED.take(TRAIN_SIZE)
TRAIN_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM: DatasetV2 = UNKNOWN_LABELED_DATASET_LABEL_MAPPED.take(TRAIN_SIZE)

# ✅ Use skip() to get the remaining part
REMAINING_YES = YES_LABELED_DATASET_LABEL_MAPPED.skip(TRAIN_SIZE)
REMAINING_NO = NO_LABELED_DATASET_LABEL_MAPPED.skip(TRAIN_SIZE)
REMAINING_UNKNOWN = UNKNOWN_LABELED_DATASET_LABEL_MAPPED.skip(TRAIN_SIZE)

# ✅ Use take() again to extract test set
TEST_DATASET_FOR_YES_LABEL_IN_WAVEFORM: DatasetV2 = REMAINING_YES.take(TEST_SIZE)
TEST_DATASET_FOR_NO_LABEL_IN_WAVEFORM: DatasetV2 = REMAINING_NO.take(TEST_SIZE)
TEST_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM: DatasetV2 = REMAINING_UNKNOWN.take(TEST_SIZE)

# ✅ Skip test size to get remaining for validation
REMAINING_YES = REMAINING_YES.skip(TEST_SIZE)
REMAINING_NO = REMAINING_NO.skip(TEST_SIZE)
REMAINING_UNKNOWN = REMAINING_UNKNOWN.skip(TEST_SIZE)

# ✅ Final validation dataset
VALIDATION_DATASET_FOR_YES_LABEL_IN_WAVEFORM: DatasetV2 = REMAINING_YES.take(VALIDATION_SIZE)
VALIDATION_DATASET_FOR_NO_LABEL_IN_WAVEFORM: DatasetV2 = REMAINING_NO.take(VALIDATION_SIZE)
VALIDATION_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM: DatasetV2 = REMAINING_UNKNOWN.take(VALIDATION_SIZE)

TRAIN_DATASET_FOR_YES_LABEL_IN_SPEC : DatasetV2= convert_waveform_to_spectrogram(TRAIN_DATASET_FOR_YES_LABEL_IN_WAVEFORM)
TRAIN_DATASET_FOR_NO_LABEL_IN_SPEC: DatasetV2 = convert_waveform_to_spectrogram(TRAIN_DATASET_FOR_NO_LABEL_IN_WAVEFORM)
TRAIN_DATASET_FOR_UNKNOWN_LABEL_IN_SPEC : DatasetV2 = convert_waveform_to_spectrogram(TRAIN_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM)

TRAIN_DATASET: DatasetV2 =TRAIN_DATASET_FOR_YES_LABEL_IN_SPEC.concatenate(TRAIN_DATASET_FOR_NO_LABEL_IN_SPEC).concatenate(TRAIN_DATASET_FOR_UNKNOWN_LABEL_IN_SPEC)

TEST_DATASET_FOR_YES_LABEL_IN_SPEC: DatasetV2 = convert_waveform_to_spectrogram(TEST_DATASET_FOR_YES_LABEL_IN_WAVEFORM)
TEST_DATASET_FOR_NO_LABEL_IN_SPEC: DatasetV2 = convert_waveform_to_spectrogram(TEST_DATASET_FOR_NO_LABEL_IN_WAVEFORM)
TEST_DATASET_FOR_UNKNOWN_LABEL_IN_SPEC: DatasetV2 = convert_waveform_to_spectrogram(TEST_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM)

TEST_DATASET: DatasetV2 = TEST_DATASET_FOR_YES_LABEL_IN_SPEC.concatenate(TEST_DATASET_FOR_NO_LABEL_IN_SPEC).concatenate(TEST_DATASET_FOR_UNKNOWN_LABEL_IN_SPEC)

VALIDATION_DATASET_FOR_YES_LABEL_IN_SPEC: DatasetV2 = convert_waveform_to_spectrogram(VALIDATION_DATASET_FOR_YES_LABEL_IN_WAVEFORM)
VALIDATION_DATASET_FOR_NO_LABEL_IN_SPEC: DatasetV2 = convert_waveform_to_spectrogram(VALIDATION_DATASET_FOR_NO_LABEL_IN_WAVEFORM)
VALIDATION_DATASET_FOR_UNKNOWN_LABEL_IN_SPEC: DatasetV2 = convert_waveform_to_spectrogram(VALIDATION_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM)

VALIDATION_DATASET: DatasetV2 = VALIDATION_DATASET_FOR_YES_LABEL_IN_SPEC.concatenate(VALIDATION_DATASET_FOR_NO_LABEL_IN_SPEC).concatenate(VALIDATION_DATASET_FOR_UNKNOWN_LABEL_IN_SPEC)


### We do not need them anymore.
del WHOLE_SAMPLE_SIZE, TRAIN_SIZE, TEST_SIZE, VALIDATION_SIZE

del YES_LABEL_PATHS, NO_LABEL_PATHS, UNKNOWN_LABEL_PATHS
del YES_LABEL_PATHS_SHUFFLED, NO_LABEL_PATHS_SHUFFLED, UNKNOWN_LABEL_PATHS_SHUFFLED

del YES_LABEL_DATASET_PIPELINE, NO_LABEL_DATASET_PIPELINE, UNKNOWN_LABEL_DATASET_PIPELINE
del YES_LABELED_DATASET_LABEL_MAPPED, NO_LABELED_DATASET_LABEL_MAPPED, UNKNOWN_LABELED_DATASET_LABEL_MAPPED

del TRAIN_DATASET_FOR_YES_LABEL_IN_WAVEFORM, TRAIN_DATASET_FOR_NO_LABEL_IN_WAVEFORM, TRAIN_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM
del TEST_DATASET_FOR_YES_LABEL_IN_WAVEFORM, TEST_DATASET_FOR_NO_LABEL_IN_WAVEFORM, TEST_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM
del VALIDATION_DATASET_FOR_YES_LABEL_IN_WAVEFORM, VALIDATION_DATASET_FOR_NO_LABEL_IN_WAVEFORM, VALIDATION_DATASET_FOR_UNKNOWN_LABEL_IN_WAVEFORM
gc.collect()

SHAPE_OF_FEATURES  = TRAIN_DATASET.element_spec[0].shape


#%% md
# ### Batching:
#%%
BATCH_SIZE = 1

TRAIN_DATASET_BATCH = TRAIN_DATASET.map(
    lambda spec, label: tf.squeeze(tf.expand_dims(spec, axis=0), axis=1),
).batch(BATCH_SIZE)

VALIDATION_DATASET_BATCH = VALIDATION_DATASET.map(
    lambda spec, label: tf.squeeze(tf.expand_dims(spec, axis=0), axis=1)
).batch(BATCH_SIZE)

for spec in TRAIN_DATASET_BATCH.take(1):
    print(spec.shape)
#%% md
# ### Inspecting the data
#%%
"""
import matplotlib.pyplot as plt

waveforms = []
labels = []

for audio, label in TRAIN_DATASET_FOR_YES_LABEL_IN_WAVEFORM.as_numpy_iterator():
    waveforms.append(audio)
    labels.append(label.decode())

# Plot waveforms
fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(15, 20))
axes = axes.flatten()

for i, (waveform, label) in enumerate(zip(waveforms, labels)):
    if i >= len(axes):
        break

    axes[i].plot(waveform)
    axes[i].set_title(f"Sample {i} - Label: {label}")  # Show label
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Amplitude")

plt.tight_layout()
plt.show()

"""
#%% md
# ### Building Model:
#%%
norm_layer = Normalization()

norm_layer.adapt(TRAIN_DATASET_BATCH)

model = models.Sequential()

model.add(layers.Input(shape=SHAPE_OF_FEATURES))
model.add(Resizing(32, 32))
model.add(norm_layer)
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(CATEGORIES)))
model.summary()

print("Model Input Shape:", model.input_shape)
print("Model Output Shape:", model.output_shape)

#%% md
# ### Compilation:
#%%
model.compile(optimizer='adam',
                loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ["accuracy"])
#%%
history = model.fit(TRAIN_DATASET_BATCH, validation_data=VALIDATION_DATASET_BATCH, epochs=25)
#%% md
# ### Testing:
#%%
test_spec_arr = []
test_label_arr = []

for spectrogram_val, label in TEST_DATASET.as_numpy_iterator():
    test_spec_arr.append(spectrogram_val.numpy())
    test_label_arr.append(label.numpy())

test_spec_arr = np.array(test_spec_arr)
test_label_arr = np.array(test_label_arr)



ith_sample: int = 0
count_of_samples: int = len(test_spec_arr)

model_prediction = model.predict(test_spec_arr)[ith_sample]
actual_label = test_label_arr[ith_sample]
probabilities = softmax(model_prediction)


print("Categories:", CATEGORIES)
print(f"Model Prediction {ith_sample + 1}'th of {count_of_samples}",model_prediction)
print("Probabilities:", probabilities)
print("Actual Label:", CATEGORIES[actual_label])
#%% md
# ### Model Saving:
#%%
model.save("models/model_0.keras");