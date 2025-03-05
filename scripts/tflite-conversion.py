#%%
import tensorflow as tf;
import os;
#%%
model = tf.keras.models.load_model("./models/model_0.keras");

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("models/model_0.tflite", "wb") as f:
    f.write(tflite_model)
#%% md
# Comparing Storage Sizes:
#%%
model_path = "./models/model_0.keras"
tflite_model_path = "./models/model_0.tflite"

model_size = os.path.getsize(model_path)
tflite_model_size = os.path.getsize(tflite_model_path)

def convert_bytes(num: float):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

print("Model size: %s" % convert_bytes(model_size))
print("TFLite model size: %s" % convert_bytes(tflite_model_size))