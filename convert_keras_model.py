import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the pre-trained model
model = keras.models.load_model('my_model_4.h5')

# Print the size of the model
print('Size of the original model:', os.path.getsize('my_model_4.h5'))

# Compress the model using TensorFlow's built-in compression API
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Write the compressed model to disk
with open('my_model_4_compressed.tflite', 'wb') as f:
    f.write(tflite_model)

# Print the size of the compressed model
#print('Size of the compressed model:', os.path.getsize('my_model_4_compressed.tflite'))