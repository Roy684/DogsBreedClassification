import tensorflow as tf
import numpy as np

IMG_SIZE = 224

# load class names (same order as training)
unique_breeds = np.load("model/class_names.npy", allow_pickle=True)

def preprocess_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.expand_dims(image, axis=0)
    return image

def get_prediction(model, image_bytes):
    image = preprocess_image(image_bytes)
    preds = model.predict(image)
    index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    label = unique_breeds[index]
    return label, confidence