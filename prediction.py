import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained model
model = load_model('model.keras')

# Define image dimensions
img_height, img_width = 270, 230

# Load and preprocess a test image
# img_path = 'PepsiandCocaColaImages/test/pepsi/20.jpg'
img_path = '/Users/gigalabs/Downloads/free-photo-of-pepsi-can-in-dew.jpeg'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values to [0, 1]

# Make predictions
predictions = model.predict(img_array)
print(predictions)

# Get the class label with the highest probability
predicted_class = np.argmax(predictions, axis=1)

# Class names and their transalations
class_mapping = {0: "Coca Cola", 1: "Pepsi"}

predicted_class_label = class_mapping.get(predicted_class[0], 'Unknown')
# Print the predicted class label
print('Predicted Class: ',predicted_class_label)
