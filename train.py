import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the data directory and batch size
data_dir = "animal-data/raw-img-small"
batch_size = 32

# Count the number of subfolders in the dataset folder, each subfolder represents a class
num_classes = len(os.listdir(data_dir))

print(num_classes)

# Create ImageDataGenerators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Split the data into training and validation sets
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Resize images to a common size
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    subset='training'  # Specify this is the training set
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify this is the validation set
)

# Load the saved model
model = load_model("model.keras")

# Compile the Model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the Model
num_epochs = 10  # You can increase this for better accuracy

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=num_epochs,
)

# Save the trained model
model.save("model.keras")

test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()