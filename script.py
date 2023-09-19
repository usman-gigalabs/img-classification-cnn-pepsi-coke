import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

import matplotlib.pyplot as plt

# Define the data directory and batch size
data_dir = "PepsiandCocaColaImages/train"
batch_size = 32

# Define image dimensions
img_height, img_width = 270, 230

# Count the number of subfolders in the dataset folder, each subfolder represents a class
num_classes = len(os.listdir(data_dir))

print(num_classes)

# Create ImageDataGenerators for training and testing with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2,  # Split the data into training and validation sets
    rotation_range=15,  # Randomly rotate images by up to 15 degrees
    width_shift_range=0.2,  # Randomly shift the width by up to 20%
    height_shift_range=0.2,  # Randomly shift the height by up to 20%
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Randomly zoom in by up to 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in new pixels with the nearest value
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),  # Resize images to a common size
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    subset='training'  # Specify this is the training set
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify this is the validation set
)

# Build a Convolutional Neural Network (CNN)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    # BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    # BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    # BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    # BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


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
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(num_epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

