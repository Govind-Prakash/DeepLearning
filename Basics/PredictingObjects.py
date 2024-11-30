import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

#Load CIFAR-10 data set

(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalise pixel values to be between 0 and 1
train_images, test_images= train_images/255.0, test_images/255.0

# Class names in CIFAR-10 
class_names= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Plot some sample images from the dataset
plt.figure(figsize=(10,10))
for i in range (25):
    plt.subplot(5,5, i+1)
    plt.imshow(train_images[i])

plt.title(class_names[train_labels[i][0]])
plt.axis('off')
plt.show()

#Create a conolutional neural Network (CNN)

model= models.Sequential([
    #First convolutional layer
    layers.Conv2D(32, (3,3),activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    #Second Convolutional layer
    layers.Conv2D(64, (3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    #Third Convolution layer
    layers.Conv2D(64, (3,3),activation='relu'),

    #Flatten layer to convert @D matrices into 1D vector
    layers.Flatten(),

    #Fully conneted layer
    layers.Dense(64,activation='relu'),

    #Output layer: 10 Nurons (one for each class)
    layers.Dense(10)
])

#Summary of the model 
model.summary()


# Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


#Train the model
history=model.fit(train_images, train_labels, epochs=10, validation_data=(test_images,test_labels))

#Evaluate the Model
test_loss, test_acc=model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_acc}")

#Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()

# Pot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Make Prediction
predictions=model.predict(test_images)

#Get the predicted label for the first image
predicted_label= np.argmax(predictions[0])

#Display the image and predicted label
plt.imshow(test_images[0])
plt.title(f"Predictd Label: {class_names[predicted_label]}")
plt.axis('off')
plt.show()

