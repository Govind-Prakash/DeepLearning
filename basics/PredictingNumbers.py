import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

#load the MNIST dataset
(train_images, train_labels),(test_images, test_labels)= tf.keras.datasets.mnist.load_data()

#Normalise the Pixel values to be between 0 and 1
train_images, test_images= train_images/255.0, test_images/255.0

# Create the Model
model = models.Sequential([layers.Flatten(input_shape=(28,28)), #Flattens the input imae
                           layers.Dense(128,activation='relu'), #Hidden layer with 128neurons and ReLu activation
                           layers.Dense(10,activation='softmax')#Output layer with 10 neurons (one for each )
                        ])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images,train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc= model.evaluate(test_images,test_labels)

print(f"Test accuracy: {test_acc}")

#Make predictions on the test dataset
predictions= model.predict(test_images)

#Show the first image and its predicted label
plt.imshow(test_images[0],
           cmap=plt.cm.binary)
plt.title(f"predicted Label: {np.argmax(predictions[0])}")

plt.show()
