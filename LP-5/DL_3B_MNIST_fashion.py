# Multi- classification
# Import libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
#Load Dataset
print("Loading Fashion MNIST data...")
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#Checking dataset shape
print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)
#Display first image
plt.figure(figsize=(3,3))
plt.imshow(x_train[0]) #, cmap='gray'
plt.title(class_names[y_train[0]])
plt.colorbar()
plt.show()
# --- Visualize the Learning Curve ---
print("\ Generating Learning Visuals...")
plt.plot(history.history['accuracy'], label='Accuracy (Learning)')
plt.plot(history.history['val_accuracy'], label='Validation (Testing itself)')
plt.title('Model Learning Progress')
plt.xlabel('Epoch (Study Round)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#Normalization converting 0-255 to 0-1
print(f"Normalizing pixels. Max value before: {x_train.max()}")
x_train = x_train / 255.0
x_test = x_test / 255.0
print(f"Max value after: {x_train.max()} (Data is now between 0 and 1)")
# Building Neural Network
model = keras.Sequential([ 
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)), 
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(128, (3,3), activation='relu'), 
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation = 'softmax')
    ])
model.summary()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("Starting Training..")
# We save the training progress into 'history'
history = model.fit(x_train, y_train, epochs=12, validation_split=0.1, verbose=1)

print("Testing on unseen images...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Accuracy: {test_acc*100:.2f}%")

# Validation
img_index = 40
img = x_test[img_index]
prediction = model.predict(np.expand_dims(img,0))

for i, score in enumerate(prediction[0]):
    print(f"{class_names[i]}: {score*100:.2f}%")

plt.imshow(img)
plt.title(f"Guessed:{class_names[np.argmax(prediction)]}")
plt.show()

# Printing all images
plt.figure(figsize = (9,7))

for i in range(0, 12):
    plt.subplot(3, 4, i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis('off')
   
plt.show()
