import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from matplotlib import pyplot as plt
from plot_keras_history import show_history, plot_history
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD


def preprocess(img, label):
    return tf.image.resize(img, [HEIGHT, WIDTH]) / 255, label


HEIGHT = 200
WIDTH = 200
split = ['train[:70%]', 'train[70%:]']

trainDataset, testDataset = tfds.load(
    name='cats_vs_dogs', split=split, as_supervised=True)

trainDataset = trainDataset.map(preprocess).batch(32)
testDataset = testDataset.map(preprocess).batch(32)

# model = keras.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
#                         input_shape=(HEIGHT, WIDTH, 3)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dense(2, activation='softmax'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

model = keras.Sequential([
    VGG16(include_top=False, input_shape=(200, 200, 3)),
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer='he_uniform'),
    # SGD(lr=0.001, momentum=0.9),
    keras.layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Rodando o modelo utilizando a GPU
trainHistory = model.fit(
    trainDataset, epochs=10, validation_data=testDataset, use_multiprocessing=True, workers=8)

show_history(trainHistory)

(loss, accuracy) = model.evaluate(testDataset)
print(loss)
print(accuracy)

plt.plot(trainHistory.history['accuracy'])
plt.plot(trainHistory.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'])
plt.grid()
plt.show()
