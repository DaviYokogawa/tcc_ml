# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history


(train_images, train_labels), (test_images,
                               test_labels) = tfds.as_numpy(tfds.load('fashion_mnist', split=['train', 'test'],
                                                                      batch_size=-1, as_supervised=True))


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(
    f"Base de treinamento:{train_images.shape[0]} imagens com {train_images.shape[1]} x {train_images.shape[2]} pixels")

print(
    f"Base de teste:{test_images.shape[0]} imagens com {test_images.shape[1]} x {test_images.shape[2]} pixels")


# Pré-processando os dados

# Podemos notar que os pixels estão entre 0 e 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# Vamos padronizar os valores para o intervalo [0,1]
# Vamos confirmar o range de pixel na nossa base
train_images.min()
train_images.max()

test_images.min()
test_images.max()


print(train_images[0])
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images[0])


print('Verificando que esta tudo ook com a base')
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Criando o modelo

model = keras.Sequential([
    # Transforma a imagem na dimensão 28x28px
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.BatchNormalization(),
    # Essas são camadas neurais densely connected, ou fully connected
    keras.layers.Dense(128, activation='relu'),  # 128 nós
    # Retorna uma probabilidade de 10 classes, a soma da 1
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
keras.utils.plot_model(model, show_shapes=True)


history = model.fit(train_images, train_labels, validation_data=(
    test_images, test_labels), epochs=10)

show_history(history)
