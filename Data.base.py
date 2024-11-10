import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# Charger le fichier .mat CNN réseau neuronal convultif qu'on applique souvent pour coder et lire les images
# Charger le fichier .mat
file_path = 'train_32x32.mat'
data = scipy.io.loadmat(file_path)
print(data.keys())

# Extraire les données
x_train = data['X']
y_train = data['y']
x_train = x_train.astype(np.uint8)  # Convertir en octets

# Transposer les dimensions de x_train pour correspondre au format attendu par Keras (n, 32, 32, 3)
x_train = np.transpose(x_train, (3, 0, 1, 2))

print(f'x_train shape: {x_train.shape}')  # (73257, 32, 32, 3)
print(f'y_train shape: {y_train.shape}')  # (73257, 1)

# Normaliser les images
x_train = x_train / 255.0

# One-hot encode les labels
y_train = to_categorical(y_train - 1, 10)  # les labels vont de 1 à 10, donc nous soustrayons 1 pour les amener dans la plage 0-9

# Séparer les données en ensembles d'entraînement et de validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Charger le dataset Fashion MNIST
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()

# Prétraiter les images Fashion MNIST
def preprocess(imgs):
    # Normaliser les images
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)  # Ajouter une dimension pour les canaux
    imgs = imgs / 255.0  # Normaliser les images
    return imgs

x_train_fashion = preprocess(x_train_fashion)
x_test_fashion = preprocess(x_test_fashion)

print(f'x_train_fashion shape: {x_train_fashion.shape}')  # (60000, 32, 32, 1)
print(f'y_train_fashion shape: {y_train_fashion.shape}')  # (60000,)

# One-hot encode les labels Fashion MNIST
y_train_fashion = to_categorical(y_train_fashion, 10)
x_test_mat: object = to_categorical(y_test_fashion, 10)

print(f'y_train_fashion one-hot shape: {y_train_fashion.shape}')  # (60000, 10)
print(f'y_test_fashion one-hot shape: {y_test_fashion.shape}')  # (10000, 10)

# Configuration des hyperparamètres
image_size = 32
channels = 1
batch_size = 100
buffer_size = 1000
embedding_dim = 2
epochs = 3

# Exemple d'affichage d'une image du jeu de données SVHN
plt.imshow(x_train[0])
plt.show()

# Exemple d'affichage d'une image du jeu de données Fashion MNIST
plt.imshow(x_train_fashion[0].reshape(32, 32), cmap='gray')
plt.show()
# Définir les paramètres
image_size = 32
channels = 1
embedding_dim = 2

# Entrée de l'encodeur
encoder_input = layers.Input(shape=(image_size, image_size, channels), name="encoder_input")

# Couches de convolution
x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(encoder_input)
x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(x)

# Stocker la forme avant l'aplatissement
shape_before_flattening = tf.keras.backend.int_shape(x)

# Aplatissement
x = layers.Flatten()(x)

# Couche dense
encoder_output = layers.Dense(embedding_dim, name="encoder_output")(x)

# Créer le modèle d'encodeur
encoder = models.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()
# Définir les paramètres
embedding_dim = 2
image_size = 32
channels = 1

# Définir l'entrée du décodeur
decoder_input = layers.Input(shape=(embedding_dim,), name="decoder_input")

# Couche dense pour connecter l'encodeur au décodeur
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)

# Reshape pour correspondre à la sortie de la dernière couche de convolution de l'encodeur
x = layers.Reshape(shape_before_flattening[1:])(x)

# Couches de convolution transpose (déconvolution)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)

# Couche de sortie pour reconstruire l'image
decoder_output = layers.Conv2DTranspose(channels, (3, 3), strides=1, activation='sigmoid', padding='same', name='decoder_output')(x)
#softmax u lieu de sigmoid, pour exécuter le CNN sur les jeux de données Train et test 32
# Créer le modèle de décodeur
decoder = models.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

# Créer l'autoencodeur en combinant l'encodeur et le décodeur
autoencoder_input = encoder_input
autoencoder_output = decoder(encoder_output)
autoencoder = models.Model(autoencoder_input, autoencoder_output, name="autoencoder")
autoencoder.summary()

# Compiler l'autoencodeur
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])# loss = crossentropy pour la classification

# Évaluation de l'autoencodeur sur les données de test
loss, accuracy = autoencoder.evaluate(x_test_fashion, x_test_fashion)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

