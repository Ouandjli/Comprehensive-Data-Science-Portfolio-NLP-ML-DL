import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Charger les fichiers .mat
train_file_path = 'train_32x32.mat'
test_file_path = 'test_32x32.mat'

train_data = scipy.io.loadmat(train_file_path)
test_data = scipy.io.loadma
t(test_file_path)

# Afficher les clés des données chargées pour vérifier
print("Clés des données d'entraînement:", train_data.keys())
print("Clés des données de test:", test_data.keys())

# Extraire les données d'entraînement
x_train_mat = train_data['X']
y_train_mat = train_data['y']

# Extraire les données de test
x_test_mat = test_data['X']
y_test_mat = test_data['y']

# Vérifier les formes des données pour s'assurer qu'elles sont correctement chargées
print("Forme de x_train_mat:", x_train_mat.shape)
print("Forme de y_train_mat:", y_train_mat.shape)
print("Forme de x_test_mat:", x_test_mat.shape)
print("Forme de y_test_mat:", y_test_mat.shape)

# Transposer les dimensions de x_train_mat et x_test_mat pour correspondre au format attendu par Keras (n, 32, 32, 3)
x_train_mat = np.transpose(x_train_mat, (3, 0, 1, 2))
x_test_mat = np.transpose(x_test_mat, (3, 0, 1, 2))

print(f'x_train_mat shape: {x_train_mat.shape}')  # Devrait être (73257, 32, 32, 3)
print(f'x_test_mat shape: {x_test_mat.shape}')  # Devrait être (26032, 32, 32, 3)

# Normalisation des données
x_train_mat = x_train_mat.astype('float32') / 255.0
x_test_mat = x_test_mat.astype('float32') / 255.0

# Reshape y_train_mat et y_test_mat pour correspondre au format attendu (n,)
y_train_mat = y_train_mat.flatten()
y_test_mat = y_test_mat.flatten()

# Convertir les labels en vecteurs one-hot
num_classes = 10
y_train_mat = to_categorical(y_train_mat - 1, num_classes)
y_test_mat = to_categorical(y_test_mat - 1, num_classes)

print(f'y_train_mat shape: {y_train_mat.shape}')
print(f'y_test_mat shape: {y_test_mat.shape}')

#Afficher quelques exemples pour vérifier les données
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train_mat[i])
    plt.title(f"Label: {np.argmax(y_train_mat[i])}")
    plt.axis('off')
plt.show()

# Définir le modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train_mat.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()

# Entraîner le modèle
epochs = 10
history = model.fit(x_train_mat, y_train_mat, batch_size=128, epochs=epochs, validation_data=(x_test_mat, y_test_mat), shuffle=True)

# Évaluer le modèle
score = model.evaluate(x_test_mat, y_test_mat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#Visualiser les résultats
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()



