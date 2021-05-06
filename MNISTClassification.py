import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
from numpy import array
from tensorflow import convert_to_tensor
from tensorflow import cast

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print()
print('x train shape: ', x_train.shape)
print('y train shape: ', y_train.shape)
print('x test shape: ', x_test.shape)
print('y test shape: ', y_test.shape)

# Data preprocessing

x_train = cast(x_train, tf.float32)
x_test = cast(x_test, tf.float32)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Shuffle data

x_train = array(x_train)
y_train = array(y_train)
x_test = array(x_test)
y_test = array(y_test)

x_train, y_train = shuffle(x_train, y_train)
x_train = convert_to_tensor(x_train)
y_train = convert_to_tensor(y_train)

x_test, y_test = shuffle(x_test, y_test)
x_test = convert_to_tensor(x_test)
y_test = convert_to_tensor(y_test)

# Model

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Train

# Losses for classification
# categorical_crossentropy: one-hot representation (example: [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
# sparse_categorical_crossentropy: integer representation (example: [2, 3, 1])

print()
print('-'*20, 'TRAIN', '-'*20)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=32)

# Test

print('-'*20, 'TEST', '-'*20)
test_loss, test_acc = model.evaluate(x_test, y_test)
print()
print('test loss: ', test_loss)
print('test accuracy: ', test_acc)
print()