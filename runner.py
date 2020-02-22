# This Spatial Pyramid Pooling Layer is for keras 2.2.4+ running over TensorFlow 2.0
import os
import numpy as np
import tensorflow
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Activation, MaxPooling2D, Dense, Dropout

from SpatialPyramidPooling import SpatialPyramidPooling

# Minimizes Tensorflow Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 64
NUM_CHANNELS = 1
NUM_CLASSES = 10

def makeModel():
	model = Sequential()

	# MODEL 1
	# uses tensorflow ordering. Note that we leave the image size as None to allow multiple image sizes
	model.add(Convolution2D(32, 3, 3, padding='same', input_shape=(None, None, NUM_CHANNELS)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Convolution2D(64, 3, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(SpatialPyramidPooling([1, 2, 4]))
	model.add(Dense(NUM_CLASSES))
	model.add(Activation('softmax'))

	# MODEL 2
	# uses tensorflow ordering. Note that we leave the image size as None to allow multiple image sizes
	# model.add(Convolution2D(96, 11, 11, padding='same', input_shape=(None, None, NUM_CHANNELS), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	# model.add(Convolution2D(32, 3, 3, padding='same', activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	# model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	# model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	# model.add(SpatialPyramidPooling([1, 2, 4]))
	# model.add(Dense(4096, activation='relu', name='dense_1'))
	# model.add(Dropout(0.5))
	# model.add(Dense(4096, activation='relu', name='dense_2'))
	# model.add(Dropout(0.5))
	# model.add(Dense(NUM_CLASSES, name='dense_3'))
	# model.add(Activation('softmax'))

	return model

def main():
	model=makeModel()
	model.summary()

	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5) / 127.5		# Normalize the images to [-1, 1]

	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
	test_images = (test_images - 127.5) / 127.5			# Normalize the images to [-1, 1]

	adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics = ["accuracy"])
	model.fit(train_images, train_labels)

	# results = model.evaluate(test_images, test_labels, batch_size=128)
	# print('test loss, test acc:', results)

	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ["accuracy"])
	# train on 64x64x3 random images
	model.fit(np.random.rand(BATCH_SIZE, 64, 64, NUM_CHANNELS), np.zeros((BATCH_SIZE, NUM_CLASSES)))
	# train on 32x32x3 random images
	model.fit(np.random.rand(BATCH_SIZE, 32, 32, NUM_CHANNELS), np.zeros((BATCH_SIZE, NUM_CLASSES)))

if __name__ == '__main__':
	main()