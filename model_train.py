import pickle
import numpy as np
import tensorflow as tf
import csv
import cv2
import os
import keras
import sklearn
import random
from sklearn.utils import shuffle
import itertools
import pandas as pd
import zipfile
from pathlib import Path

def load_data_df():

	dir_path = os.getcwd()
	data_dir_zip = dir_path + '/data.zip'
	file_exists_path = dir_path + '/data/driving_log.csv'
	my_file = Path(file_exists_path)

	if not my_file.is_file():
		zfile = zipfile.ZipFile(data_dir_zip)
		zfile.extractall(dir_path + '/')
		print('File Extracted')
	else:
		print('Data file already exists.')

	samples = pd.read_csv(dir_path + '/data/driving_log.csv')

	vald_num = round(len(samples) * 0.8)

	samples_X = samples[0:vald_num]
	samples_Y = samples[vald_num:]

	return samples_X, samples_Y

def get_augmented_row(l):

	dir_path = os.getcwd()
	correction = 0.2
	center_angle = float(l[3])
	camera = np.random.choice(['center', 'left', 'right'])
	
	if camera == "left":
		path_name = dir_path + '/data/IMG/'+l[1].split('/')[-1]
		image = cv2.imread(path_name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		angle = center_angle + correction
	elif camera == "right":
		path_name = dir_path + '/data/IMG/'+l[2].split('/')[-1]
		image = cv2.imread(path_name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		angle = center_angle - correction
	elif camera == "center":
		path_name = dir_path + '/data/IMG/'+l[0].split('/')[-1]
		image = cv2.imread(path_name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		angle = center_angle

	# decide whether to horizontally flip the image:
	flip_prob = np.random.random()
	if flip_prob > 0.75:
		# flip the image and reverse the steering angle
		angle = -1*angle
		image = cv2.flip(image, 1)

	return image, angle


def generator(csv_data, batch_size = 32):

	N = csv_data.shape[0]

	batches_per_epoch = N

	i = 0

	   # Generator should yield images indefinitely

	while(True):

		start = i*batch_size

		end = start+batch_size - 1
		# initialize your batch data

		X_batch = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)

		y_batch = np.zeros((batch_size,), dtype=np.float32)

		j = 0

		# slice a `batch_size` sized chunk from the csv_data
    	# and generate augmented data for each row in the chunk on the fly


		for index, row in csv_data.loc[start:end].iterrows():

			X_batch[j], y_batch[j] = get_augmented_row(row)

			# perform image augmentation for each row
			j += 1

		i += 1

		if i == batches_per_epoch - 1:

			# reset the index so that we can cycle over the csv_data again

			i = 0

		yield X_batch, y_batch 



def train_network_sim():

	# Initial Setup for Keras
	from keras.models import Sequential
	from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
	from keras.layers import Lambda,Cropping2D
	from keras.layers.convolutional import Convolution2D
	from keras.layers.pooling import MaxPooling2D
	import gc
	from keras import backend as K
	import tensorflow as tf
	from keras.models import Model
	import matplotlib.pyplot as plt
	from datetime import datetime 

	samples_X, samples_Y = load_data_df()

	train_generator = generator(samples_X, batch_size=32)
	validation_generator = generator(samples_Y, batch_size=32)
	
	size_of_batch = 32
	num_samples = len(samples_X)
	print('num_samples: ',num_samples)
	Xlen = num_samples
	Remainder = Xlen % size_of_batch
	batch_per_epoch = Xlen
	if(Remainder != 0):
		batch_per_epoch = Xlen - Remainder
	print('batch_per_epoch: ',batch_per_epoch)

	samples_generated = 0
	steering_angles = None

	samples_generated = int(batch_per_epoch/size_of_batch)
	print('samples_generated: ',samples_generated)


	for i in range(samples_generated):
		x_batch, y_batch = next(train_generator)
		#print ('y_batch: ',y_batch)
		if steering_angles is not None:
			steering_angles = np.concatenate([steering_angles, y_batch])
		else:
			steering_angles = y_batch


	import matplotlib.pyplot as plt
	fig = plt.figure()
	n = plt.hist([steering_angles],bins=30)
	plt.show()

	print("Xlen:{}".format(Xlen))
	print("Number of batch per epoch = {}".format(batch_per_epoch))

	# TODO: Build the Final Test Neural Network in Keras Here
	print(keras.__version__)
	model = Sequential()
	model.add(Lambda(lambda x: ((x / 255.0) - 0.5),input_shape=(160, 320, 3),output_shape=(160, 320, 3)))
	print(model.layers[-1].output_shape)
	#model.add(Cropping2D(cropping=((96, 0), (1, 255)), input_shape=(160, 320, 3)))
	#model.add(Cropping2D(cropping=((88, 1), (1, 119)), input_shape=(160, 320, 3)))
	print(model.layers[-1].output_shape)
	#model.add(MaxPooling2D((2, 2)))
	#print(model.layers[-1].output_shape)
	model.add(Convolution2D(16,8,8))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((4, 4)))
	print('Conv1: ', model.layers[-1].output_shape)
	model.add(Convolution2D(32,5,5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2)))
	print('Conv2: ', model.layers[-1].output_shape)
	model.add(Convolution2D(64,5,5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2)))
	print('Conv3: ', model.layers[-1].output_shape)
	model.add(Flatten())
	print('Flatten: ', model.layers[-1].output_shape)
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Dense(1))
	print('output: ',model.layers[-1].output_shape)
	model.compile(loss='mse',optimizer='adam')
	print("Start Time: ", str(datetime.now()))
	history_object = model.fit_generator(train_generator, samples_per_epoch = batch_per_epoch,validation_data = validation_generator, nb_val_samples = len(samples_Y), nb_epoch = 1, verbose = 1)
	model.save('model.h5')
	print("End Time: ", str(datetime.now()))
	#plot_model(model, to_file='model.png')
	### print the keys contained in the history object
	#print(history_object.history.keys())
	### plot the training and validation loss for each epoch
	#plt.plot(history_object.history['loss'])
	#plt.plot(history_object.history['val_loss'])
	#plt.title('model mean squared error loss')
	#plt.ylabel('mean squared error loss')
	#plt.xlabel('epoch')
	#plt.legend(['training set', 'validation set'], loc='upper right')
	#plt.show()
	gc.collect()
	K.clear_session()

train_network_sim()
