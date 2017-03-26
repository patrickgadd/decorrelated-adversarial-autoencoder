from tensorflow.examples.tutorials.mnist import input_data
import os
import scipy.io as sio
import numpy as np
from random import sample
from random import uniform

def one_hot_to_int(one_hot):
	# the digits are encoded as 0 = idx0, 9 == idx9, all good and straightforward
	for i in range(len(one_hot)):
		if one_hot[i] > 0.1: # greater than zero, really. The >0.1 is float-paranoia
			return i

	print(one_hot)
	assert(False)
	return -1


def extract_few_mnist_labels(cnt_per_class, n_classes, batch_size, mnist):
	# NB: there's a risk of repeated samples because of the way this is implemented
	print('Creating a subset of labelled data for the semi-supervised learning.')
	def one_hot_to_int(one_hot):
		# the digits are encoded as 0 = idx0, 9 == idx9, all good and straightforward
		for i in range(len(one_hot)):
			if one_hot[i] > 0.1: # greater than zero, really. The >0.1 is float-paranoia
				return i

		assert(False)
		return -1

	label_cnts = np.zeros((n_classes))
	labelled_cnt = cnt_per_class * n_classes
	few_annotated_imgs = np.zeros((labelled_cnt, 28*28))
	few_annotated_ys = np.zeros((labelled_cnt, n_classes))
	idx = 0

	test_cnt = 0
	# For training the model on different *random* annotated subsets
	# Not optimal, but better than using the same labels and it's been easy to implement
	skip_prob = 0.999

	am_i_done = False
	while not am_i_done:
		test_cnt += batch_size
		images, y_ = mnist.train.next_batch(batch_size)
		for i in range(y_.shape[0]):

			label_int = one_hot_to_int(y_[i])

			if(label_cnts[label_int] < cnt_per_class):
				# Found a datum which is relevant
				if uniform(0,1) < skip_prob:
					continue

				few_annotated_imgs[idx] = images[i]
				few_annotated_ys[idx] = y_[i]
				idx += 1
				label_cnts[label_int] += 1

			if np.sum(label_cnts) == labelled_cnt:
				am_i_done = True

	print('No. of data points evaluated when creating the labelled subset: {0}'.format(test_cnt))

	return few_annotated_imgs, few_annotated_ys



class Data:
	def __load_svhn(self, n_classes, cnt_per_class):
		res = 32

		test_data = sio.loadmat('../SVHN/test_32x32.mat')
		test_data['X'] = np.transpose(test_data['X'], axes=[3,0,1,2])

		train_data = sio.loadmat('../SVHN/train_32x32.mat')
		train_data['X'] = np.transpose(train_data['X'], axes=[3,0,1,2])

		extra_data = sio.loadmat('../SVHN/extra_32x32.mat')
		extra_data['X'] = np.transpose(extra_data['X'], axes=[3,0,1,2])

		# Zero is originally labelled as 10, convert that to 0
		test_data['y'] = test_data['y'] % 10
		train_data['y'] = train_data['y'] % 10
		extra_data['y'] = extra_data['y'] % 10


		self.X_train = np.concatenate([train_data['X'], extra_data['X']], axis=0)
		self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)

		self.y_train = np.concatenate([train_data['y'], extra_data['y']], axis=0)
		self.y_train = self.y_train.reshape(self.y_train.shape[0])

		few_annotated_imgs, few_annotated_ys = self.__extract_few_labels_svhn(self.X_train, self.y_train, res, cnt_per_class, n_classes)
		self.X_train_few = few_annotated_imgs
		self.y_train_few = few_annotated_ys

		self.X_test = test_data['X']
		self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)

		self.y_test = test_data['y']
		self.y_test = self.y_test.reshape(self.y_test.shape[0])

	def __load_mnist(self, n_classes, cnt_per_class, working_directory, batch_size):
		data_directory = os.path.join(working_directory, "MNIST")
		if not os.path.exists(data_directory):
			os.makedirs(data_directory)

		self.mnist = input_data.read_data_sets(data_directory, one_hot=True)

		# labelled_cnt = cnt_per_class * n_classes # How many labelled data points will we use for the semi-supervised learning process?
		self.X_train_few, self.y_train_few = extract_few_mnist_labels(cnt_per_class, n_classes, batch_size, self.mnist)


	def __init__(self, dataset='MNIST', n_classes=10, cnt_per_class=10, working_directory=None, batch_size=-1):
		self.dataset = dataset
		if dataset == 'SVHN':
			self.__load_svhn(n_classes, cnt_per_class)
		elif dataset == 'MNIST':
			self.__load_mnist(n_classes, cnt_per_class, working_directory, batch_size)

	def __get_random_minibatch_svhn(self, batch_size, n_classes, purpose):
		if purpose == 'train_few':
			Xs = self.X_train_few
			ys = self.y_train_few
		elif purpose == 'train':
			Xs = self.X_train
			ys = self.y_train
		elif purpose == 'test':
			Xs = self.X_test
			ys = self.y_test
		else:
			assert(False)

		rnd_idxs = sample(range(ys.shape[0]), batch_size)
		y_batch = ys[rnd_idxs]
		X_batch = Xs[rnd_idxs, :]

		X_batch = X_batch.astype('float32')
		X_batch = X_batch / np.max(X_batch)

		y_batch = y_batch.astype('int32')

		# One-hot encoding of ys?
		b = np.zeros((batch_size, n_classes))
		b[np.arange(batch_size), y_batch] = 1
		y_batch = b

		return X_batch, y_batch

	def __get_random_minibatch_mnist(self, batch_size, n_classes, purpose):
		if purpose == 'train_few':
			if batch_size <= self.y_train_few.shape[0]:
				rnd_idxs = sample(range(self.y_train_few.shape[0]), batch_size)
			else:
				assert(batch_size % self.y_train_few.shape[0] == 0)
				rnd_idxs = range(self.y_train_few.shape[0])
				rnd_idxs = sample(rnd_idxs, len(rnd_idxs))
				# Not random, but it can't be really, so just shuffle it
				while len(rnd_idxs) < batch_size:
					rnd_idxs.extend(range(self.y_train_few.shape[0]))

			y_batch = self.y_train_few[rnd_idxs, :]
			X_batch = self.X_train_few[rnd_idxs, :]
		elif purpose == 'train':
			X_batch, y_batch = self.mnist.train.next_batch(batch_size)
		elif purpose == 'test':
			X_batch, y_batch = self.mnist.test.next_batch(batch_size)
		else:
			assert(False)

		return X_batch, y_batch

	def get_first_x_mnist(self, batch_size, n_classes):
		# Used for getting the style of a digit, and then reproducing it along with the other 9 digits.

		# TODO: implement for SVHN
		assert(self.dataset == 'MNIST')
		assert(batch_size % n_classes == 0)
		n = int(batch_size / n_classes)

		X = self.mnist.test.images[:n]
		y = self.mnist.test.labels[:n]

		return X, y

	def get_random_minibatch(self, batch_size, n_classes, purpose='train_few'):
		assert(purpose in ['train_few', 'train', 'test'])

		if self.dataset == 'SVHN':
			return self.__get_random_minibatch_svhn(batch_size, n_classes, purpose)
		elif self.dataset == 'MNIST':
			return self.__get_random_minibatch_mnist(batch_size, n_classes, purpose)

	def __extract_few_labels_svhn(self, X_train, y_train, res, cnt_per_class, n_classes):
		print('Creating a subset of labelled data for the semi-supervised learning.')
		data_cnt = X_train.shape[0]


		label_cnts = np.zeros((n_classes))
		labelled_cnt = cnt_per_class * n_classes
		few_annotated_imgs = np.zeros((labelled_cnt, res*res*3))
		few_annotated_ys = np.zeros((labelled_cnt))
		idx = 0

		test_cnt = 0
		sample_size = 10

		# For training the model on different *random* annotated subsets
		# Not optimal, but better than using the same labels and it's been easy to implement

		am_i_done = False
		while not am_i_done:
			test_cnt += sample_size
			sample_idxs = sample(range(data_cnt), sample_size)

			images = X_train[sample_idxs,:]
			y_ = y_train[sample_idxs]

			# images, y_ = mnist.train.next_batch(batch_size)
			for i in range(y_.shape[0]):
				label_int = y_[i]

				if(label_cnts[label_int] < cnt_per_class):
					# NB: there's a risk of repeated samples because of the way this is implemented

					few_annotated_imgs[idx] = images[i]
					few_annotated_ys[idx] = label_int
					idx += 1
					label_cnts[label_int] += 1

				if np.sum(label_cnts) == labelled_cnt:
					am_i_done = True

		print('No. of data points evaluated when creating the labelled subset: {0}'.format(test_cnt))

		assert((np.sum(few_annotated_ys == 2)) == cnt_per_class)

		return few_annotated_imgs, few_annotated_ys



