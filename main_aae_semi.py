'''TensorFlow implementation of https://arxiv.org/abs/1511.05644 (with variations)'''

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from progressbar import ETA, Bar, Percentage, ProgressBar

from data import Data
from aae_semi import AAE_Semi

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("max_epoch", 150, "max epoch")
flags.DEFINE_integer("updates_per_iteration", 500, "number of updates per iteration")
flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
flags.DEFINE_string("working_directory", "./data", "the directory in which the results will be stored")
flags.DEFINE_integer("z_dim", 9, "dimensionality of the z space")
flags.DEFINE_float("network_scale", 2.0, "scaling the number of neurons/filters in the network")
flags.DEFINE_float("decorrelation_importance", 0.5, "The importance of the de-correlation of q(y|X) and q(z|X)")
flags.DEFINE_integer("cnt_per_class", 10, "Number of labelled examples per class")

FLAGS = flags.FLAGS

if __name__ == "__main__":
	# is 10 for both MNIST and SVHN
	n_classes = 10

	updates_per_epoch = 50000 / FLAGS.batch_size
	max_iteration = int(FLAGS.max_epoch * updates_per_epoch / FLAGS.updates_per_iteration )

	dataset = 'MNIST'
	data = Data(dataset=dataset, n_classes=n_classes, cnt_per_class=FLAGS.cnt_per_class,
				working_directory=FLAGS.working_directory, batch_size=FLAGS.batch_size)
	if dataset == 'MNIST':
		img_res = 28
		img_channels = 1
	elif dataset == 'SVHN':
		img_res = 32
		img_channels = 3
	else:
		assert(False)

	# Important: Even if batch-norm is on, it's not applied for q(y|X) (personal taste)
	normalizer_fn = batch_norm # alternatively, use None

	log_path = 'log.txt'
	with open(log_path, 'a') as log:
		log.write('epoch\ttest acc.\ttrain acc.\ty from z acc.\n')

	model = AAE_Semi(n_classes, FLAGS.z_dim, FLAGS.batch_size, normalizer_fn,
					 decorr_scale=FLAGS.decorrelation_importance, network_scale=FLAGS.network_scale,
					 img_res=img_res, img_channels=img_channels)

	learning_rate = FLAGS.learning_rate

	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('checkpoints/')  # get latest checkpoint (if any)
	if ckpt and ckpt.model_checkpoint_path:
		# if checkpoint exists, restore the parameters and set epoch_n and i_iter
		saver.restore(model.sess, ckpt.model_checkpoint_path)
		start_iteration_n = int(ckpt.model_checkpoint_path.split('-')[1])
		print('Restored iteration no.: {0}'.format(start_iteration_n))
	else:
		# no checkpoint exists. create checkpoints directory if it does not exist.
		if not os.path.exists('checkpoints'):
			os.makedirs('checkpoints')
		if tf.__version__ == '0.10.0':
			init = tf.initialize_all_variables()
		else:
			init = tf.global_variables_initializer()
		model.sess.run(init)
		start_iteration_n = 0



	for iteration_n in range(start_iteration_n, max_iteration+1 ):
		epoch = iteration_n * FLAGS.updates_per_iteration / updates_per_epoch
		print('Beginning epoch {0}'.format(epoch))

		if epoch >= 20:
			learning_rate = 0.00003
		if epoch >= 50:
			learning_rate = 0.000003

		reconstruction_loss = 0.0
		discriminative_loss = 0.0
		generative_loss = 0.0
		classification_loss = 0.0
		corr_classification_loss = 0.0
		decorr_classification_loss = 0.0

		pbar = ProgressBar()
		for i in pbar(range(FLAGS.updates_per_iteration)):
			img_batch_unlabelled, _ = data.get_random_minibatch(FLAGS.batch_size, n_classes, purpose='train')
			img_batch_labelled, y_batch_labelled = data.get_random_minibatch(FLAGS.batch_size, n_classes, purpose='train_few')


			loss_value = model.reconstruction_phase(img_batch_unlabelled, learning_rate)
			reconstruction_loss += loss_value

			loss_value = model.discriminator_phase(img_batch_unlabelled, learning_rate)
			discriminative_loss += loss_value

			loss_value = model.generator_phase(img_batch_unlabelled, learning_rate)
			generative_loss += loss_value

			loss_value = model.supervised_phase(img_batch_labelled, y_batch_labelled, learning_rate)
			classification_loss += loss_value

			loss_value = model.correlation_classifier_phase(img_batch_unlabelled, learning_rate)
			corr_classification_loss += loss_value
			if epoch >= 1:
				loss_value = model.decorrelation_phase(img_batch_unlabelled, learning_rate)
				decorr_classification_loss += loss_value


		if int(epoch * 100) % 20 == 0:
			reconstruction_loss = reconstruction_loss / (FLAGS.updates_per_iteration * FLAGS.batch_size)
			print('Reconstruction loss: {0}'.format(reconstruction_loss))

			discriminative_loss = discriminative_loss / (FLAGS.updates_per_iteration * FLAGS.batch_size)
			print('Discriminative loss: {0}'.format(discriminative_loss))

			generative_loss = generative_loss / (FLAGS.updates_per_iteration * FLAGS.batch_size)
			print('Generative loss: {0}'.format(generative_loss))

			classification_loss = classification_loss / (FLAGS.updates_per_iteration * FLAGS.batch_size)
			print('Classification loss: {0}'.format(classification_loss))

			corr_classification_loss = corr_classification_loss / (FLAGS.updates_per_iteration * FLAGS.batch_size)
			print('Corr-Classification loss: {0}'.format(corr_classification_loss))

			# Printing some stats about q(y|X) and q(z|X)
			img_batch_unlabelled, _ = data.get_random_minibatch(FLAGS.batch_size, n_classes, purpose='train')
			model.test_print_q_z_given_x(img_batch_unlabelled)

			# Produce imagery
			X, y = data.get_first_x_mnist(FLAGS.batch_size, n_classes)
			model.generate_similar_style(X, y, FLAGS.batch_size, FLAGS.working_directory, img_res, img_channels, n_classes, FLAGS.z_dim)
			model.generate_digits(FLAGS.batch_size, FLAGS.working_directory, img_res, img_channels, n_classes, FLAGS.z_dim)
			model.interpolate_digits(FLAGS.batch_size, FLAGS.working_directory, img_res, img_channels, n_classes, FLAGS.z_dim)

		# Logging
		if int(epoch * 100) % 50 == 0:
			print('Computing the accuracies for train and test')
			test_acc_n = 10
			test_acc_sum = 0.0
			for i in range(test_acc_n):
				images, y_ = data.get_random_minibatch(FLAGS.batch_size, n_classes, purpose='test')
				test_acc_sum += model.compute_accuracy(images, y_)
			test_acc = 100*test_acc_sum/float(test_acc_n)
			print('Avg. acc for {0} test samples: {1:.2f} %'.format(FLAGS.batch_size*test_acc_n, test_acc))

			train_acc_n = 5
			train_acc_sum = 0.0
			for i in range(train_acc_n):
				images, y_ = data.get_random_minibatch(FLAGS.batch_size, n_classes, purpose='train_few')
				train_acc_sum += model.compute_accuracy(images, y_)
			train_acc = 100*train_acc_sum/float(train_acc_n)
			print('Avg. acc for {0} training samples: {1:.2f} %'.format(FLAGS.batch_size*test_acc_n, train_acc))

			zy_acc_n = 5
			zy_acc_sum = 0.0
			for i in range(zy_acc_n):
				images, y_ = data.get_random_minibatch(FLAGS.batch_size, n_classes, purpose='train')
				zy_acc_sum += model.compute_accuracy_2(images)
			zy_acc = 100*zy_acc_sum/float(zy_acc_n)
			print('Avg. acc for {0} classification samples: {1:.2f} %'.format(FLAGS.batch_size*test_acc_n, zy_acc))

			with open(log_path, 'a') as log:
				log.write('{0}\t{1}\t{2}\t{3}\n'.format(epoch, test_acc, train_acc, zy_acc))

		# Saving the network
		if int(epoch * 100) % 100 == 0:
			print('Saving the model')
			saver.save(model.sess, 'checkpoints/model.ckpt', iteration_n)
