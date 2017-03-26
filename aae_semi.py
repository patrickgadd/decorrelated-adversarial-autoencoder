from __future__ import absolute_import, division, print_function


from scipy.misc import imsave
import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope


from components import aa_discriminator, semi_supervised_encoder, semi_supervised_decoder, correlation_classifier

def random_one_hot(batch_size, n_classes):
	rnd_indices = tf.random_uniform([batch_size], minval=0, maxval=n_classes, dtype=tf.int32)
	p_y = tf.one_hot(rnd_indices, n_classes, on_value=1.0, off_value=0.0, dtype=tf.float32)

	return p_y

class AAE_Semi():
	def __init__(self, n_classes, z_dim, batch_size, normalizer_fn, img_res=28, img_channels=1, do_convolutional=True,
				 decorr_scale=0.5, network_scale=1.0, adversarial_mean=0.0, adversarial_stddev=1.0):

		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		learning_rate = self.learning_rate

		self.input_x = tf.placeholder(
			tf.float32, [batch_size, img_res * img_res * img_channels])

		self.z_tensor = tf.placeholder(
			tf.float32, [batch_size, z_dim])

		self.target_y = tf.placeholder(
			tf.float32, [batch_size, n_classes])

		self.dummy_p_yz = tf.placeholder(
			tf.float32, [batch_size, n_classes + z_dim])


		with arg_scope([layers.conv2d, layers.conv2d_transpose, layers.fully_connected],
					   activation_fn=tf.nn.relu,
					   normalizer_fn=normalizer_fn,
					   normalizer_params={'scale': True}):

			with tf.variable_scope("encoder") as scope:
				noise = tf.random_normal((batch_size, img_res*img_res*img_channels), mean=0.0, stddev=0.3, dtype=tf.float32)
				input_img = tf.add(self.input_x, noise)

				self.unnormalized_q_y_given_x, q_z_given_x = semi_supervised_encoder(input_img, z_dim, n_classes, batch_size, do_convolutional, network_scale, img_res, img_channels)

				self.q_z_given_x = q_z_given_x
				self.q_y_given_x = tf.nn.softmax(self.unnormalized_q_y_given_x)
				if tf.__version__ == '0.10.0':
					self.q_yz_given_x = tf.concat(1, [self.q_y_given_x, self.q_z_given_x])
				else:
					self.q_yz_given_x = tf.concat([self.q_y_given_x, self.q_z_given_x],1 )

				encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

				if normalizer_fn == None:
					G_shared_params = encoder_params[:-4]
					G_y_params = G_shared_params + encoder_params[-4:-2]
					G_z_params = G_shared_params + encoder_params[-2:]
				else:
					G_shared_params = encoder_params[:-6]
					G_y_params = G_shared_params + encoder_params[-6:-3]
					G_z_params = G_shared_params + encoder_params[-3:]

			with tf.variable_scope("decoder") as scope:
				output_x = semi_supervised_decoder(self.q_yz_given_x, batch_size, n_classes+z_dim, do_convolutional, network_scale, img_res, img_channels)
				decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')


			with tf.variable_scope("z_discriminator") as scope:
				# Should predict "false"
				self.z_D2 = aa_discriminator(self.q_z_given_x, batch_size, z_dim)

			with tf.variable_scope("z_discriminator", reuse=True) as scope:
				# Not the full density function, just a random sample
				self.p_z = tf.random_normal([batch_size, z_dim], mean=adversarial_mean, stddev=adversarial_stddev)
				# The output of the discriminator for p(z)
				self.z_D1 = aa_discriminator(self.p_z, batch_size, z_dim)

				D_z_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='z_discriminator')

			with tf.variable_scope("y_discriminator") as scope:
				# Not the full categorical density function, just a random sample
				# The output of the discriminator for p(y) (assuming all classes are equally likely)
				p_y = random_one_hot(batch_size, n_classes)
				self.y_D1 = aa_discriminator(p_y, batch_size, n_classes)

				D_y_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='y_discriminator')

			with tf.variable_scope("y_discriminator", reuse=True) as scope:
				self.y_D2 = aa_discriminator(self.q_y_given_x, batch_size, n_classes)

			with tf.variable_scope("decoder", reuse=True) as scope:
				# For generating digits with the same style
				self.sampled_style_digits = semi_supervised_decoder(self.dummy_p_yz, batch_size, n_classes+z_dim, do_convolutional, network_scale, img_res, img_channels)

			with tf.variable_scope("decoder", reuse=True) as scope:
				# For generating images from the original images
				self.x_given_yz_given_x = semi_supervised_decoder(self.q_yz_given_x, batch_size, n_classes+z_dim, do_convolutional, network_scale, img_res, img_channels)

			with tf.variable_scope("correlation_classifier") as scope:
				# testing if y and z are correlated. If they are, try to make this not happen, as this should mean that information other than "style", but also class, is floating through z.
				self.q_y_given_z = correlation_classifier(self.q_z_given_x, batch_size, n_classes=n_classes)

				corr_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='correlation_classifier')

		reconstruction_loss = self.__reconstruction_loss(output_x, self.input_x)
		D_z_loss = self.__discriminator_loss(self.z_D1, self.z_D2, batch_size)
		D_y_loss = self.__discriminator_loss(self.y_D1, self.y_D2, batch_size)
		G_z_loss = self.__generator_loss(self.z_D2, batch_size)
		G_y_loss = self.__generator_loss(self.y_D2, batch_size)
		classification_loss = self.__classification_loss(self.target_y, self.unnormalized_q_y_given_x, batch_size, n_classes)
		correlation_classification_loss = self.__classification_loss(self.q_y_given_x, self.q_y_given_z, batch_size, n_classes)

		global_step = tf.contrib.framework.get_or_create_global_step()

		optimizer = 'Adam'

		ae_params = encoder_params + decoder_params
		self.train_reconstruction = layers.optimize_loss(
			reconstruction_loss, global_step, learning_rate, optimizer=optimizer, variables=ae_params, update_ops=[])

		self.train_z_generator = layers.optimize_loss(
			G_z_loss, global_step, learning_rate, optimizer=optimizer, variables=G_z_params, update_ops=[])
		self.train_y_generator = layers.optimize_loss(
			G_y_loss, global_step, learning_rate, optimizer=optimizer, variables=G_y_params, update_ops=[])

		self.train_z_discrimator = layers.optimize_loss(
			D_z_loss, global_step, learning_rate, optimizer=optimizer, variables=D_z_params, update_ops=[])
		self.train_y_discrimator = layers.optimize_loss(
			D_y_loss, global_step, learning_rate, optimizer=optimizer, variables=D_y_params, update_ops=[])

		self.train_y_classifier = layers.optimize_loss(
			classification_loss, global_step, learning_rate, optimizer=optimizer, variables=G_y_params, update_ops=[])

		self.train_correlation_classifier = layers.optimize_loss(
			correlation_classification_loss, global_step, learning_rate, optimizer=optimizer, variables=corr_params, update_ops=[])

		self.train_decorrelation = layers.optimize_loss(
			-correlation_classification_loss*decorr_scale, global_step, learning_rate*decorr_scale, optimizer=optimizer, variables=encoder_params, update_ops=[])

		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())

	def reconstruction_phase(self, input_x, learning_rate):
		return self.sess.run(self.train_reconstruction, {self.input_x: input_x, self.learning_rate: learning_rate})

	def discriminator_phase(self, input_x, learning_rate):
		return self.sess.run(self.train_z_discrimator, {self.input_x: input_x, self.learning_rate: learning_rate}) + \
				self.sess.run(self.train_y_discrimator, {self.input_x: input_x, self.learning_rate: learning_rate})

	def generator_phase(self, input_x, learning_rate):
		return self.sess.run(self.train_z_generator, {self.input_x: input_x, self.learning_rate: learning_rate}) +\
			   self.sess.run(self.train_y_generator, {self.input_x: input_x, self.learning_rate: learning_rate})

	def supervised_phase(self, input_x, target_y, learning_rate):
		return self.sess.run(self.train_y_classifier, {self.input_x: input_x, self.target_y: target_y, self.learning_rate: learning_rate})

	def correlation_classifier_phase(self, input_x, learning_rate):
		return self.sess.run(self.train_correlation_classifier, {self.input_x: input_x, self.learning_rate: learning_rate})

	def decorrelation_phase(self, input_x, learning_rate):
		return self.sess.run(self.train_decorrelation, {self.input_x: input_x, self.learning_rate: learning_rate})


	# The various losses
	def __classification_loss(self, target_y, pred_y, batch_size, n_classes):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_y, logits=pred_y))

	def __discriminator_loss(self, D1, D2, batch_size):
		return (tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([batch_size],dtype=tf.int32), logits=D1))) +
			   tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([batch_size],dtype=tf.int32), logits=D2))))

	def __generator_loss(self, D2, batch_size):
		return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([batch_size],dtype=tf.int32), logits=D2))

	def __reconstruction_loss(self, output_tensor, target_tensor):
		return tf.reduce_mean(tf.square(target_tensor - output_tensor))

	# Just for sanity-checking
	def test_print_q_z_given_x(self, input_x):
		# q(z|X) should have mean 0.0 and std. dev. 1.0
		q_z = self.sess.run(self.q_z_given_x, {self.input_x: input_x})
		print('q_z:')
		print('mean(q_z):\n{0}'.format(np.mean(q_z)))
		print('stddev(q_z):\n{0}'.format(np.std(q_z)))
		print('Note-to-self: It seems that it\'s good to minimize the GAN learning rate as long as stddev(q_z) is close to 1.0')

		# for 10 classes q(y|X) should have mean 0.1 and std. dev. 0.3
		q_y, unnormalized_q_y = self.sess.run([self.q_y_given_x, self.unnormalized_q_y_given_x], {self.input_x: input_x})
		print('mean(q_y):\n{0}'.format(np.mean(q_y)))
		print('stddev(q_y):\n{0}'.format(np.std(q_y)))


	def compute_accuracy_2(self, input_x):
		# Compute the accuracy of the classifier predicting q(y|X) given q(z|X)
		correct_prediction = tf.equal(tf.argmax(self.q_y_given_z,1), tf.argmax(self.q_y_given_x,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy = self.sess.run(accuracy, {self.input_x: input_x})
		return accuracy

	def compute_accuracy(self, input_x, target_y):
		# Compute the accurace of the classifier predicting the labelled examples
		correct_prediction = tf.equal(tf.argmax(self.q_y_given_x,1), tf.argmax(self.target_y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy = self.sess.run(accuracy, {self.input_x: input_x, self.target_y: target_y})
		return accuracy

	def generate_similar_style(self, X_labelled, y_labelled, batch_size, directory, img_res, img_channels, n_classes, z_dim):
		assert(batch_size % n_classes == 0)

		# TODO: implement for SVHN
		assert(img_channels == 1)

		n = int(batch_size / n_classes)
		X_labelled = np.repeat(X_labelled, n, axis=0)

		q_yz_given_x = self.sess.run(self.q_yz_given_x, {self.input_x: X_labelled})
		q_z_given_x = q_yz_given_x[:,n_classes:] # Only pick the first N samples, and only the style hereof

		indices = []
		for i in range(batch_size):
			indices.append(i % n_classes)
		indices = np.asarray(indices)
		p_y = np.zeros((batch_size, n_classes))
		p_y[np.arange(batch_size), indices] = 1

		p_yz = np.concatenate([p_y, q_z_given_x], axis=1)

		imgs = self.sess.run(self.sampled_style_digits, {self.dummy_p_yz: p_yz})


		combined_img = np.zeros((n*img_res, (n_classes+1)*img_res))
		# The left column the is the original images
		for i in range(n):
			combined_img[i*img_res:(i+1)*img_res, 0:img_res] = X_labelled[i*n_classes].reshape(img_res, img_res)

		# The remaining ones are the digits produced from the style captured from the original imgs
		for r in range(n):
			for c in range(n_classes):

				img = imgs[r*n_classes+c].reshape(img_res, img_res)
				combined_img[r*img_res:(1+r)*img_res, (c+1)*img_res:(c+2)*img_res] = img

		imgs_folder = os.path.join(directory, 'imgs')
		if not os.path.exists(imgs_folder):
			os.makedirs(imgs_folder)

		imsave(os.path.join(imgs_folder, 'captured_digit_style.png'), combined_img)


	def generate_digits(self, batch_size, directory, img_res, img_channels, n_classes, z_dim):
		assert(batch_size % n_classes == 0)

		# TODO: implement for SVHN
		assert(img_channels == 1)

		for n in range(3):
			indices = []
			for i in range(batch_size):
				indices.append(i % n_classes)
			indices = np.asarray(indices)
			p_y = np.zeros((batch_size, n_classes))
			p_y[np.arange(batch_size), indices] = 1


			p_z = np.zeros((batch_size, z_dim), dtype='float32')
			for i in range(int(batch_size / n_classes)):
				rnd_z = np.random.normal(0, 1.0, (1, z_dim))

				# Use the same random Z for digits 0..9 to see the different digits of the same style.
				for j in range(n_classes):
					p_z[i*n_classes + j] = rnd_z


			p_yz = np.concatenate([p_y, p_z], axis=1)

			imgs = self.sess.run(self.sampled_style_digits, {self.dummy_p_yz: p_yz})

			combined_img = np.zeros((int(batch_size/n_classes)*img_res, n_classes*img_res))

			for r in range(int(batch_size/n_classes)):
				for c in range(n_classes):

					img = imgs[r*n_classes+c].reshape(img_res, img_res)
					combined_img[r*img_res:(1+r)*img_res, c*img_res:(c+1)*img_res] = img

			imgs_folder = os.path.join(directory, 'imgs')
			if not os.path.exists(imgs_folder):
				os.makedirs(imgs_folder)

			imsave(os.path.join(imgs_folder, 'digit_style_{0}.png'.format(n)), combined_img)

	def interpolate_digits(self, batch_size, directory, img_res, img_channels, n_classes, z_dim):
		assert(batch_size % n_classes == 0)

		# TODO: implement for SVHN
		assert(img_channels == 1)

		for n in range(3):
			indices = []
			for i in range(batch_size):
				indices.append(i % n_classes)
			indices = np.asarray(indices)
			p_y = np.zeros((batch_size, n_classes))
			p_y[np.arange(batch_size), indices] = 1

			p_z_1 = np.random.normal(0, 1.0, (1, z_dim))
			p_z_2 = np.random.normal(0, 1.0, (1, z_dim))
			p_z = np.zeros((batch_size, z_dim), dtype='float32')
			N = int(batch_size / n_classes)
			for i in range(N):
				interpol = i / float(N-1)
				rnd_z = p_z_1 * interpol + p_z_2 * (1-interpol)

				# Use the same random Z for digits 0..9 to see the different digits of the same style.
				for j in range(n_classes):
					p_z[i*n_classes + j] = rnd_z


			p_yz = np.concatenate([p_y, p_z], axis=1)

			imgs = self.sess.run(self.sampled_style_digits, {self.dummy_p_yz: p_yz})

			combined_img = np.zeros((int(batch_size/n_classes)*img_res, n_classes*img_res))

			for r in range(int(batch_size/n_classes)):
				for c in range(n_classes):

					img = imgs[r*n_classes+c].reshape(img_res, img_res)
					combined_img[r*img_res:(1+r)*img_res, c*img_res:(c+1)*img_res] = img

			imgs_folder = os.path.join(directory, 'imgs')
			if not os.path.exists(imgs_folder):
				os.makedirs(imgs_folder)

			imsave(os.path.join(imgs_folder, 'digit_style_interpolation_{0}.png'.format(n)), combined_img)
