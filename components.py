import tensorflow as tf
from tensorflow.contrib import layers

def semi_supervised_encoder_convolutional(input_tensor, z_dim, y_dim, batch_size, network_scale=1.0, img_res=28, img_channels=1):
	f_multiplier = network_scale

	net = tf.reshape(input_tensor, [-1, img_res, img_res, img_channels])

	net = layers.conv2d(net, int(16*f_multiplier), 3, stride=2)
	net = layers.conv2d(net, int(16*f_multiplier), 3, stride=1)
	net = layers.conv2d(net, int(32*f_multiplier), 3, stride=2)
	net = layers.conv2d(net, int(32*f_multiplier), 3, stride=1)
	net = layers.conv2d(net, int(64*f_multiplier), 3, stride=2)
	net = layers.conv2d(net, int(64*f_multiplier), 3, stride=1)
	net = layers.conv2d(net, int(128*f_multiplier), 3, stride=2)

	net = tf.reshape(net, [batch_size, -1])
	net = layers.fully_connected(net, 1000)

	y = layers.fully_connected(net, y_dim, activation_fn=None, normalizer_fn=None)

	z = layers.fully_connected(net, z_dim, activation_fn=None)

	return y, z

def semi_supervised_encoder_fully_connected(input_tensor, z_dim, y_dim, network_scale=1.0):
	hidden_size = int(1000 * network_scale)
	net = layers.fully_connected(input_tensor, hidden_size)
	net = layers.fully_connected(net, hidden_size)

	y = layers.fully_connected(net, y_dim, activation_fn=None, normalizer_fn=None)

	z = layers.fully_connected(net, z_dim, activation_fn=None)

	return y, z

def semi_supervised_encoder(input_tensor, z_dim, y_dim, batch_size, do_convolutional, network_scale=1.0, img_res=28, img_channels=1):
	if do_convolutional:
		return semi_supervised_encoder_convolutional(input_tensor, z_dim, y_dim, batch_size, network_scale, img_res, img_channels)
	else:
		return semi_supervised_encoder_fully_connected(input_tensor, z_dim, y_dim, network_scale)

def semi_supervised_decoder_convolutional(input_tensor, batch_size, n_dimensions, network_scale=1.0, img_res=28, img_channels=1):
	f_multiplier = network_scale

	net = layers.fully_connected(input_tensor, 2*2*int(128*f_multiplier))
	net = tf.reshape(net, [-1, 2, 2, int(128*f_multiplier)])

	assert(img_res in [28, 32])

	if img_res==28:
		net = layers.conv2d_transpose(net, int(64*f_multiplier), 3, stride=2)
		net = layers.conv2d_transpose(net, int(64*f_multiplier), 3, stride=1)
		net = layers.conv2d_transpose(net, int(32*f_multiplier), 4, stride=1, padding='VALID')
		net = layers.conv2d_transpose(net, int(32*f_multiplier), 4, stride=1)
		net = layers.conv2d_transpose(net, int(16*f_multiplier), 3, stride=2)
		net = layers.conv2d_transpose(net, int(16*f_multiplier), 3, stride=1)
		net = layers.conv2d_transpose(net, int(8*f_multiplier), 3, stride=2)
		net = layers.conv2d_transpose(net, int(8*f_multiplier), 3, stride=1)
	else:
		net = layers.conv2d_transpose(net, int(64*f_multiplier), 3, stride=2)
		net = layers.conv2d_transpose(net, int(64*f_multiplier), 3, stride=1)
		net = layers.conv2d_transpose(net, int(32*f_multiplier), 3, stride=2)
		net = layers.conv2d_transpose(net, int(32*f_multiplier), 3, stride=1)
		net = layers.conv2d_transpose(net, int(16*f_multiplier), 3, stride=2)
		net = layers.conv2d_transpose(net, int(16*f_multiplier), 3, stride=1)
		net = layers.conv2d_transpose(net, int(8*f_multiplier), 3, stride=2)
		net = layers.conv2d_transpose(net, int(8*f_multiplier), 3, stride=1)

	net = layers.conv2d_transpose(net, img_channels, 5, stride=1, activation_fn=tf.nn.sigmoid)
	net = layers.flatten(net)

	return net


def semi_supervised_decoder_fully_connected(input_tensor, batch_size, n_dimensions, network_scale=1.0, img_res=28, img_channels=1):
	output_size = img_res*img_res*img_channels
	n_hid = int(1000*network_scale)

	net = layers.fully_connected(input_tensor, n_hid)
	net = layers.fully_connected(net, n_hid)

	net = layers.fully_connected(net, output_size, activation_fn=tf.nn.sigmoid)

	return net


def semi_supervised_decoder(input_tensor, batch_size, n_dimensions, do_convolutional, network_scale=1.0, img_res=28, img_channels=1):
	if do_convolutional:
		return semi_supervised_decoder_convolutional(input_tensor, batch_size, n_dimensions, network_scale, img_res, img_channels)
	else:
		return semi_supervised_decoder_fully_connected(input_tensor, batch_size, n_dimensions, network_scale, img_res, img_channels)

def aa_discriminator(input_tensor, batch_size, n_dimensions):
	n_hid = 1000

	net = layers.fully_connected(input_tensor, n_hid)
	net = layers.fully_connected(net, n_hid)

	return layers.fully_connected(net, 2, activation_fn=None)

def correlation_classifier(input_tensor, batch_size, n_classes=10):
	n_hid = 1000

	net = layers.fully_connected(input_tensor, n_hid)
	net = layers.fully_connected(net, n_hid)
	net = layers.fully_connected(net, n_classes, activation_fn=None)

	return net