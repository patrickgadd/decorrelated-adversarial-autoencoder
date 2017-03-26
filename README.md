# decorrelated-adversarial-autoencoder
Tensorflow implementation of Adversarial Autoencoders (with extra option to decorrelate style and classes)


## Capturing the style of real examples

Using existing MNIST examples, **X**, one can feed these through to capture their style q(**z** | **X**) and generate all the possible digits from 0 to 9 in the same style.

The first column below contains samples from the MNIST dataset, and each row next to these samples is the network's interpretation of their style applied to the 10 digits:

![Capturing of the style of MNIST digits](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/captured_digit_style.png)

## Sampling the style space

As **z**, the style space, is an N-dimensional space with values independently distributed as Gaussian distributions with mean 0, and variance 1, this can be sampled and the digits from 0 to 9 can be generated in the same style:

![Randomly styled digits, no. 1](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/digit_style_0.png)

![Randomly styled digits, no. 2](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/digit_style_1.png)

![Randomly styled digits, no. 3](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/digit_style_2.png)

## Style-interpolation.

This can be done by sampling two points in **z** and interpolating linearly (or otherwise) between them:

![Interpolation between random styles, no. 1](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/digit_style_interpolation_0.png)

![Interpolation between random styles, no. 2](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/digit_style_interpolation_1.png)

![Interpolation between random styles, no. 3](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/digit_style_interpolation_2.png)

![Interpolation between random styles, no. 4](https://raw.githubusercontent.com/patrickgadd/decorrelated-adversarial-autoencoder/master/assets/digit_style_interpolation_3.png)