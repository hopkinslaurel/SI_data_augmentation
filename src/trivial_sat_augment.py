import numpy as np
import random
import torch
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import math


class Scale(object):
	"""
	Scales image to be between 0-1.
	"""

	def __call__(self, img):
		return img / 255


class ToTensor(object):
	"""
	Converts numpy arrays to float Variables in Pytorch.
	"""

	def __call__(self, img):
		img = torch.from_numpy(img)
		return img


class ToFloatTensor(object):
	"""
	Converts numpy and torch arrays to float tensors.
	"""

	def __call__(self, img):
		if isinstance(img, np.ndarray):
			img = torch.from_numpy(img).float()
		elif torch.is_tensor(img):
			img = img.float()
		else:
			raise AssertionError('Img must be numpy array or torch tensor')
		return img


class ToUnit8Tensor(object):
	"""
	Converts numpy arrays to float Variables in Pytorch.
	"""

	def __call__(self, img):
		img = torch.from_numpy(img).type(torch.uint8)
		return img


#### Color Transformations ####
# based on https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide
class Brightness(object):
	"""
	Adjust image brightness.
	"""

	def __init__(self):
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, 0.99, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
		if torch.randint(2, (1,)):
			magnitude *= -1.0
		F.adjust_brightness(img, 1.0 + magnitude)
		return img


class Color(object):
	"""
	Adjust image color balance.
	"""

	def __init__(self):
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, 0.99, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
		if torch.randint(2, (1,)):
			magnitude *= -1.0
		img = F.adjust_saturation(img, 1.0 + magnitude)
		return img


class Contrast(object):
	"""
	Control the contrast of an image.
	0: completely grey image, 1: original image, >1: higher contrast
	"""

	def __init__(self):
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, 0.99, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
		if torch.randint(2, (1,)):
			magnitude *= -1.0
		F.adjust_contrast(img, 1.0 + magnitude)
		return img


class Sharpness(object):
	"""
	Adjust image sharpness.
	"""

	def __init__(self):
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, 0.99, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
		if torch.randint(2, (1,)):  
			magnitude *= -1.0
		F.adjust_sharpness(img, 1.0 + magnitude)
		return img


class Posterize(object):
	"""
	Reduce the number of bits for each color channel.
	"""

	def __init__(self):
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = 8 - (torch.arange(self.num_bins) / ((self.num_bins - 1) / 6)).round().int()
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
		img = F.posterize(img, int(magnitude))
		return img


class Solarize(object):
	"""
	All pixels above this greyscale level are inverted.
	"""

	def __init__(self):
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(255.0, 0.0, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
		img = F.solarize(img, magnitude)
		return img


class AutoContrast(object):
	"""
	AutoContrast
	"""

	def __call__(self, img):
		img = F.autocontrast(img)
		return img


class Equalize(object):
	"""
	Equalize
	"""

	def __call__(self, img):
		img = F.equalize(img)
		return img


#### Geometric transformations ####
class TranslateX(object):
	def __init__(self, size):
		self.size = size
		self.max_translate = 0.1
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, max_translate*self.size, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())

		if torch.randint(2, (1,)):
			magnitude *= -1.0

		img = F.affine(
			img,
			angle=0.0,
			translate=[int(magnitude), 0],
			scale=1.0,
			shear=[0.0, 0.0],
			fill=None,
		)

		return img

class TranslateY(object):
	def __init__(self, size):
		self.size = size
		self.max_translate = 0.1
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, self.max_translate*self.size, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())

		if torch.randint(2, (1,)):
			magnitude *= -1.0

		img = F.affine(
			img,
			angle=0.0,
			translate=[0, int(magnitude)],
			scale=1.0,
			shear=[0.0, 0.0],
			fill=None,
		)
		return img

class ShearX(object):
	"""
	Shear image in the x-direction.
	"""

	def __init__(self):
		self.max_shear = 0.3  # max shear
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, self.max_shear, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())

		if torch.randint(2, (1,)):
			magnitude *= -1.0

		img = F.affine(
			img,
			angle=0.0,
			translate=[0, 0],
			scale=1.0,
			shear=[math.degrees(math.atan(magnitude)), 0.0],
			fill=None,
			center=[0, 0],
		)
		return img


class ShearY(object):
	"""
	Shear image in the y-direction.
	"""

	def __init__(self):
		self.max_shear = 0.3
		self.num_bins = 31

	def __call__(self, img):
		magnitudes = torch.linspace(0.0, self.max_shear, self.num_bins)
		magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())

		if torch.randint(2, (1,)):
			magnitude *= -1.0

		img = F.affine(
			img,
			angle=0.0,
			translate=[0, 0],
			scale=1.0,
			shear=[0.0, math.degrees(math.atan(magnitude))],
			fill=None,
			center=[0, 0],
		)

		return img



class RandomFlip(object):
	"""
	Random horizontal and vertical flips. Flip image horizontally and vertically w/50% chance.
	"""

	def __call__(self, img):
		img = v2.RandomHorizontalFlip(p=0.5)(img)
		img = v2.RandomVerticalFlip(p=0.5)(img)
		return img


class RandomRotate(object):
	"""
	Randomly rotates image (0, 90, 180, 270 degrees).
	"""

	def __call__(self, img):
		rotations = np.random.choice([0, 1, 2, 3])
		if rotations > 0: img = torch.rot90(img, k=rotations, dims=[1, 2])
		return img


class Erase(object):
	"""
	Randomly sets 0-9 patches in image to 0 (black).
	"""

	def __init__(self):
		self.aug_max = 0.005

	def __call__(self, img):
		patches = np.random.choice(np.arange(1, 10))
		for i in range(patches):
			t = transforms.RandomErasing(p=1, scale=(0.0002, self.aug_max), ratio=(0.5, 1), value=0)
			img = t(img)

		return img


class Saturate(object):
	"""
	Randomly sets 0-9 patches in image to 1 (white).
	"""

	def __init__(self):
		self.aug_max = 0.005

	def __call__(self, img):
		patches = np.random.choice(np.arange(1, 10))
		for i in range(patches):
			t = transforms.RandomErasing(p=1, scale=(0.0002, self.aug_max), ratio=(0.5, 1), value=1)
			img = t(img)
		return img


class GaussianNoise(object):
	"""
	Applies random Gaussian noise.
	"""

	def __init__(self):
		self.aug_max = 0.04  # max standard deviation
		self.num_bins = 31

	def __call__(self, img):
		mean = 0
		magnitudes = torch.linspace(0, self.aug_max, self.num_bins)
		std = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
		return img + torch.randn(img.size()) * std + mean


def trivial_sat_transformation(augment_type, img_size, bands, means):
	"""
	Returns a randomly sampled augmentation for satellite images.
	"""

	transformation_lookup = {'auto_contrast': AutoContrast(),
							 'equalize': Equalize(),
							 'color': Color(),
							 'solarize': Solarize(),
							 'posterize': Posterize(),
							 'contrast': Contrast(),
							 'brightness': Brightness(),
							 'sharpness': Sharpness(),
							 'translateX': TranslateX(img_size),
							 'translateY': TranslateY(img_size),
							 'shearX': ShearX(),
							 'shearY': ShearY(),
							 'flip': RandomFlip(),
							 'rotate': RandomRotate(),
							 'erase': Erase(),
							 'saturate': Saturate(),
							 'noise': GaussianNoise()
							 }

	transform_list = []

	sampled_transformation = random.choice(augment_type)  # randomly sample transformation

	if sampled_transformation == 'identity':  # no transformation
		transform_list.append(ToFloatTensor())
		transform_list.append(Scale())  # converts pixels to [0-1]

	elif sampled_transformation in ['flip', 'rotate', 'noise', 'erase', 'saturate', 'translateX',
									'translateY', 'shearX', 'shearY']:
		transform_list.append(ToFloatTensor())
		transform_list.append(Scale())
		transform_list.append(transformation_lookup[sampled_transformation])

	else:
		transform_list.append(ToUnit8Tensor())  # [0-255] tensor
		transform_list.append(transformation_lookup[sampled_transformation])
		transform_list.append(ToFloatTensor())
		transform_list.append(Scale())

	if means is not None:
		transform_list.append(transforms.Normalize(means, (1,) * bands))  # do not scale by standard deviation

	return transforms.Compose(transform_list)
