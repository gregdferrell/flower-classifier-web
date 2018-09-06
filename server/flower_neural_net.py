#
# Neural network class and functions used for prediction.
#

import json
from collections import OrderedDict
from enum import Enum
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# Declare Image Transformation Constants for Inference
NORMAL_MEANS = (0.485, 0.456, 0.406)
NORMAL_STD_DEVIATIONS = (0.229, 0.224, 0.225)

TRANSFORM_TEST_VALIDATION = transforms.Compose([transforms.Resize(256),
												transforms.CenterCrop(224),
												transforms.ToTensor(),
												transforms.Normalize(NORMAL_MEANS, NORMAL_STD_DEVIATIONS)])


class NetworkArchitectures(Enum):
	"""
	Enum representing the network architectures available to the neural network class.
	"""
	VGG11 = 'vgg11'
	VGG13 = 'vgg13'
	VGG16 = 'vgg16'
	VGG19 = 'vgg19'


class Network:
	def __init__(self, arch: NetworkArchitectures,
				 learning_rate: float,
				 dropout_rate: float,
				 input_size: int,
				 hidden_units: Tuple,
				 output_size: int,
				 criterion,
				 epochs: int,
				 state_dict_checkpoint_path: str,
				 class_to_index_json_path: str,
				 category_to_name_json_path: str,
				 gpu: bool):
		"""
		Constructor for a network.
		:param arch: the network architecture
		:param learning_rate: the learning rate used when training the network
		:param dropout_rate: the dropout rate used when training the network
		:param input_size: the input size of the classifier
		:param hidden_units: the number of nodes in the classifier hidden layer
		:param output_size: the output size of the classifier (should equal the number of categories)
		:param criterion: function to calculate loss
		:param epochs: the number of epochs this network has been trained
		:param state_dict_checkpoint_path: path to state dict checkpoint file
		:param class_to_index_json_path: path to class to index json file
		:param category_to_name_json_path: path to category to name json file
		:param gpu: boolean indicating to use gpu or not
		"""
		self.arch = arch
		self.learning_rate = learning_rate
		self.dropout_rate = dropout_rate
		self.input_size = input_size
		self.hidden_units = hidden_units
		self.output_size = output_size
		self.criterion = criterion
		self.epochs = epochs
		self.gpu = gpu

		# Build the model using transfer learning, basing it off of the specified input architecture
		if arch == NetworkArchitectures.VGG11:
			self.model = models.vgg11(pretrained=True)
		elif arch == NetworkArchitectures.VGG13:
			self.model = models.vgg13(pretrained=True)
		elif arch == NetworkArchitectures.VGG16:
			self.model = models.vgg16(pretrained=True)
		elif arch == NetworkArchitectures.VGG19:
			self.model = models.vgg19(pretrained=True)
		else:
			raise ValueError('Invalid Network Architecture: {}'.format(arch))

		# Create custom classifier
		self.model.classifier = self.create_classifier()

		# Load state dict from pre-trained model
		device_map_location = "cuda:0" if self.gpu and torch.cuda.is_available() else "cpu"
		state_dict = torch.load(state_dict_checkpoint_path, map_location=device_map_location)
		self.model.load_state_dict(state_dict)

		# Set model to eval mode for inference
		self.model.eval()

		# Open the category names JSON file
		with open(category_to_name_json_path, 'r') as file_category_to_name:
			self.cat_to_name = json.load(file_category_to_name)

		# Mapping of classes to indexes
		with open(class_to_index_json_path, 'r') as file_class_to_index:
			self.class_to_idx = json.load(file_class_to_index)
			self.class_to_idx_items = self.class_to_idx.items()

	def create_classifier(self):
		"""
		Creates a network classifier given the current properties of the network.
		:return: the network classifier
		"""
		layers = OrderedDict([
			('fcstart', nn.Linear(self.input_size, self.hidden_units[0])),
			('relustart', nn.ReLU()),
			('dropoutstart', nn.Dropout(self.dropout_rate)),
		])
		for i in range(len(self.hidden_units) - 1):
			layers['fc{}'.format(i)] = nn.Linear(self.hidden_units[i], self.hidden_units[i + 1])
			layers['relu{}'.format(i)] = nn.ReLU()
			layers['dropout{}'.format(i)] = nn.Dropout(self.dropout_rate)
		layers['output'] = nn.Linear(self.hidden_units[-1], self.output_size)
		layers['logsoftmax'] = nn.LogSoftmax(dim=1)
		classifier = nn.Sequential(layers)
		return classifier

	def predict(self, image_path, topk=5):
		"""
		Predict the class (or classes) of an image using a trained deep learning model.
		:param image_path:
		:param topk: the number of classes to return
		:return: tuple containing [0]class_name_list, [1]class_id_list, [2]probabilities_list
		"""
		# Setup Cuda
		device = torch.device("cuda:0" if self.gpu and torch.cuda.is_available() else "cpu")
		self.model.to(device)

		# Make sure model is in eval mode
		self.model.eval()

		# Process image into numpy image, then convert to torch tensor
		np_image = self.process_image(image_path)
		torch_image = torch.from_numpy(np_image)
		torch_image = torch_image.to(device)

		with torch.no_grad():
			output = self.model(torch_image.unsqueeze_(0))
			probabilities = torch.exp(output)
			kprobs, kindex = probabilities.topk(topk)

		probabilities_list = kprobs[0].cpu().numpy().tolist()
		indexes_list = kindex[0].cpu().numpy().tolist()

		# For every kindex value, look up the class and return it instead of the index
		idx_to_class = {v: k for k, v in self.class_to_idx_items}
		class_id_list = [idx_to_class[idx] for idx in indexes_list]
		class_name_list = [self.cat_to_name[image_class].title() for image_class in class_id_list]

		return class_name_list, class_id_list, probabilities_list

	@staticmethod
	def process_image(image_path):
		"""
		Given a path to a file, pre-process that image in preparation for making a prediction.
		:param image_path: the path to the image file
		:return: the image represented by a flattened numpy array
		"""
		im_transforms = TRANSFORM_TEST_VALIDATION

		# Open image
		im = Image.open(image_path)

		# Transform it: creates pytorch tensor
		im_transformed_tensor = im_transforms(im)

		# Return np array
		np_image = np.array(im_transformed_tensor)
		return np_image
