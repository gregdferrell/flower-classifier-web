#
# Flask app exposing access to flower recognition neural network.
#
import configparser
import glob
import os
import random
from os import path

from flask import Flask, flash, jsonify, render_template, request
from torch import nn

from flower_neural_net import Network, NetworkArchitectures

# Load Configuration
app_config = configparser.ConfigParser()
app_config.read(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app_config.ini')))

# Constants
CATEGORY_TO_NAME_JSON = 'cat_to_name.json'
CLASS_TO_INDEX_JSON = 'class_to_index.json'
CHECKPOINT_PATH = app_config['DEFAULT']['state.dict.file.path']
CHECKPOINT_DOWNLOAD_PATH = app_config['DEFAULT']['state.dict.download.url']

# Configure Flask
app = Flask(__name__)
app.url_map.strict_slashes = False
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024  # 1024KB, ~1MB

# Load the network from the checkpoint
nw = Network(arch=NetworkArchitectures.VGG13,
			 learning_rate=0.0001,
			 dropout_rate=0.2,
			 input_size=25088,
			 hidden_units=(12544,),
			 output_size=102,
			 criterion=nn.NLLLoss(),
			 epochs=6,
			 state_dict_checkpoint_path=CHECKPOINT_PATH,
			 state_dict_checkpoint_download_path=CHECKPOINT_DOWNLOAD_PATH,
			 class_to_index_json_path=CLASS_TO_INDEX_JSON,
			 category_to_name_json_path=CATEGORY_TO_NAME_JSON,
			 gpu=False)


@app.route('/', methods=['GET'])
def index():
	files, file_urls = get_images_random_sampling(1)
	return render_template('index.html', file=file_urls[0])


@app.route('/flower_categories', methods=['GET'])
def flower_categories():
	categories = sorted([cat.title() for cat in list(nw.cat_to_name.values())])
	return render_template('flower_categories.html', categories=categories)


@app.route('/test_images', methods=['GET'])
def classify_test_images():
	# Get and validate request param: number of images
	num_images = request.args.get('images', 3, type=int)
	if num_images > 10:
		flash('Number of images cannot exceed 10', 'warning')
		num_images = 10

	# Get and validate request param: number of predictions
	num_predictions = request.args.get('predictions', 5, type=int)
	if num_predictions > 10:
		flash('Number of predictions cannot exceed 10', 'warning')
		num_predictions = 10

	# Get all images, full paths & links
	files, file_urls = get_images_random_sampling(num_images)
	classifications = [classify(file, num_predictions) for file in files]

	return render_template('test_images.html', results=zip(file_urls, classifications),
						   num_images=num_images,
						   num_predictions=num_predictions)


@app.route('/upload_image', methods=['GET'])
def get_upload_image():
	return render_template('upload_image.html')


@app.route('/classify_upload_image', methods=['POST'])
def classify_upload_image():
	# Get the attached file if present
	file = None
	if 'flower-image' in request.files:
		if request.files.get('flower-image').filename:
			file = request.files['flower-image']

	# Validate presence of image file
	if file is None:
		bad_request_response_code = 400
		message = {
			'status': bad_request_response_code,
			'message': "You forgot to attach a file!",
			'url': request.url
		}

		return jsonify(message), bad_request_response_code

	classification = classify(file, 5)

	return jsonify(classification)


def classify(image_file, top_matches: int):
	"""
	Uses the neural net to classify the given image
	:param image_file: the path to the image file
	:param top_matches: the number of predictions to return
	:return: a dict containing lists of class names, class ids and probabilities
	"""
	# Use network to get top predictions
	category_name_list, class_id_list, probabilities_list = nw.predict(image_file, top_matches)
	return {
		'class_names': category_name_list,
		'class_ids': class_id_list,
		'probabilities': probabilities_list
	}


def convert_file_path_to_url(file):
	"""
	Gets a file url given a path to an image file.
	:param file: the path to the image file
	:return: a URL friendly string that Flask can use to link to the image
	"""
	# Remove static prefix and update file path separators to URL style
	file = file[len('static/'):].replace('\\', '/')
	return file


def get_images_random_sampling(num_images: int):
	"""
	Gets a random sampling of images from the test dataset.
	:param num_images: the number of images to get
	:return: a tuple containing the [0]file paths (to be used in classification) and [1]file urls (to be used by flask for showing the images)
	"""
	# Get all images, full paths & links
	files_path = path.join('static', 'img', 'test', '**/*.jpg')
	files = glob.glob(files_path, recursive=True)

	# Select random sample of flower images
	files = random.sample(files, num_images)

	# Remove prefix 'static' & update file paths to URLs
	file_urls = [file[len('static/'):].replace('\\', '/') for file in files]

	return files, file_urls


if __name__ == '__main__':
	app.debug = True
	app.secret_key = 'TODO:CHANGE-SECRET-KEY!'
	app.run(host='localhost', port=8000)
