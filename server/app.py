#
# Flask app exposing access to flower recognition neural network.
#
import glob
import random
from os import path

from flask import Flask, flash, render_template, request
from torch import nn

# Constants
from server.flower_neural_net import Network, NetworkArchitectures

CATEGORY_TO_NAME_JSON = 'cat_to_name.json'
CLASS_TO_INDEX_JSON = 'class_to_index.json'
CHECKPOINT_PATH = 'checkpoint.pth'

# Configure Flask
app = Flask(__name__)
app.url_map.strict_slashes = False
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024  # 2048KB, ~2MB

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
			 class_to_index_json_path=CLASS_TO_INDEX_JSON,
			 category_to_name_json_path=CATEGORY_TO_NAME_JSON,
			 gpu=False)


@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')


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
	files_path = path.join('static', 'img', 'test', '**/*.jpg')
	files = glob.glob(files_path, recursive=True)

	classifications = []
	if files:
		# Select random sample of flower images
		files = random.sample(files, num_images)

		# Classify each image
		classifications = [classify(file, num_predictions) for file in files]

		# Remove prefix 'static' & update file paths to URLs
		files = [file[len('static/'):].replace('\\', '/') for file in files]

	return render_template('test_images.html', results=zip(files, classifications))

# TODO Complete
# @app.route('/upload', methods=['POST'])
# def classify_upload_image():
# 	# Get params from form
# 	top_matches = request.form.get('top-matches', 5)
#
# 	# Get the attached file if present
# 	file = None
# 	if 'flower-image' in request.files:
# 		if request.files.get('flower-image').filename:
# 			file = request.files['flower-image']
#
# 	# Validate presence of image file
# 	if file is None:
# 		bad_request_response_code = 400
# 		message = {
# 			'status': bad_request_response_code,
# 			'message': "You forgot to attach a file!",
# 			'url': request.url
# 		}
#
# 		return jsonify(message), bad_request_response_code
#
# 	classification = classify(file, top_matches)
#
# 	return jsonify(classification)


def classify(image_file, top_matches: int):
	# Use network to get top predictions
	category_name_list, class_id_list, probabilities_list = nw.predict(image_file, top_matches)
	return {
		'class_names': category_name_list,
		'class_ids': class_id_list,
		'probabilities': probabilities_list
	}


if __name__ == '__main__':
	app.debug = True
	app.secret_key = 'RANDOMKEY!'
	app.run(host='localhost', port=8000)
