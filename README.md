# Flower Classifier App
Flower Classifier App is an image recognition app written in Python using Flask and PyTorch.

The network state can be loaded from checkpoints saved after training on my [network training repo](https://github.com/gregdferrell/aipy-p1-image-classifier).

## Getting Started With Development

### Dependencies
- Python: 3.5.2
- flask: 1.0.2
- flask-uploads: 0.2.1
- pytorch: 0.4.0
- torchvision: 0.2.1

### Setup
- Install dependencies from `requirements.txt` into your environment.
- Create a model state dict that your network will load on startup.
  - You can train a network using my [network training repo here](https://github.com/gregdferrell/aipy-p1-image-classifier).
    - Save a checkpoint (model state dict only) after training.
  - Then update your local `app.py` (if needed) to instantiate your network using the same hyperparameters used when training your network.
- Copy `app_config_template.ini` to `app_config.ini` and fill in properties for your env (see descriptions below).

#### Properties: `app_config_template.ini`

Name | Description | Required
------------ | ------------- | -------------
state.dict.file.path | The file path to the model state dict to load on startup. | Yes
state.dict.download.url | The URL where the application can download a model state dict to use for the network. It will be downloaded to state.dict.file.path. | No (as long as the checkpoint file exists in the path specified in state.dict.file.path)

## Running the App
- Execute `python app.py`

## TODO
- Improve C3 graphs
