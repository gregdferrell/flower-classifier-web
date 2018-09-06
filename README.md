# Flower Classifier App
Flower Classifier App is an image recognition app written in Python using Flask and PyTorch.

The network state can be loaded from checkpoints saved after training on my [other repo here](https://github.com/gregdferrell/aipy-p1-image-classifier).

## Getting Started With Development

### Dependencies
- Python: 3.5.2
- flask: 1.0.2
- flask-uploads: 0.2.1
- requests: 2.19.1
- pytorch: 0.4.0
- torchvision: 0.2.1

### Setup
- Create Conda environment based off of Python 3.5.2
- Follow the instructions on the PyTorch site to install PyTorch and TorchVision on your system: https://pytorch.org/
    ```
    # Windows 10
    conda install pytorch -c pytorch
    pip install torchvision # or pip3 install torchvision

    ```
- Install the dependencies specified in `requirements.txt`
    ```
    conda install --yes --file requirements.txt
    ```
- Train the flower recognition neural network on my [other repo here](https://github.com/gregdferrell/aipy-p1-image-classifier) and save a checkpoint (model state dict only) for the trained network.
  - Then instantiate your network here using the same hyperparameters you used when training.
- Download test images from [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and place the `test` image folder under `static\img\test`.

## Running the App
- Execute `python app.py`
