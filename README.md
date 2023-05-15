# A Novel Approach for Securing DNN Architectures using Blockchain

## Environment Setup
The proposed framework is implemented in `Python` and all the required depedencies are list in `requirements.txt` file. 

**Note:** Make sure that you have installed the latest version of the `Python` interpreter, preferably $3.10.6$ or greater, and all the commands need to run the home directory of the repository.

To install the required dependencies type the following 
commands:

- Create a virtual environment and activate it.
```
python3 -m venv .env
source .env/bin/activate
```

- Install the required dependencies. You also need to install `OpenSSL` and `SQLite3` on your system.
```
pip install -r requirements.txt
```

## Running the CLI interface
The implementation ships with a minimal CLI interface that take care of the verification, user keys and output formatting.

To start the CLI type the following command
```
python main.py
```

## Setup the model
This sample implementation support on VGG11 at the moment and the model needs to be setup before performing inference on the *VM*.

To setup the model type following command in the CLI interface
```
>>> setup VGG11 ./vgg11.sqlite <key stored in model_key.txt>
```

## Performing queries on the model
After the model is setup you can performance inference on the model using the following command in the CLI interface.
```
>>> query <path of an image file>
```

