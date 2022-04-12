import os


# Path variables
BASE_PATH = "/content/drive/MyDrive/colab_notebooks/AWS_project/cats_dogs_classifier/data"
RAW_DIRECTORY = os.path.sep.join([BASE_PATH, "raw"])
INTERIM_DIRECTORY = os.path.sep.join([BASE_PATH, "interim"])
PROCESSED_DIRECTORY = os.path.sep.join([BASE_PATH, "processed"])

REMOTE_URL = "https://keras-training-code.s3.eu-west-1.amazonaws.com/kagglecatsanddogs_3367a.zip"


# Model variables
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
