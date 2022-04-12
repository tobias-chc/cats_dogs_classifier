import os


# Path variables
BASE_PATH = "data"
RAW_DIRECTORY = os.path.sep.join([BASE_PATH, "raw"])
INTERIM_DIRECTORY = os.path.sep.join([BASE_PATH, "interim"])
PROCESSED_DIRECTORY = os.path.sep.join([BASE_PATH, "processed"])

REMOTE_URL = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'


# Model variables
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
BUFFER_SIZE = 32
NUM_EPOCHS = 2
