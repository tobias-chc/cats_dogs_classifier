import os


# Path variables
BASE_PATH = "data"
RAW_DIRECTORY = os.path.sep.join([BASE_PATH, "raw"])
INTERIM_DIRECTORY = os.path.sep.join([BASE_PATH, "interim"])
PROCESSED_DIRECTORY = os.path.sep.join([BASE_PATH, "processed"])

REMOTE_URL = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'


# Model variables
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 2
BUFFER_SIZE = 32
NUM_EPOCHS = 2

# Discord notifications

#Class
#WEBHOOK_URL = "https://discord.com/api/webhooks/963737276649701416/1nqToKcI4G-GxHB1g72Iu-mWqf6PNFpfQdDfpFO7Wjq4qwZ_-N7a9hPPLpvjf_fJjGKc"

#Personal
WEBHOOK_URL = 'https://discord.com/api/webhooks/963930993843113994/OmmuPHpWgYdalAQQbzlcacoT6Sra4VX2wZpszfec5xEDOlVy3OZIO3c9vXyg5_3FIFSb'
