import src.data.load_data as data
import src.models.train_model as model_trainer
import src.config as config
from src.models.save_model_s3 import upload_file

# 1. Download and unzip the data

print("Creating DataProcessor object ...")
data_processor = data.DataProcessor(
    config.RAW_DIRECTORY, config.INTERIM_DIRECTORY, config.PROCESSED_DIRECTORY
)

print("Downloading the data ...")
data_processor.get_data(config.REMOTE_URL)


print("Unzipping the data ...")
data_processor.unzip_data()

print("Cleanning the data ...")
data_processor.clean_data()

print("Splitting the data")
train_ds, val_ds = data_processor.split_data(config.IMAGE_SIZE, config.BATCH_SIZE)

print("Split information: \n")
print(type(train_ds), type(val_ds))

print("Done!")

# 2. Train and saving the model

print("Creating the model ...")
model = model_trainer.make_model(input_shape=config.IMAGE_SIZE + (3,), num_classes=2)
print("Model summary:", "\n")
print(model.summary())
print("Training and saving the model")
model_trainer.train(model, train_ds, val_ds)
print(f"Pushing the model into {config.BUCKET} s3 bucket")
upload_file(config.MODEL_PATH, config.BUCKET, config.MODEL_PATH)
print("Done!")
