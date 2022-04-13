# import the necessary packages
import tensorflow as tf
import wget
import glob2
import zipfile
import os

# Personal modules
import src.config


class DataProcessor:
    """
    Class for reading, processing and writing data from the
    s3 bucket.
    """

    def __init__(self, raw_directory, interim_directory, processed_directory):
        self.raw_directory = raw_directory
        self.interim_directory = interim_directory
        self.processed_directory = processed_directory

    def get_data(self, remote_url):
        """
        Downloads a file from a remote source src to a local destination dst.
        """
        wget.download(remote_url, self.raw_directory)

    def unzip_data(self):
        """
        Unpack zip files in the raw_directory
        """

        self.filename = glob2.glob(f"{self.raw_directory}/*.zip")[0]
        with zipfile.ZipFile(self.filename, "r") as zip_ref:
            zip_ref.extractall(path=self.processed_directory)

    def clean_data(self):
        """
        Clean processed data
        """
        num_skipped = 0
        for folder_name in ("Cat", "Dog"):
            folder_path = os.path.join(
                self.processed_directory, 'PetImages', folder_name)
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                #print(fpath)
                try:
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                finally:
                    fobj.close()

                if not is_jfif:
                    num_skipped += 1
                    # Delete corrupted image
                    os.remove(fpath)

        print("Deleted %d images" % num_skipped)

    def split_data(self, image_size, batch_size):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.processed_directory,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.processed_directory,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )

        return train_ds, val_ds


if __name__ == "__main__":
    print("Creating DataProcessor object ...")
    data_processor = DataProcessor(
        config.RAW_DIRECTORY, config.INTERIM_DIRECTORY, config.PROCESSED_DIRECTORY
    )

    print("Downloading the data ...")
    data_processor.get_data(config.REMOTE_URL)

    print("Unzipping the data ...")
    data_processor.unzip_data()

    print("Cleanning the data ...")
    data_processor.clean_data()

    print("Splitting the data")
    train_ds, val_ds = data_processor.split_data(
        config.IMAGE_SIZE, config.BATCH_SIZE)

    print("Split information: \n")
    print(type(train_ds), type(val_ds))

    print("Done!")
