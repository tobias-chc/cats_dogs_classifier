import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from knockknock import discord_sender


# Personal modules
import src.config as config


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


def make_model(input_shape, num_classes, data_augmentation=data_augmentation):
    """
    Function use to generate the model architecture.
    """
    inputs = keras.Input(shape=input_shape)

    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


@discord_sender(webhook_url = config.WEBHOOK_URL)
def train(model, train_ds, val_ds):
    """
    Function to create, fit and save the model.
    """
    train_ds = train_ds.prefetch(buffer_size=config.BUFFER_SIZE)
    val_ds = val_ds.prefetch(buffer_size=config.BUFFER_SIZE)

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("[INFO] training de model...")
    model.fit(
        train_ds,
        epochs=config.NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    # serialize the model to disk
    print("[INFO] saving the model...")
    model.save(config.MODEL_PATH, save_format="h5")


if __name__ == "__main__":
    import config
    print("Creating the model ...")
    model = make_model(input_shape=config.IMAGE_SIZE + (3,), num_classes=2)
    print("Model summary:", "\n")
    print(model.summary())
    print("Done!")
