import argparse
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _create_model():
    baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # construct the head of the model that will be placed on top of the
    # base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)  # create binary classifier.  Deer exists or not

    # place the head FC model on top of the base model ( this will become
    # the actual model we will train )
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freezse them so they will
    # NOT be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def _load_datasets(dataset_dir):
    """
    Use Keras DataSet API to create dataset for training and validation
    :param dataset_dir:
    :type dataset_dir:
    :return:
    :rtype:
    """
    train_dataset = image_dataset_from_directory(
        Path(dataset_dir) / "train",
        image_size=(224, 224),
        batch_size=32
    )
    validation_dataset = image_dataset_from_directory(
        Path(dataset_dir) / "validation",
        image_size=(224, 224),
        batch_size=32
    )

    # Page 219 Deep Learning with Python 2nd Edition
    # Use Rescaling layer in the DataSet API
    # To rescale an input in the [0, 255] range to be in the [-1, 1] range, you would pass scale=1./127.5, offset=-1.
    # https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/
    # https://stackoverflow.com/questions/65025396/data-preprocessing-keras-layers-vs-tf-image-resize
    normalization_layer = Rescaling(scale=(1. / 127.5), offset=-1)
    normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    normalized_validation_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

    return normalized_train_ds, normalized_validation_ds


def _plot_model_history(H, num_epochs):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("model_history")
    plt.show()


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, default="./datasets/model",
                    help="path to input dataset")
    ap.add_argument("--model-dir", type=str,
                    default="./deer_detector_model",
                    help="path to output deer detector model")
    ap.add_argument("--dry-run", action='store_true', help="Do not train")

    args = vars(ap.parse_args())
    model_dir = args['model_dir']

    # load the datasets used for training.
    train_dataset, validation_dataset = _load_datasets(args['dataset'])

    # create the directory to save model checkpoints
    Path.mkdir(Path(model_dir), parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(filepath=model_dir, save_best_only=True, save_weights_only=False,
                        monitor="val_loss")
    ]

    model = _create_model()

    if args['dry_run']:
        print(model.summary())
    else:
        num_epochs = 5
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        H = model.fit(train_dataset,
                      epochs=num_epochs,
                      validation_data=validation_dataset,
                      callbacks=callbacks)

        _plot_model_history(H, num_epochs)


if __name__ == '__main__':
    main()
