from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.utils import image_dataset_from_directory
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow.keras.backend as kb
import cv2
import numpy as np

"""
Usage:

python evaluate_model.py --dataset ./datasets/model/test  --model-dir ./deer_detector_model
python evaluate_model.py --dataset ./datasets/backyard/images  --model-dir ./deer_detector_model

"""


def _load_test_dataset(dataset_dir):
    test_dataset = image_dataset_from_directory(
        Path(f"{dataset_dir}"),
        image_size=(224, 224),
        batch_size=32
    )

    # Page 219 Deep Learning with Python 2nd Edition
    # Use Rescaling layer in the DataSet API
    # To rescale an input in the [0, 255] range to be in the [-1, 1] range, you would pass scale=1./127.5, offset=-1.
    normalization_layer = Rescaling(scale=(1. / 127.5), offset=-1)
    normalized_test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return normalized_test_ds


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, default="./datasets/model/test",
                    help="path to input dataset")
    ap.add_argument("--model-dir", type=str,
                    default="./deer_detector_model",
                    help="path to output deer detector model")

    args = vars(ap.parse_args())

    # create a Keras DataSet for the test images
    test_dataset = _load_test_dataset(args['dataset'])

    # Load the trained model
    model_dir = args['model_dir']
    model = load_model(model_dir)

    # Evaluate the model on the test data
    evaluate_model_on_dataset(model, test_dataset)


def evaluate_model_on_dataset(model, test_dataset):
    # Evaluate on the test dataset as specified from cmd line options.
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")

    # for each image in test dataset, predict for each image.
    label_names = ['Background', 'Deer']
    for data_batch in test_dataset:
        # data_batch is a batch of 32 images
        batch_of_images = data_batch[0]
        batch_of_labels = data_batch[1]

        batch_of_preds = model.predict(batch_of_images)

        for image, label, pred in zip(batch_of_images, batch_of_labels, batch_of_preds):
            label = kb.get_value(label)
            pred = kb.get_value(pred)[0]

            # if we get the prediction wrong display it
            if label != int(round(pred, 0)):
                print(f"Label: {label}, Pred: {pred}")
                x = image.numpy() + 1
                x = x * 127.5
                x = x.astype(np.uint8)
                x = cv2.resize(x, (400, 400))
                label_text = "No Deer"
                if label == 1:
                    label_text = "Deer"
                cv2.putText(x, f"Actual: [{label_text}]", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if pred >= 0.5:
                    # print(f"Deer: [{pred}]")
                    cv2.putText(x, f"Pred: Deer: [{pred:0.1f}]", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # print(f"Deer: [{pred}]")
                    cv2.putText(x, f"Pred: No Deer: [{pred:0.1f}]", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                cv2.imshow("Incorrect Pred", x)
                cv2.waitKey(0)


if __name__ == '__main__':
    try:
        print('Start evaluation...')
        main()
    except:
        cv2.destroyAllWindows()
    print("Done")
