import cv2  # opencv - display the videos stream
import depthai  # depthai - access the camera and its data packets
import numpy as np
from time import monotonic
import argparse

# convert the video frames from VideoCapture to format to send to OAK via XLinkIn
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-file", required=False, default=None,
                    help="[Optional] path to mp4 video to use.  If not specified the webcam will be used")
    ap.add_argument("--openvino-model", required=False, default='./openvino_model/saved_model_openvino_2021.4_6shave.blob')
    ap.add_argument("--threshold", required=False, default=0.5, type=float)

    args = vars(ap.parse_args())
    video_file = args['video_file']
    openvino_model_path = args['openvino_model']
    pred_threshold = args['threshold']

    pipeline = depthai.Pipeline()

    xinFrame = pipeline.create(depthai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")

    # Create NeuralNetwork Node that we will use to load our saved customer model
    custom_nn = pipeline.create(depthai.node.NeuralNetwork)
    custom_nn.setBlobPath(openvino_model_path)

    # Link Output of ImageManip -> Input of Custom NN node
    # manip.out.link(custom_nn.input)
    xinFrame.out.link(custom_nn.input)

    # Create XLinkOut nodes to send data to host for the neural network
    xout_nn = pipeline.create(depthai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    # connect neural network to XLinkOut
    custom_nn.out.link(xout_nn.input)

    # get the virtual device, and loop forever reading messages
    # from the internal queue
    with depthai.Device(pipeline) as device:
        # get a reference to the rgb queue, which contains the 600x600 frames from OAK camera
        # and a reference to the NeuralNetwork output with the model outputs of the mask
        # predictor
        qIn = device.getInputQueue(name="inFrame")
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        if video_file is not None:
            cap = cv2.VideoCapture(video_file)
        else:
            cap = cv2.VideoCapture(0)
        while cap.isOpened():
            read_correctly, frame = cap.read()
            if not read_correctly:
                break

            img = depthai.ImgFrame()
            img.setData(to_planar(frame, (224, 224)))
            img.setTimestamp(monotonic())
            img.setWidth(224)
            img.setHeight(224)
            qIn.send(img)

            # read a message from each of the queues
            in_nn = q_nn.tryGet()

            if in_nn is not None:
                # then we have a prediction
                # NN can output from multiple layers. Print all layer names:

                # The output of the print statement below looked like the following:
                # Layer name: StatefulPartitionedCall/model/dense_1/Softmax, Type: DataType.FP16, Dimensions: [1, 2]
                # this is where I got the name of the layer to pull the prediction from.
                # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in in_nn.getAllLayers()]

                # read the prediction from the output layer of the custom model
                pred = in_nn.getLayerFp16('StatefulPartitionedCall/model/dense_1/Sigmoid')
                # print the results of the prediction
                pred=pred[0]
                print(pred)
                if pred >= pred_threshold:
                    # print(f"Deer: [{pred}]")
                    cv2.putText(frame, f"Deer: [{pred:0.1f}]", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
                else:
                    # print(f"Deer: [{pred}]")
                    cv2.putText(frame, f"No Deer: [{pred:0.1f}]", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

            if frame is not None:
                cv2.imshow("Test Video", frame)

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
