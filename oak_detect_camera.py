import cv2  # opencv - display the videos stream
import depthai  # depthai - access the camera and its data packets
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvino-model", required=False, default='./openvino_model/saved_model_openvino_2021.4_6shave.blob')
    ap.add_argument("--threshold", required=False, default=0.5, type=float)

    args = vars(ap.parse_args())
    openvino_model_path = args['openvino_model']
    pred_threshold = args['threshold']

    pipeline = depthai.Pipeline()

    # Create ColorCamera Node
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    # 600x600 is this size of the image that will be displayed
    # however, the model assumes a 224x224 so we will add an ImageManip Node
    # to resize the preview image BEFORE it goes to the model
    cam_rgb.setPreviewSize(400, 400)
    # we are not going to set the color order, because in the OpenVino Model Optimizer we specified
    # the --reverse_input_channels so the resulting image will convert from BGR (OpenCV Format) to
    # RGB which the model expects.  If you did not use that option then you would need to setColorOrder
    # cam_rgb.setColorOrder(colorOrder=depthai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setInterleaved(False)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    manip = pipeline.create(depthai.node.ImageManip)
    manip.initialConfig.setResize(224, 224)

    # connect the 600x600 preview output to the input of the ImageManip node
    cam_rgb.preview.link(manip.inputImage)

    # Create NeuralNetwork Node that we will use to load our saved customer model
    custom_nn = pipeline.create(depthai.node.NeuralNetwork)
    # use the setBlobPath below ( after you change for your path ) if you used the script create_openvino.sh from
    # https://github.com/youngsoul/pyimagesearch-face-mask-detector.git
    # custom_nn.setBlobPath("/Users/patrickryan/.cache/blobconverter/saved_model_openvino_2021.4_6shave.blob")

    # load the converted TF2 model from
    # https://github.com/youngsoul/pyimagesearch-face-mask-detector.git
    custom_nn.setBlobPath(openvino_model_path)

    # Link Output of ImageManip -> Input of Custom NN node
    manip.out.link(custom_nn.input)

    # Create XLinkOut nodes to send data to host for the camera
    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    # connect ColorCamera to XLinkOut
    cam_rgb.preview.link(xout_rgb.input)

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
        q_rgb = device.getOutputQueue("rgb")
        q_nn = device.getOutputQueue("nn")

        frame = None
        while True:
            # read a message from each of the queues
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                # then we have a frame from the OAK
                frame = in_rgb.getCvFrame()

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
                    cv2.putText(frame, f"Deer: [{pred:0.1f}]", (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6)
                else:
                    # print(f"Deer: [{pred}]")
                    cv2.putText(frame, f"No Deer: [{pred:0.1f}]", (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)

            if frame is not None:
                # Show the frame from the OAK device
                cv2.imshow("Deer Detect", frame)

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
