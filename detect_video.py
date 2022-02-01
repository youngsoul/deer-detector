import cv2
import argparse
import numpy as np
from tensorflow.keras.models import load_model

"""
Usage:

# to use webcam from computer
python detect_video.py 

# to use a test video
python detect_video.py --video-file datasets/backyard/movies/test_deer.MOV

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-file", required=False, default=None,
                    help="[Optional] path to mp4 video to use.  If not specified the webcam will be used")
    ap.add_argument("--trained-model", required=False, default='./deer_detector_model')

    args = vars(ap.parse_args())
    video_file = args['video_file']
    trained_model_path = args['trained_model']

    if video_file is None:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_file)

    trained_model = load_model(trained_model_path)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        if frame is not None:
            # the model expects the following image format:
            # * 224x224
            # * values scaled between (-1, 1)
            # * RGB formatted images

            # Resize image to expected model size
            x = cv2.resize(frame, (224, 224))

            # opencv reads in BGR format, convert to RGB
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

            # Scale images values between -1,1
            x = x / 127.5
            x = x - 1

            # add a 'batch' dimension of 1
            x = np.expand_dims(x, axis=0)

            pred = trained_model.predict(x)
            pred = pred[0][0]
            print(pred)
            if pred >= 0.5:
                # print(f"Deer: [{pred}]")
                cv2.putText(frame, f"Deer: [{pred:0.1f}]", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
            else:
                # print(f"Deer: [{pred}]")
                cv2.putText(frame, f"No Deer: [{pred:0.1f}]", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
