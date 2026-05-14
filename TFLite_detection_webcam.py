import os
import cv2
import time
import numpy as np
import tensorflow as tf

from threading import Thread


# ==========================================
# SETTINGS
# ==========================================

MODEL_DIR = "."
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"

MIN_CONF_THRESHOLD = 0.5

IM_WIDTH = 640
IM_HEIGHT = 480


# ==========================================
# Video Stream
# ==========================================

class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):

        self.stream = cv2.VideoCapture(0)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.grabbed, self.frame = self.stream.read()

        if not self.grabbed:
            raise RuntimeError("Could not access webcam.")

        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):

        while not self.stopped:

            self.grabbed, self.frame = self.stream.read()

            if not self.grabbed:
                self.stop()

        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# ==========================================
# Paths
# ==========================================

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(
    CWD_PATH,
    MODEL_DIR,
    GRAPH_NAME
)

PATH_TO_LABELS = os.path.join(
    CWD_PATH,
    MODEL_DIR,
    LABELMAP_NAME
)


# ==========================================
# Load Labels
# ==========================================

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])


# ==========================================
# Load TFLite Model
# ==========================================

print("[INFO] Loading TensorFlow Lite model...")

interpreter = tf.lite.Interpreter(
    model_path=PATH_TO_CKPT
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("[INFO] Model loaded.")


# ==========================================
# Model Info
# ==========================================

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (
    input_details[0]['dtype'] == np.float32
)

input_mean = 127.5
input_std = 127.5


# ==========================================
# Output Tensor Mapping
# ==========================================

outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:
    # TF2 models
    boxes_idx = 1
    classes_idx = 3
    scores_idx = 0
else:
    # TF1 models
    boxes_idx = 0
    classes_idx = 1
    scores_idx = 2


# ==========================================
# FPS
# ==========================================

frame_rate_calc = 1
freq = cv2.getTickFrequency()


# ==========================================
# Start Camera
# ==========================================

print("[INFO] Starting webcam...")

videostream = VideoStream(
    resolution=(IM_WIDTH, IM_HEIGHT)
).start()

time.sleep(1)


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    t1 = cv2.getTickCount()

    # Grab frame
    frame1 = videostream.read()

    if frame1 is None:
        print("[ERROR] Empty frame.")
        break

    frame = frame1.copy()

    # Convert to RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    frame_resized = cv2.resize(
        frame_rgb,
        (width, height)
    )

    input_data = np.expand_dims(
        frame_resized,
        axis=0
    )

    # Normalize if needed
    if floating_model:
        input_data = (
            np.float32(input_data) - input_mean
        ) / input_std

    # Run inference
    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(
        output_details[boxes_idx]['index']
    )[0]

    classes = interpreter.get_tensor(
        output_details[classes_idx]['index']
    )[0]

    scores = interpreter.get_tensor(
        output_details[scores_idx]['index']
    )[0]

    # Draw detections
    for i in range(len(scores)):

        if (
            (scores[i] > MIN_CONF_THRESHOLD)
            and
            (scores[i] <= 1.0)
        ):

            ymin = int(max(
                1,
                (boxes[i][0] * IM_HEIGHT)
            ))

            xmin = int(max(
                1,
                (boxes[i][1] * IM_WIDTH)
            ))

            ymax = int(min(
                IM_HEIGHT,
                (boxes[i][2] * IM_HEIGHT)
            ))

            xmax = int(min(
                IM_WIDTH,
                (boxes[i][3] * IM_WIDTH)
            ))

            # Draw box
            cv2.rectangle(
                frame,
                (xmin, ymin),
                (xmax, ymax),
                (10, 255, 0),
                2
            )

            # Label
            object_name = labels[
                int(classes[i])
            ]

            label = '%s: %d%%' % (
                object_name,
                int(scores[i] * 100)
            )

            labelSize, baseLine = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            label_ymin = max(
                ymin,
                labelSize[1] + 10
            )

            cv2.rectangle(
                frame,
                (
                    xmin,
                    label_ymin - labelSize[1] - 10
                ),
                (
                    xmin + labelSize[0],
                    label_ymin + baseLine - 10
                ),
                (255, 255, 255),
                cv2.FILLED
            )

            cv2.putText(
                frame,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

    # FPS
    cv2.putText(
        frame,
        'FPS: {0:.2f}'.format(frame_rate_calc),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Show frame
    cv2.imshow(
        'TensorFlow Lite Object Detector',
        frame
    )

    # FPS calc
    t2 = cv2.getTickCount()

    time1 = (t2 - t1) / freq

    frame_rate_calc = 1 / time1

    # Quit
    if cv2.waitKey(1) == ord('q'):
        break


# ==========================================
# Cleanup
# ==========================================

print("[INFO] Exiting...")

cv2.destroyAllWindows()

videostream.stop()
