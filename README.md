# TFLite Object Detection

Real-time object detection using a TensorFlow Lite model and a webcam, running on Raspberry Pi.

Detects objects live from a webcam feed, draws bounding boxes and confidence scores on screen, and displays the FPS in the corner.

---

## What Changed From the Original

The original script by EdjeElectronics was written in 2019 and had compatibility issues with newer versions of TensorFlow and OpenCV. This updated version fixes those issues:

- **VideoStream** modernized for newer OpenCV — added `MJPG` codec via `CAP_PROP_FOURCC` to fix webcam initialization issues
- **TensorFlow import** now auto-detects whether `tflite_runtime` or full `tensorflow` is installed and imports accordingly
- **Command-line arguments** added via `argparse` — model directory, graph name, label map, threshold, resolution, and Edge TPU flag are all configurable at runtime
- **TF1 / TF2 model support** — output tensor order is detected automatically so both old and new models work
- Tested and working on Raspberry Pi with modern Python and library versions

---

## Files

| File | Description |
|---|---|
| `TFLite_detection_webcam.py` | Main detection script |
| `detect.tflite` | TensorFlow Lite model (neural network) |
| `labelmap.txt` | List of object names the model can detect |

---

## Requirements

- Raspberry Pi (tested) with a USB webcam or Picamera
- Python 3.7+
- OpenCV (`cv2`)
- TensorFlow Lite — either `tflite_runtime` or full `tensorflow`
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
pip install tflite-runtime
```

---

## How to Run

```bash
python3 TFLite_detection_webcam.py --modeldir=.
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--modeldir` | *(required)* | Folder containing the `.tflite` and `labelmap.txt` files |
| `--graph` | `detect.tflite` | Name of the `.tflite` model file |
| `--labels` | `labelmap.txt` | Name of the label map file |
| `--threshold` | `0.5` | Minimum confidence to show a detection (0.0 – 1.0) |
| `--resolution` | `1280x720` | Webcam resolution in WxH format |
| `--edgetpu` | off | Use Coral USB Edge TPU accelerator |

### Example with custom options

```bash
python3 TFLite_detection_webcam.py --modeldir=. --threshold=0.6 --resolution=640x480
```

### With Coral Edge TPU

```bash
python3 TFLite_detection_webcam.py --modeldir=. --edgetpu
```

Press **`q`** to quit the detection window.

---

## How It Works

1. The webcam runs in a **background thread** so the main program always has a fresh frame ready without waiting
2. Each frame is resized and formatted to match what the TFLite model expects
3. The model returns **bounding boxes**, **class labels**, and **confidence scores** for every detected object
4. Any detection above the confidence threshold gets a green box and label drawn on screen
5. FPS is calculated and displayed in the top-left corner

---

## Credits

Based on the original work by **EdjeElectronics**:
> [TensorFlow Lite Object Detection on Android and Raspberry Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
> Original script author: Evan Juras (2019)

The VideoStream threading approach is based on a technique by Adrian Rosebrock at [PyImageSearch](https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/).

This repository contains a modernized and updated version of the original webcam detection script for compatibility with newer TensorFlow and OpenCV versions.

---

## License

This project is licensed under the **Apache License 2.0**
See the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0) for full terms.

You are free to use, modify, and distribute this code as long as you retain the original attribution notices.
