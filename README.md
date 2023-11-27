# Gesture Control with MMPose

Gesture Control with MMPose is a project that enables controlling your mouse cursor and performing mouse clicks using hand gestures detected through MMPose, a pose estimation library., with the help of OpenCV and PyAutoGUI,

## Demo

[Watch the demo](https://www.youtube.com/watch?v=qGHyay7qtNI)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Downloading Models](#downloading-models)
- [Dependencies](#dependencies)
- [Demo](#demo)
- [License](#license)

## Installation

### MMPose Installation

MMPose is a critical component of this project, and it needs to be installed separately. Follow the steps on the MMPose website to install it correctly.
https://mmpose.readthedocs.io/en/latest/

1. Clone the repository:

 ```bash
 git clone https://github.com/namas191297/mouse_gesture_control
 ```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set the MMPose path in `inference.py` on line 1 to line 3.

```python
import sys
MMPOSE_PATH = '/path/to/mmpose/'
sys.path.append(MMPOSE_PATH)
```

## Usage

1. Run the main script.
   
```bash
python gesture_control.py
```

2. Follow the instructions displayed on the screen to enable or disable gesture control.

## Downloading Models

Before running the script, download the required MMPose models and move them into the `saved_checkpoints` folder. You can find the models [here](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_2d_keypoint.html#rtmpose-rtmpose-on-hand5).

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- MMPose
- PyAutoGUI

## License

This project is licensed under the MIT License.

