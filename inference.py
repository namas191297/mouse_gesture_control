import sys
MMPOSE_PATH = ''
sys.path.append(MMPOSE_PATH)

import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from utils import draw_keypoint_ids, draw_gesture_control_roi, enclosed_in, get_centroid_from_pts
import pyautogui

# instantiate the inferencer using the model alias
model_ckpt_path = 'saved_checkpoints/rtm_hand5.pth'
inferencer = MMPoseInferencer(pose2d='hand', pose3d_weights=model_ckpt_path)
kpts_score_threshold = 0.50
thumb_index_threshold = 30.0
smoothing_thres = 5.0
sensitivity_multiplier = 3

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
cap = cv2.VideoCapture(0)
# Get screen x and y coordinates.
screen_x, screen_y = pyautogui.size()
screen_cx, screen_cy = screen_x // 2, screen_y // 2

# Initialize flags to indicate whether gesture control should be enabled.
gesture_control_enabled = False
palm_centroid = None
prev_x = prev_y = None

# Run the loop.
while True:

    # Read the return flag and the frame.
    ret, frame = cap.read()

    # If no frame/ invalid frame was returned, break the loop.
    if not ret:
        break

    # Get the frame height and width.
    frame_w, frame_h = frame.shape[0], frame.shape[1]

    # Predict the keypoints
    result_generator = inferencer(frame, return_vis = True, thickness=2, kpt_thr=kpts_score_threshold, device='cuda')
    result = next(result_generator)
    kpts = result['predictions'][0][0]['keypoints']
    kpts_scores_mean = np.mean(result['predictions'][0][0]['keypoint_scores'])
    frame = result['visualization'][0]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw text on the screen.
    frame = cv2.putText(frame, 'Gesture Control:', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, 'Action:', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if not gesture_control_enabled:
        frame = cv2.putText(frame, 'Disabled', (170, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        frame = cv2.putText(frame, 'Enabled', (170, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


    # Get the output frame with the rectangle and the rect points (corners)
    if not gesture_control_enabled:
        output_frame, rect_pts = draw_gesture_control_roi(frame)

    # Only process this frame if the mean score for all keypoints is greater than the threshold.
    if kpts_scores_mean > kpts_score_threshold:

         # Draw the keypoints with ids on the frame and get the distance between the thumb and index finger, 
        output_frame, thumb_index_top_distance = draw_keypoint_ids(frame, kpts)

        # Draw the gesture control rectangle on the screen if gesture control has not been enabled.
        # Check if the palm is inside the gesture control 
        if not gesture_control_enabled:

            palm_inside_gesture_roi = enclosed_in(np.array(kpts), rect_pts)
            
            # If the predicted keypoints for the palm are all inside the rectangle, we can start gesture control.
            if palm_inside_gesture_roi:

                # Get the centroid of the keypoints
                palm_centroid = get_centroid_from_pts(kpts)
                print('Palm centroid is:', palm_centroid)

                # Set the position of the mouse cursor to the center of the screen.
                pyautogui.moveTo(screen_cx, screen_cy)

                # Enable gesture control
                gesture_control_enabled = True

        # Check if gesture control is enabled.
        if gesture_control_enabled:

            # Get the current centroid position of the palm keypoints.
            current_palm_centroid = get_centroid_from_pts(kpts)
            x_offset, y_offset = palm_centroid[0] - current_palm_centroid[0], palm_centroid[1] - current_palm_centroid[1]
            new_x, new_y = screen_cx + (x_offset * sensitivity_multiplier), screen_cy - (y_offset * sensitivity_multiplier)

            if prev_x is None and prev_y is None:
                prev_x = new_x
                prev_y = new_y

            else:
                if abs(prev_x - new_x) < smoothing_thres:
                    new_x = prev_x
                
                if abs(prev_y - new_y) < smoothing_thres:
                    new_y = prev_y
            
            # Move the cursor.
            if new_x > screen_x or new_x < 0 or new_y > screen_y or new_y < 0:
                continue
            else:
                pyautogui.moveTo(new_x, new_y)

            # If the distance between the thumb top and index finger top is less than the threshold, press mouse click.
            if thumb_index_top_distance < thumb_index_threshold:
                frame = cv2.putText(frame, 'CLICK', (80, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                pyautogui.click()
            else:
                frame = cv2.putText(frame, 'CONTROLLING MOUSE', (80, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, 'IDLE', (80, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        frame = cv2.putText(frame, 'IDLE', (80, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Display
    output_frame = cv2.resize(output_frame, (800,800))
    cv2.imshow('res', output_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
