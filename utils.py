import math
import cv2
import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1: Tuple or list representing the coordinates of the first point (x1, y1).
    - point2: Tuple or list representing the coordinates of the second point (x2, y2).

    Returns:
    - Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_keypoint_ids(frame, kpts):
    '''
    Function to draw the predicted keypoints on the frame and distance between the index finger top and the thumb finger top.

    Parameters:
    - frame: np.ndarray source image.
    - kpts: A list of keypoints predicted for the given image.

    Returns:
    - frame: np.ndarray image with keypoints draw on it.
    - thumb_index_top_dist: int or float distance between the top keypoint of the thumb and index finger.
    '''

    # Initialize variables for the loop
    index_top = thumb_top = None

    # Loop through all the given keypoints.
    for i, (x, y) in enumerate(kpts):

        # Get the index and thumb points.
        if i == 8: # Index finger top point
            index_top = (x,y)
        if i == 4: # Thumb finger top point.
            thumb_top = (x,y)

        # Calculate index and thumb distance.
        if index_top is not None and thumb_top is not None:
            thumb_index_top_dist = euclidean_distance(index_top, thumb_top)
        
        # Draw the circle and text on the frame.
        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, thumb_index_top_dist

def enclosed_in(pointsA, pointsB):
    '''
    Function to check whether a given set of points A are enclosed within points B.

    Parameters:
    - pointsA - np.ndarray of shape (n,2)
    - pointsB - np.ndarray of shape (n,2)

    Returns:
    - flag: boolean indicating whether pointsA are enclosed by points B
    '''

    # Set the flag to True.
    flag = True

    # Iterate through all the points in pointsA.
    for point in pointsA:
        if cv2.pointPolygonTest(pointsB, point, False) < 0: # If point is not enclosed by PointsB, the dist return will be less than 0.
            flag = False
            break
    
    return flag

def get_centroid_from_pts(pts):
    '''
    Function to obtain the centroid (x,y) from a given set of points.

    Parameters:
    pts: np.ndarray of shape (n,2) containing all the predicted keypoints.

    Returns:
    centroid: tuple (x,y).
    '''
    pts = np.int32(np.array(pts)[:, np.newaxis, :])
    M = cv2.moments(pts)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    centroid = (x,y)
    return centroid

def draw_gesture_control_roi(frame, rect_scale=0.15):
    '''
    Draws a big rectangle in the center of the given frame and returns the frame.

    Parameters:
    frame: np.ndarray source image.
    rect_scale: float indicatinqg the scale parameter for the rectangle so that it adapts based on the frame width and frame height.

    Returns:
    frame: np.ndarray image with a rectangle at the center of the screen.
    rect_pts: np.ndarray containing all four corners for the rectangle that has been drawn on the frame (topleft, topright, bottomright, bottomleft)
    '''

    # Get the frame height and the width.
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # Initialize the rectangle width and height.
    rect_w = int(frame_w * rect_scale)
    rect_h = int(frame_w * rect_scale)

    # Calculate all the points of interest of the framec
    frame_cx = frame_w // 2 # center x
    frame_cy = frame_h // 2 # center y
    center = (frame_cx, frame_cy)
    top_left = (frame_cx - rect_w, frame_cy - rect_h)
    top_right = (frame_cx + rect_w, frame_cy - rect_h)
    bottom_right = (frame_cx + rect_w, frame_cy + rect_h)
    bottom_left = (frame_cx - rect_w, frame_cx + rect_h)
    rect_pts = [top_left, top_right, bottom_right, bottom_left]
    rect_pts = np.array(rect_pts)

    # Draw the rectangle
    frame = cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)
    frame = cv2.circle(frame, center, 3, (0, 0, 255), 2)
    frame = cv2.putText(frame, 'Place your palm here to enable gesture control!', (top_left[0] - 70, top_left[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, rect_pts

