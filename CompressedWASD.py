###############################################
##      Open CV and MediaPipe integration    ##
###############################################

import numpy as np
import cv2
import mediapipe as mp
import pyautogui # For cursor control
# import speech_recognition as sr
# import threading
# import queue
import time
from pynput.keyboard import Controller
from pynput.mouse import Button, Controller as MouseController

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands, Pose and Face Mesh
mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2  # Allow detection of both hands
)

# pose = mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
#     model_complexity=1,
#     smooth_landmarks=True
# )

# face_mesh = mp_face_mesh.FaceMesh(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
#     max_num_faces=1,
#     refine_landmarks=True
# )

mp_draw = mp.solutions.drawing_utils
# pose_connections = mp_pose.POSE_CONNECTIONS

# Custom drawing specs
# pose_drawing_spec = mp_draw.DrawingSpec(
#     color=(0, 255, 0),
#     thickness=2,
#     circle_radius=2
# )

hand_drawing_spec = mp_draw.DrawingSpec(
    color=(255, 0, 0),
    thickness=2,
    circle_radius=2
)

# face_drawing_spec = mp_draw.DrawingSpec(
#     color=(0, 0, 255),
#     thickness=1,
#     circle_radius=1
# )

# Initialize window position
window_x = 100  # Initial x position of window
window_y = 100  # Initial y position of window
cv2.namedWindow('Webcam Tracking')  # Create named window
cv2.moveWindow('Webcam Tracking', window_x, window_y)  # Set initial position

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize keyboard and mouse controllers
keyboard = Controller()
mouse = MouseController()

# Track key and mouse states
pressing_w = False
pressing_a = False
pressing_s = False
pressing_d = False
pressing_left_click = False

try:
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            continue

        newframe = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB)

        # Process for hand, pose and face detection
        hand_results = hands.process(frame_rgb)
        # pose_results = pose.process(frame_rgb)
        # face_results = face_mesh.process(frame_rgb)

        # Get image dimensions
        height, width = newframe.shape[:2]

        # Draw pose landmarks if detected
        # if pose_results.pose_landmarks:
        #     landmark_points = []
            
            # Convert landmarks to pixel coordinates
            # for landmark in pose_results.pose_landmarks.landmark:
            #     x = int(landmark.x * width)
            #     y = int(landmark.y * height)
            #     landmark_points.append((x, y))
                
            # Draw skeleton
            # mp_draw.draw_landmarks(
            #     newframe,
            #     pose_results.pose_landmarks,
            #     pose_connections,
            #     pose_drawing_spec,
            #     pose_drawing_spec
            # )

            # Label upper body points
            # upper_body_landmarks = [
            #     (0, "Nose"),
            #     (11, "Left Shoulder"),
            #     (12, "Right Shoulder"),
            #     (13, "Left Elbow"),
            #     (14, "Right Elbow"),
            #     (15, "Left Wrist"),
            #     (16, "Right Wrist"),
            # ]

            # Display labels
            # text_y = 30
            # for idx, label in upper_body_landmarks:
            #     if 0 <= idx < len(landmark_points):
            #         x, y = landmark_points[idx]
            #         if 0 <= x < width and 0 <= y < height:
            #             cv2.putText(newframe, label, 
            #                       (10, text_y), 
            #                       cv2.FONT_HERSHEY_SIMPLEX, 
            #                       0.6, (0, 255, 0), 2)
            #             text_y += 25
            #             cv2.circle(newframe, (x, y), 4, (0, 0, 255), -1)

        # Draw currently pressed keys
        key_status = []
        if pressing_w:
            key_status.append("W")
        if pressing_a:
            key_status.append("A")
        if pressing_s:
            key_status.append("S")
        if pressing_d:
            key_status.append("D")
        if pressing_left_click:
            key_status.append("LClick")
        
        status_text = "Pressed Keys: " + " ".join(key_status) if key_status else "No Keys Pressed"
        cv2.putText(newframe, status_text, 
                   (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)

        # Draw hand landmarks and handle controls if detected
        if hand_results.multi_hand_landmarks:
            # Track gestures for both hands
            left_index_extended = False
            right_index_extended = False
            left_thumb_extended = False
            right_thumb_extended = False
            right_pinky_extended = False
            
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                mp_draw.draw_landmarks(
                    newframe,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    hand_drawing_spec,
                    hand_drawing_spec
                )
                
                # Determine if this is left or right hand
                handedness = hand_results.multi_handedness[hand_idx].classification[0].label
                
                # Get finger positions
                index_tip = hand_landmarks.landmark[8]  # Index tip
                index_pip = hand_landmarks.landmark[6]  # Index PIP joint
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                thumb_ip = hand_landmarks.landmark[3]   # Thumb IP joint
                pinky_tip = hand_landmarks.landmark[20] # Pinky tip
                pinky_pip = hand_landmarks.landmark[18] # Pinky PIP joint
                wrist = hand_landmarks.landmark[0]      # Wrist point
                
                # Check finger extensions
                index_extended = index_tip.y < index_pip.y
                thumb_extended = thumb_tip.x < thumb_ip.x if handedness == "Left" else thumb_tip.x > thumb_ip.x
                pinky_extended = pinky_tip.y < pinky_pip.y
                
                if handedness == "Left":
                    # Move cursor to top of left fist (wrist point)
                    # cursor_x = int(wrist.x * screen_width)
                    # cursor_y = int(wrist.y * screen_height)
                    # pyautogui.moveTo(cursor_x, cursor_y)
                    
                    if index_extended:
                        left_index_extended = True
                    if thumb_extended:
                        left_thumb_extended = True
                else:  # Right hand
                    if index_extended:
                        right_index_extended = True
                    if thumb_extended:
                        right_thumb_extended = True
                    if pinky_extended:
                        right_pinky_extended = True
            
            # Handle W key (right thumb)
            if right_thumb_extended and not pressing_w:
                keyboard.press('w')
                pressing_w = True
                print("Pressing W")
            elif not right_thumb_extended and pressing_w:
                keyboard.release('w')
                pressing_w = False
                print("Releasing W")
                print("\033c", end="")  # Clear the console
            # Handle A key (right index)
            if right_index_extended and not pressing_a:
                keyboard.press('d')
                pressing_a = True
                print("Pressing D")
            elif not right_index_extended and pressing_a:
                keyboard.release('d')
                pressing_a = False
                print("Releasing D")
                print("\033c", end="")  # Clear the console
            # Handle S key (left thumb)
            if left_thumb_extended and not pressing_s:
                keyboard.press('s')
                pressing_s = True
                print("Pressing S")
            elif not left_thumb_extended and pressing_s:
                keyboard.release('s')
                pressing_s = False
                print("Releasing S")
                print("\033c", end="")  # Clear the console
            # Handle D key (left index)
            if left_index_extended and not pressing_d:
                keyboard.press('a')
                pressing_d = True
                print("Pressing A")
            elif not left_index_extended and pressing_d:
                keyboard.release('a')
                pressing_d = False
                print("Releasing A")
                print("\033c", end="")  # Clear the console

            # Handle left click (right pinky only)
            # if right_pinky_extended and not pressing_left_click:
            #     mouse.press(Button.left)
            #     pressing_left_click = True
            #     print("Pressing Left Click")
            # elif not right_pinky_extended and pressing_left_click:
            #     mouse.release(Button.left)
            #     pressing_left_click = False
            #     print("Releasing Left Click")

        # Show frame
        cv2.imshow('Webcam Tracking', newframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    # pose.close()
    # face_mesh.close()
