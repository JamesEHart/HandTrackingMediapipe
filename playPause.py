###############################################
##      Open CV and MediaPipe integration    ##
###############################################

import numpy as np
import cv2
import mediapipe as mp
import pyautogui # For cursor control
from pynput.keyboard import Controller
# import speech_recognition as sr
import threading
import queue

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set larger resolution for webcam capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increase height to 1280 pixels
cap.set(cv2.CAP_PROP_FPS, 60)  # Try to set to 60 FPS if supported
print(f"Actual FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Initialize MediaPipe Hands, Pose and Face Mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    smooth_landmarks=True
)

# face_mesh = mp_face_mesh.FaceMesh(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
#     max_num_faces=1,
#     refine_landmarks=True
# )

mp_draw = mp.solutions.drawing_utils
pose_connections = mp_pose.POSE_CONNECTIONS

# Custom drawing specs
pose_drawing_spec = mp_draw.DrawingSpec(
    color=(0, 255, 0),
    thickness=3,
    circle_radius=3
)

hand_drawing_spec = mp_draw.DrawingSpec(
    color=(255, 0, 0),
    thickness=3,
    circle_radius=3
)

# face_drawing_spec = mp_draw.DrawingSpec(
#     color=(0, 0, 255),
#     thickness=1,
#     circle_radius=1
# )

# Initialize window position and size
window_x = 100  # Initial x position of window
window_y = 100  # Initial y position of window
cv2.namedWindow('Webcam Tracking', cv2.WINDOW_NORMAL)  # Create resizable window
cv2.moveWindow('Webcam Tracking', window_x, window_y)  # Set initial position
cv2.resizeWindow('Webcam Tracking', 1280, 720)  # Set window size to 1280x720

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize keyboard controller
keyboard = Controller()

# Track gesture states
was_pinched = False
last_y_pos = None

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
        pose_results = pose.process(frame_rgb)
        # face_results = face_mesh.process(frame_rgb)

        # Get image dimensions
        height, width = newframe.shape[:2]

        # Draw pose landmarks if detected
        if pose_results.pose_landmarks:
            landmark_points = []
            
            # Convert landmarks to pixel coordinates
            for landmark in pose_results.pose_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmark_points.append((x, y))
                
            # Draw skeleton
            mp_draw.draw_landmarks(
                newframe,
                pose_results.pose_landmarks,
                pose_connections,
                pose_drawing_spec,
                pose_drawing_spec
            )

            # Label upper body points
            upper_body_landmarks = [
                (0, "Nose"),
                (11, "Left Shoulder"),
                (12, "Right Shoulder"),
                (13, "Left Elbow"),
                (14, "Right Elbow"),
                (15, "Left Wrist"),
                (16, "Right Wrist"),
            ]

            # Display labels
            text_y = 30
            for idx, label in upper_body_landmarks:
                if 0 <= idx < len(landmark_points):
                    x, y = landmark_points[idx]
                    if 0 <= x < width and 0 <= y < height:
                        cv2.putText(newframe, label, 
                                  (10, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (0, 255, 0), 2)
                        text_y += 25
                        cv2.circle(newframe, (x, y), 4, (0, 0, 255), -1)

        # Draw hand landmarks and handle gestures if detected
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    newframe,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    hand_drawing_spec,
                    hand_drawing_spec
                )
                
                # Label finger tips
                finger_tips = {
                    4: "Thumb",
                    8: "Index",
                    12: "Middle",
                    16: "Ring",
                    20: "Pinky"
                }
                
                # Check thumb and index finger pinch
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                # Calculate distance between thumb and index finger
                distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                          (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                
                # If distance is small enough and wasn't pinched before, toggle play/pause
                is_pinched = distance < 0.05  # Adjust threshold as needed
                if is_pinched and not was_pinched:
                    keyboard.press(' ')  # Space bar to play/pause
                    keyboard.release(' ')
                    print("Play/Pause toggled")
                was_pinched = is_pinched
                
                # Check for fist gesture (all fingers curled)
                fingers_curled = all(
                    hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id-2].y 
                    for tip_id in [8, 12, 16, 20]
                )
                
                if fingers_curled:
                    # Get y position of hand
                    hand_y = hand_landmarks.landmark[9].y  # Palm center
                    
                    if last_y_pos is not None:
                        # Calculate vertical movement
                        y_diff = hand_y - last_y_pos
                        
                        # Adjust volume based on vertical movement
                        # if abs(y_diff) > 0.01:  # Threshold to avoid minor movements
                        #     if y_diff < 0:  # Moving up - increase volume
                        #         pyautogui.press('volumeup')
                        #     else:  # Moving down - decrease volume
                        #         pyautogui.press('volumedown')
                    
                    last_y_pos = hand_y
                else:
                    last_y_pos = None
                
                # Draw finger tips
                for tip_id, finger_name in finger_tips.items():
                    x = int(hand_landmarks.landmark[tip_id].x * width)
                    y = int(hand_landmarks.landmark[tip_id].y * height)
                    cv2.circle(newframe, (x, y), 6, (255, 0, 0), -1)
                    cv2.putText(newframe, finger_name, (x-20, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw face mesh if detected
        # if face_results.multi_face_landmarks:
        #     for face_landmarks in face_results.multi_face_landmarks:
        #         mp_draw.draw_landmarks(
        #             newframe,
        #             face_landmarks,
        #             mp_face_mesh.FACEMESH_CONTOURS,
        #             face_drawing_spec,
        #             face_drawing_spec
        #         )
                
                # Get facial expression landmarks
                # left_eyebrow = face_landmarks.landmark[65].y
                # right_eyebrow = face_landmarks.landmark[295].y
                
                # upper_lip = face_landmarks.landmark[13].y
                # lower_lip = face_landmarks.landmark[14].y
                # mouth_distance = lower_lip - upper_lip
                
                # Simple expression detection
                # expression = "Neutral"
                # if mouth_distance > 0.05:
                #     expression = "Mouth Open"
                # if left_eyebrow < 0.33 and right_eyebrow < 0.33:
                #     expression = "Surprised"
                
                # Display expression
                # cv2.putText(newframe, f"Expression: {expression}", 
                        #   (10, height - 30),
                        #   cv2.FONT_HERSHEY_SIMPLEX, 
                        #   0.8, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('Webcam Tracking', newframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
    # face_mesh.close()
