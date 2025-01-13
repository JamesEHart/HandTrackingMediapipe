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

# Initialize webcam with higher resolution and improved framerate
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to 1920 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to 1080 pixels
cap.set(cv2.CAP_PROP_FPS, 60)  # Try to set to 60 FPS if supported
print(f"Actual FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Initialize MediaPipe Hands, Pose and Face Mesh
mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
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
    color=(128, 0, 127),
    thickness=5,
    circle_radius=5
)

# face_drawing_spec = mp_draw.DrawingSpec(
#     color=(0, 0, 255),
#     thickness=1,
#     circle_radius=1
# )

# Get screen size and calculate center position
screen_width, screen_height = pyautogui.size()
window_width = 1920  # Default window width
window_height = 1080  # Default window height
window_x = (screen_width - window_width) // 2  # Center horizontally 
window_y = (screen_height - window_height) // 2  # Center vertically

# Initialize window position and size
cv2.namedWindow('Hand Skeleton', cv2.WINDOW_NORMAL)  # Create resizable window
cv2.moveWindow('Hand Skeleton', window_x, window_y)  # Set to center position
cv2.resizeWindow('Hand Skeleton', window_width, window_height)  # Set window size

# Initialize speech recognition
# recognizer = sr.Recognizer()
# text_queue = queue.Queue()

# def listen_for_speech():
#     while True:
#         with sr.Microphone() as source:
#             try:
#                 audio = recognizer.listen(source, timeout=1)
#                 text = recognizer.recognize_google(audio)
#                 text_queue.put(text)
#                 print(f"Heard: {text}")  # Print what was heard
#                 pyautogui.write(text + ' ')
#             except sr.WaitTimeoutError:
#                 continue
#             except sr.UnknownValueError:
#                 continue
#             except sr.RequestError:
#                 continue

# Start speech recognition in a separate thread
# speech_thread = threading.Thread(target=listen_for_speech, daemon=True)
# speech_thread.start()

# Track pinch states
was_pinched = False
was_middle_pinched = False
pinch_start_time = 0
middle_pinch_start_time = 0

# Input finger to track
tracked_finger = input("Enter the finger to track (Thumb, Index, Middle, Ring, Pinky, None): ")

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
            # landmark_points = []
            
            # Convert landmarks to pixel coordinates
            # for landmark in pose_results.pose_landmarks.landmark:
            #     x = int(landmark.x * width)
            #     y = int(landmark.y * height)
            #     landmark_points.append((x, y))
                
            # Draw skeleton
            # mp_draw.draw_landmarks(
            #     newframe,
            #     pose_results.pose_landmarks,
                # pose_connections,
                # pose_drawing_spec,
                # pose_drawing_spec
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

        # Draw hand landmarks and control cursor position if detected
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
                
                # Get tracked finger position
                if tracked_finger in finger_tips.values() or tracked_finger.lower() == "none":
                    if tracked_finger.lower() == "none":
                        continue  # Skip drawing and cursor movement if "none" is selected
                    for tip_id, finger_name in finger_tips.items():
                        if finger_name == tracked_finger:
                            x = int(hand_landmarks.landmark[tip_id].x * width)
                            y = int(hand_landmarks.landmark[tip_id].y * height)
                            cv2.circle(newframe, (x, y), 6, (255, 0, 0), -1)
                            cv2.putText(newframe, finger_name, (x-20, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                            # Scale coordinates to screen resolution and move cursor
                            screen_x = int((x / width) * screen_width)
                            screen_y = int((y / height) * screen_height)
                            pyautogui.moveTo(screen_x, screen_y, duration=0.01) # Reduced duration for faster updates
                else:
                    print(f"Invalid finger selection: {tracked_finger}")

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
                
                # # Simple expression detection
                # expression = "Neutral"
                # if mouth_distance > 0.05:
                #     expression = "Mouth Open"
                # if left_eyebrow < 0.33 and right_eyebrow < 0.33:
                #     expression = "Surprised"
                
                # Display expression
                # cv2.putText(newframe, f"Expression: {expression}", 
                #           (10, height - 30),
                #           cv2.FONT_HERSHEY_SIMPLEX, 
                #           0.8, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('Hand Skeleton', newframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    # pose.close()
    # face_mesh.close()
