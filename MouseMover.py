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

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to 1920 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to 1080 pixels
cap.set(cv2.CAP_PROP_FPS, 60)  # Try to set to 60 FPS if supported
print(f"Actual FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Initialize MediaPipe Hands, Pose and Face Mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
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
    thickness=2,
    circle_radius=2
)

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

# Get screen size and calculate center position
screen_width, screen_height = pyautogui.size()
window_width = 800  # Default window width
window_height = 600  # Default window height
window_x = (screen_width - window_width) // 2  # Center horizontally 
window_y = (screen_height - window_height) // 2  # Center vertically

# Initialize window position and size
cv2.namedWindow('Webcam Tracking', cv2.WINDOW_NORMAL)  # Create resizable window
cv2.moveWindow('Webcam Tracking', window_x, window_y)  # Set to center position
cv2.resizeWindow('Webcam Tracking', window_width, window_height)  # Set window size

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
                
                # Get ring finger position (landmark 16)
                ring_x = int(hand_landmarks.landmark[16].x * width)
                ring_y = int(hand_landmarks.landmark[16].y * height)
                
                # Scale coordinates to screen resolution and move cursor
                screen_x = int((ring_x / width) * screen_width)
                screen_y = int((ring_y / height) * screen_height)
                pyautogui.moveTo(screen_x, screen_y, duration=0.01) # Reduced duration for faster updates
                
                # Check thumb and index finger pinch
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                
                # Calculate distances
                index_distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                                (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                middle_distance = ((thumb_tip.x - middle_tip.x) ** 2 + 
                                 (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
                
                # Track pinch states
                is_index_pinched = index_distance < 0.05  # Adjust threshold as needed
                is_middle_pinched = middle_distance < 0.05  # Adjust threshold as needed
                
                current_time = time.time()
                
                # Handle index finger pinch (left click)
                if is_index_pinched and not was_pinched:
                    pinch_start_time = current_time
                    was_pinched = True
                elif not is_index_pinched and was_pinched:
                    pinch_duration = current_time - pinch_start_time
                    if pinch_duration < 0.3:  # Fast pinch threshold
                        pyautogui.click()
                        print(f"Left click at coordinates: ({screen_x}, {screen_y})")
                    pyautogui.mouseUp()  # Release any held clicks
                    was_pinched = False
                elif is_index_pinched and was_pinched:
                    if current_time - pinch_start_time > 0.3:  # Hold threshold
                        pyautogui.mouseDown()

                # Handle middle finger pinch (right click)
                if is_middle_pinched and not was_middle_pinched:
                    middle_pinch_start_time = current_time
                    was_middle_pinched = True
                elif not is_middle_pinched and was_middle_pinched:
                    middle_pinch_duration = current_time - middle_pinch_start_time
                    if middle_pinch_duration < 0.3:  # Fast pinch threshold
                        pyautogui.rightClick()
                        print(f"Right click at coordinates: ({screen_x}, {screen_y})")
                    was_middle_pinched = False
                
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
