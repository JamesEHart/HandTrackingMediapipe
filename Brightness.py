###############################################
##      Open CV and MediaPipe integration    ##
###############################################

import numpy as np
import cv2
import mediapipe as mp
import pyautogui # For cursor control
# import speech_recognition as sr
import threading
import queue
import time
import screen_brightness_control as sbc # For controlling screen brightness

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands, Pose and Face Mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

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

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1,
    refine_landmarks=True
)

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

face_drawing_spec = mp_draw.DrawingSpec(
    color=(0, 0, 255),
    thickness=1,
    circle_radius=1
)

# Initialize window position
window_x = 100  # Initial x position of window
window_y = 100  # Initial y position of window
cv2.namedWindow('Webcam Tracking')  # Create named window
cv2.moveWindow('Webcam Tracking', window_x, window_y)  # Set initial position

# Get screen size
screen_width, screen_height = pyautogui.size()

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

# Track pinch and fist states
was_pinched = False
pinch_start_time = 0
was_fist = False
last_fist_y = None

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
        face_results = face_mesh.process(frame_rgb)

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
                
                # Get middle finger position (landmark 12)
                middle_x = int(hand_landmarks.landmark[12].x * width)
                middle_y = int(hand_landmarks.landmark[12].y * height)
                
                # Scale coordinates to screen resolution and move cursor
                screen_x = int((middle_x / width) * screen_width)
                screen_y = int((middle_y / height) * screen_height)
                pyautogui.moveTo(screen_x, screen_y, duration=0.01) # Reduced duration for faster updates
                
                # Check thumb and index finger pinch
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                # Calculate distance between thumb and index finger
                distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                          (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                
                # If distance is small enough, track pinch state
                is_pinched = distance < 0.05  # Adjust threshold as needed
                
                current_time = time.time()
                
                if is_pinched and not was_pinched:
                    # Start timing when pinch begins
                    pinch_start_time = current_time
                    was_pinched = True
                elif not is_pinched and was_pinched:
                    # When pinch ends, check duration
                    pinch_duration = current_time - pinch_start_time
                    if pinch_duration < 0.3:  # Fast pinch threshold
                        pyautogui.click()
                        print(f"Click detected at coordinates: ({screen_x}, {screen_y})")
                    pyautogui.mouseUp()  # Release any held clicks
                    was_pinched = False
                elif is_pinched and was_pinched:
                    # Check if we should start holding
                    if current_time - pinch_start_time > 0.3:  # Hold threshold
                        pyautogui.mouseDown()

                # Check for fist gesture
                finger_heights = []
                for tip_id in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky tips
                    tip_y = hand_landmarks.landmark[tip_id].y
                    pip_y = hand_landmarks.landmark[tip_id - 2].y  # PIP joint
                    finger_heights.append(tip_y > pip_y)  # True if finger is curled

                is_fist = all(finger_heights)  # All fingers must be curled
                
                # Get fist position for brightness control
                if is_fist:
                    fist_y = hand_landmarks.landmark[9].y  # Using middle finger MCP as reference
                    
                    if last_fist_y is not None:
                        # Calculate vertical movement
                        y_diff = last_fist_y - fist_y
                        
                        # Adjust brightness based on movement
                        current_brightness = sbc.get_brightness()[0]
                        brightness_change = int(y_diff * 200)  # Scale the movement to brightness
                        new_brightness = max(0, min(100, current_brightness + brightness_change))
                        sbc.set_brightness(new_brightness)
                        
                        # Display current brightness
                        cv2.putText(newframe, f"Brightness: {new_brightness}%", 
                                  (10, height - 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.8, (0, 255, 255), 2)
                    
                    last_fist_y = fist_y
                else:
                    last_fist_y = None
                
                if is_fist and not was_fist:
                    print("Fist detected - brightness control active")
                elif not is_fist and was_fist:
                    print("Fist released - brightness control inactive")
                was_fist = is_fist
                
                for tip_id, finger_name in finger_tips.items():
                    x = int(hand_landmarks.landmark[tip_id].x * width)
                    y = int(hand_landmarks.landmark[tip_id].y * height)
                    cv2.circle(newframe, (x, y), 6, (255, 0, 0), -1)
                    cv2.putText(newframe, finger_name, (x-20, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw face mesh if detected
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    newframe,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    face_drawing_spec,
                    face_drawing_spec
                )
                
                # Get facial expression landmarks
                left_eyebrow = face_landmarks.landmark[65].y
                right_eyebrow = face_landmarks.landmark[295].y
                
                upper_lip = face_landmarks.landmark[13].y
                lower_lip = face_landmarks.landmark[14].y
                mouth_distance = lower_lip - upper_lip
                
                # Simple expression detection
                expression = "Neutral"
                if mouth_distance > 0.05:
                    expression = "Mouth Open"
                if left_eyebrow < 0.33 and right_eyebrow < 0.33:
                    expression = "Surprised"
                
                # Display expression
                cv2.putText(newframe, f"Expression: {expression}", 
                          (10, height - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, (0, 0, 255), 2)

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
    face_mesh.close()
