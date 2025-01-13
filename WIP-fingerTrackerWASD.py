import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize webcam with higher resolution and improved framerate
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
print(f"Actual FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

mp_draw = mp.solutions.drawing_utils
hand_drawing_spec = mp_draw.DrawingSpec(
    color=(255, 0, 0),
    thickness=1,
    circle_radius=1
)

# Get screen size and set up window
screen_width, screen_height = pyautogui.size()
window_width = 800
window_height = 600
window_x = (screen_width - window_width) // 2
window_y = (screen_height - window_height) // 2

# Initialize window
cv2.namedWindow('Hand WASD Controls', cv2.WINDOW_NORMAL)
cv2.moveWindow('Hand WASD Controls', window_x, window_y)
cv2.resizeWindow('Hand WASD Controls', window_width, window_height)

# Threshold for finger movement detection (adjust as needed)
MOVEMENT_THRESHOLD = 0.1
THUMB_EXTENSION_THRESHOLD = 0.3  # Threshold for horizontal thumb extension

# Track key states to avoid repeated presses
key_states = {'w': False, 'a': False, 's': False, 'd': False}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        newframe = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB)
        height, width = newframe.shape[:2]

        # Process hands
        results = hands.process(frame_rgb)
        
        # Reset all key states if no hands detected
        if not results.multi_hand_landmarks:
            for key in key_states:
                if key_states[key]:
                    pyautogui.keyUp(key)
                    key_states[key] = False
            
        if results.multi_hand_landmarks:
            # Dictionary to store hand landmarks
            hand_info = {}
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine if it's left or right hand
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Get relevant landmark positions for thumb extension detection
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                thumb_base = hand_landmarks.landmark[2]  # Thumb IP joint
                thumb_cmc = hand_landmarks.landmark[1]  # Thumb CMC joint
                index_y = hand_landmarks.landmark[8].y  # Index finger tip
                
                # Calculate horizontal thumb extension
                thumb_extension = abs(thumb_tip.x - thumb_cmc.x)
                
                hand_info[handedness] = {
                    'thumb_extension': thumb_extension,
                    'index_y': index_y,
                    'palm_y': hand_landmarks.landmark[0].y  # Palm position for reference
                }
                
                # Draw landmarks for visualization
                mp_draw.draw_landmarks(
                    newframe,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    hand_drawing_spec,
                    hand_drawing_spec
                )
            
            # Process WASD controls
            # Right hand controls (W and D)
            if 'Right' in hand_info:
                # W - Right Thumb Extended
                if hand_info['Right']['thumb_extension'] > THUMB_EXTENSION_THRESHOLD:
                    if not key_states['w']:
                        pyautogui.keyDown('w')
                        key_states['w'] = True
                elif key_states['w']:
                    pyautogui.keyUp('w')
                    key_states['w'] = False
                
                # D - Right Index
                if hand_info['Right']['index_y'] < hand_info['Right']['palm_y'] - MOVEMENT_THRESHOLD:
                    if not key_states['d']:
                        pyautogui.keyDown('d')
                        key_states['d'] = True
                elif key_states['d']:
                    pyautogui.keyUp('d')
                    key_states['d'] = False
            
            # Left hand controls (A and S)
            if 'Left' in hand_info:
                # S - Left Thumb Extended
                if hand_info['Left']['thumb_extension'] > THUMB_EXTENSION_THRESHOLD:
                    if not key_states['s']:
                        pyautogui.keyDown('s')
                        key_states['s'] = True
                elif key_states['s']:
                    pyautogui.keyUp('s')
                    key_states['s'] = False
                
                # A - Left Index
                if hand_info['Left']['index_y'] < hand_info['Left']['palm_y'] - MOVEMENT_THRESHOLD:
                    if not key_states['a']:
                        pyautogui.keyDown('a')
                        key_states['a'] = True
                elif key_states['a']:
                    pyautogui.keyUp('a')
                    key_states['a'] = False
            
            # Display active keys
            active_keys = [key.upper() for key, state in key_states.items() if state]
            if active_keys:
                cv2.putText(newframe, f"Active: {' '.join(active_keys)}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Hand WASD Controls', newframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources and ensure all keys are released
    for key in key_states:
        if key_states[key]:
            pyautogui.keyUp(key)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()