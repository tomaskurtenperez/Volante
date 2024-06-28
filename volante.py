import mediapipe as mp
import cv2
import numpy as np
import math
from pynput.keyboard import Controller
from math import acos, degrees

# Initializing Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize variables and keyboard controller
rx, ry, lx, ly = [0, 0, 0, 0]
keyboard = Controller()
thumb_points = [1, 2, 4]
font = cv2.FONT_HERSHEY_SIMPLEX

def detectsig(thickness):
     resultado=" "
     if thickness == [2, -1, 2, 2, 2]:
          resultado="UNO"
     if thickness == [-1, 2, 2, 2, 2]:
          resultado="DIEZ"
     if thickness == [-1, -1, 2, 2, 2]:
        resultado="ONCE"
     return resultado

def angle_of_singleline(rx, ry, lx, ly):
    x_diff = lx - rx
    y_diff = ly - ry
    return math.degrees(math.atan2(-y_diff, x_diff))

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    return int(centroid[0]), int(centroid[1])

def get_label(index, hand, results):
    for classification in results.multi_handedness:
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640, 480]).astype(int))
            return label, coords
    return None

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define finger landmark points
thumb_points = [1, 2, 4]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Initialize Mediapipe Hands
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Convert frame to RGB and flip horizontally
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        height, width, _ = frame.shape
        
        # Set image to not writable for performance
        image.flags.writeable = False
        
        # Hand detection
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        thickness = [2, 2, 2, 2, 2]
        
        # Process each hand detected
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                label_data = get_label(num, hand, results)
                if label_data:
                    text, coord = label_data
                    
                    # Process right hand
                    if text == "Right":
                        h, w, c = image.shape
                        rx, ry = int(hand.landmark[9].x * w), int(hand.landmark[9].y * h)
                        cv2.circle(image, (rx, ry), 5, (255, 0, 255), cv2.FILLED)
                    
                    # Process left hand
                    elif text == "Left":
                        h, w, c = image.shape
                        lx, ly = int(hand.landmark[9].x * w), int(hand.landmark[9].y * h)
                        cv2.circle(image, (lx, ly), 5, (255, 0, 255), cv2.FILLED)
                        
                        coordinates_thumb, coordinates_palm = [], []
                        coordinates_ft, coordinates_fb = [], []
                        
                        for index in thumb_points:
                            x, y = int(hand.landmark[index].x * width), int(hand.landmark[index].y * height)
                            cv2.circle(image, (x, y), 5, (255, 0, 255), cv2.FILLED)
                            coordinates_thumb.append([x, y])
                        
                        for index in palm_points:
                            x, y = int(hand.landmark[index].x * width), int(hand.landmark[index].y * height)
                            coordinates_palm.append([x, y])
                        
                        for index in fingertips_points:
                            x, y = int(hand.landmark[index].x * width), int(hand.landmark[index].y * height)
                            coordinates_ft.append([x, y])
                        
                        for index in finger_base_points:
                            x, y = int(hand.landmark[index].x * width), int(hand.landmark[index].y * height)
                            coordinates_fb.append([x, y])
                    
                        # Calculate thumb angle
                        p1, p2, p3 = np.array(coordinates_thumb)
                        l1, l2, l3 = np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3), np.linalg.norm(p1 - p2)
                        angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                        thumb_finger = angle > 150
                        keyboard.release('x')
                        
                        # Calculate palm centroid
                        nx, ny = palm_centroid(coordinates_palm)
                        coordinates_centroid = np.array([nx, ny])
                        
                        # Calculate distances
                        d_centrid_ft = np.linalg.norm(coordinates_centroid - np.array(coordinates_ft), axis=1)
                        d_centrid_fb = np.linalg.norm(coordinates_centroid - np.array(coordinates_fb), axis=1)
                        dif = d_centrid_ft - d_centrid_fb
                        fingers = np.append(thumb_finger, dif > 0)
                        fingers_counter = str(np.count_nonzero(fingers == True))
                        for i, finger in enumerate(fingers):
                            if finger:
                                thickness[i] = -1

                # Calculate steering angle
                radius = int(math.dist((rx, ry), (lx, ly)) // 2)
                cenx, ceny = (min(rx, lx) + abs(rx - lx) // 2), (min(ry, ly) + abs(ry - ly) // 2)
                cv2.circle(image, (cenx, ceny), radius, (0, 0, 0), 20)
                result = detectsig(thickness)
                
                # Handle keypresses based on detected signals
                if result in ("DIEZ", "ONCE"):
                    keyboard.press('x')
                    cv2.putText(image, 'X', (50, 100), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    keyboard.release('x')
                
                if result in ("UNO", "ONCE"):
                    keyboard.press('t')
                    cv2.putText(image, 'T', (50, 150), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    keyboard.release('t')

                ang = angle_of_singleline(lx, ly, rx, ry)
                if ang > 90:
                    ang -= 180
                elif ang < -90:
                    ang += 180
                if ang > 30:
                    last = 'a'
                    keyboard.release('d')
                    keyboard.press('a')
                    cv2.putText(image, 'LEFT', (50, 50), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
                elif ang < -30:
                    last = 'd'
                    keyboard.release('a')
                    keyboard.press('d')
                    cv2.putText(image, 'RIGHT', (50, 50), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    keyboard.release('d')
                    keyboard.release('a')

        # Display the resulting frame
        cv2.imshow('Hand Tracking', image)

        # Break the loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
