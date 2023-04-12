import mediapipe as mp
import cv2
import numpy as np
import math
import pyautogui
from pynput.keyboard import Key, Controller
from math import acos, degrees


rx,ry,lx,ly=[0,0,0,0]
keyboard = Controller()
thumb_points = [1, 2, 4]


def angle_of_singleline(rx,ry,lx,ly):
    x_diff = lx - rx
    y_diff = ly - ry
    return math.degrees(math.atan2(-y_diff, x_diff))




def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
           
            # Process results
            label = classification.classification[0].label
            text = '{}'.format(label)
           
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
           
            output = text, coords
           
    return output




mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_hands.HandLandmark.WRIST




cap = cv2.VideoCapture(0)








with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
       
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        # Flip on horizontal
        image = cv2.flip(image, 1)
        height, width, _ = frame.shape
       
        # Set flag
        image.flags.writeable = False
       
        # Detections
        results = hands.process(image)
       
        # Set flag to true
        image.flags.writeable = True
       
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
        # Rendering results
        if results.multi_hand_landmarks:
            coordinates_thumb = []
            for num, hand in enumerate(results.multi_hand_landmarks):
           
               
                # Pulgar
                for index in thumb_points:
                         x = int(hand.landmark[index].x * width)
                         y = int(hand.landmark[index].y * height)
                         cv2.circle(image, (x, y), 5, (255, 0, 255), cv2.FILLED)
                         coordinates_thumb.append([x, y])


               
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])


                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)


                # Calcular el ángulo
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                # thumb_finger = np.array(False)
                keyboard.release('x')
                if angle > 150:
                        # thumb_finger = np.array(True)
                        #print("pulgar",angle)
                        keyboard.press('x')
                else:
                    keyboard.release('x')


                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                   
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                   
                    if text=="Right":
                        h, w, c = image.shape
                        rx, ry = int(hand.landmark[9].x * w), int(hand.landmark[9].y * h)
                        cv2.circle(image, (rx, ry), 5, (255, 0, 255), cv2.FILLED)
                       
                    elif text=="Left":
                        h, w, c = image.shape
                        lx, ly = int(hand.landmark[9].x * w), int(hand.landmark[9].y * h)
                        cv2.circle(image, (lx, ly), 5, (255, 0, 255), cv2.FILLED)




                    ang=angle_of_singleline(lx,ly,rx,ry)
                    if ang>90:
                        ang=ang-180
                    elif ang<-90:
                        ang=ang+180
                    if ang>30:
                            last='a'
                            keyboard.release('d')
                            keyboard.press('a')
                            # print("left",ang)
                    elif ang<-30:
                            last='d'
                            keyboard.release('a')
                            keyboard.press('d')
                            # print("right",ang)
                    else:
                        keyboard.release('d')
                        keyboard.release('a')
                        # print(ang)
                else:
                    keyboard.release('x')


        cv2.imshow('Hand Tracking', image)
        #rx,ry,lx,ly=[0,0,0,0]




        if cv2.waitKey(10) & 0xFF == ord('q'):
            break




cap.release()
cv2.destroyAllWindows()