import cv2
import mediapipe as mp
import numpy as np
from src.utils import extract_landmarks

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
data = []  # Store landmarks with labels

with mp_hands.Hands(model_complexity=1, max_num_hands=1) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = extract_landmarks(hand_landmarks)
                label = input("Enter gesture label (hello/thankyou/ok): ")
                data.append([landmarks, label])

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
np.save("../data/gesture_data.npy", data)  # Save data
