import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from src.utils import extract_landmarks

# Load the pre-trained model
model = tf.keras.models.load_model("../models/cnn_model.h5")
gesture_labels = ['hello', 'thankyou', 'ok']  # Adjust based on your training labels

# Initialize Mediapipe hands detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

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
                
                # Extract landmarks and reshape for CNN input
                landmarks = extract_landmarks(hand_landmarks).reshape(1, 21, 3, 1)
                
                # Make a prediction
                prediction = model.predict(landmarks)
                predicted_label = gesture_labels[np.argmax(prediction)]

                # Display the predicted gesture on the frame
                cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
