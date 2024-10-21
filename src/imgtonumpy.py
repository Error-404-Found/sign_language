import os
import cv2
import mediapipe as mp
import numpy as np

# Set up Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Directory containing gesture images
image_dir = 'images'
data = []  # List to hold (landmarks, label) pairs

# Iterate through each gesture class
for gesture in os.listdir(image_dir):
    gesture_path = os.path.join(image_dir, gesture)
    
    if not os.path.isdir(gesture_path):
        continue  # Skip if it's not a directory

    # Process each image in the gesture folder
    for image_file in os.listdir(gesture_path):
        image_path = os.path.join(gesture_path, image_file)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}")
            continue

        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks
        with mp_hands.Hands(model_complexity=1, max_num_hands=1) as hands:
            result = hands.process(rgb_image)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract landmarks and flatten them
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    # Append the landmarks and label to the data list
                    data.append((landmarks, gesture))

                    # Optional: Draw landmarks on the image
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Show the image with landmarks for debugging
                    cv2.imshow("Landmarks", image)
                    cv2.waitKey(100)  # Show each image for 100 ms

# Close all OpenCV windows
cv2.destroyAllWindows()

# Save collected data to gesture_data.npy
if data:
    np.save("gesture_data.npy", data, allow_pickle=True)
    print(f"Data saved successfully with {len(data)} samples.")
else:
    print("No data collected.")
