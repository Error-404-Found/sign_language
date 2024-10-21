import numpy as np

def extract_landmarks(landmarks):
    """Extract 21 hand landmarks as a flattened numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
