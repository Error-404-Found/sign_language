import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load new or existing data (gesture_data.npy)
data = np.load("../data/gesture_data.npy", allow_pickle=True)
X = np.array([item[0] for item in data])  # Landmarks
y = np.array([item[1] for item in data])  # Labels

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

# Reshape data for CNN input (21 landmarks * 3 coordinates + 1 channel)
X = X.reshape(-1, 21, 3, 1)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load existing model or create a new CNN model
try:
    model = tf.keras.models.load_model("../models/cnn_model.h5")
    print("Loaded existing CNN model.")
except:
    print("No existing model found. Creating a new model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(21, 3, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(label_binarizer.classes_), activation='softmax')
    ])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the trained model
model.save("../models/cnn_model.h5")
print("Model saved as cnn_model.h5")
