import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- 1. LOAD AND PREPARE THE DATA ---
data_path = r"dataset\Train"
num_classes = 43
image_size = (32, 32)
images = []
labels = []

print("1. Loading and Preprocessing Data (Hold on, this takes about 30 seconds)...")
for class_id in range(num_classes):
    folder_path = os.path.join(data_path, str(class_id))
    image_names = os.listdir(folder_path)
    for img_name in image_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(class_id)

# Convert to math arrays and normalize colors to be between 0 and 1
X = np.array(images) / 255.0
y = np.array(labels)

# --- 2. SPLIT THE DATA ---
print("2. Shuffling and splitting data into Training and Testing sets...")
# We use 80% to teach the AI and save 20% to test it later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. BUILD THE AI BRAIN (CNN) ---
print("3. Building the Neural Network Architecture...")
model = tf.keras.models.Sequential([
    # First layer: Looking for basic edges and shapes
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Second layer: Combining shapes to understand more complex patterns
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Flatten the image into a single line for the decision-making layers
    tf.keras.layers.Flatten(),
    
    # Final "thinking" layers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax') # 43 outputs for 43 sign types
])

# Tell the brain how to measure success
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 4. TRAIN THE AI ---
print("\n--- STARTING TRAINING (10 Epochs) ---")
# An "Epoch" is one full read-through of all 31,000 training images
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# --- 5. SAVE THE FINISHED MODEL ---
print("\nTraining Complete! Saving your trained AI...")
model.save('my_road_sign_model.keras')
print("SUCCESS! Model saved as 'my_road_sign_model.keras'")