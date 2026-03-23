import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time

# --- 1. SETUP & CONFIGURATION ---
print("Loading AI Model...")
model = tf.keras.models.load_model('my_road_sign_model.keras')

labels = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing for vehicles over 3.5 tons', 
    11: 'Right-of-way at intersection', 12: 'Priority road', 13: 'Yield', 14: 'STOP', 
    15: 'No vehicles', 16: 'Vehicles over 3.5 tons prohibited', 17: 'No entry', 
    18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right', 
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 
    24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals', 
    27: 'Pedestrians', 28: 'Children crossing', 29: 'Bicycles crossing', 
    30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed/passing limits', 33: 'Turn right ahead', 
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 
    40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End of no passing (over 3.5t)'
}

# --- 2. VOICE ENGINE SETUP ---
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()

last_spoken = ""

def get_sign_color(class_id):
    danger_signs = [14, 17, 9, 10, 15, 16]
    warning_signs = list(range(18, 32))
    mandatory_signs = list(range(33, 41))
    
    if class_id in danger_signs: return (50, 50, 255)
    elif class_id in warning_signs: return (0, 200, 255)
    elif class_id in mandatory_signs: return (255, 200, 0)
    else: return (0, 255, 100)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Play startup animation
start_time = time.time()
while time.time() - start_time < 2:
    success, frame = cap.read()
    if success:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(frame, "INITIALIZING TARGETING SCANNER...", (70, 240), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Road Sign AI Dashboard", frame)
        cv2.waitKey(1)

threading.Thread(target=speak, args=("Scanner Activated.",), daemon=True).start()

# --- 5. MAIN AI LOOP ---
frame_count = 0
current_label = "Awaiting Target..."
current_color = (255, 255, 255)
prob_text = ""

# Define the Targeting Box size and position
box_size = 200
x_start = int(640/2 - box_size/2)
y_start = int(480/2 - box_size/2)
x_end = x_start + box_size
y_end = y_start + box_size

while True:
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    # Draw the targeting box on the screen
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), current_color, 2)
    cv2.putText(frame, "PLACE SIGN INSIDE BOX", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 1)

    if frame_count % 3 == 0:
        # CRITICAL FIX: Crop the image to ONLY what is inside the box!
        roi = frame[y_start:y_end, x_start:x_end]
        
        # Now resize just that perfect square
        img = cv2.resize(roi, (32, 32)) / 255.0
        img = img.reshape(1, 32, 32, 3)
        
        prediction = model.predict_on_batch(img)
        class_index = np.argmax(prediction)
        probability = np.max(prediction)

        # Raised threshold to 95% to stop guessing on random background noise
        if probability > 0.95:
            sign_name = labels.get(class_index, 'Unknown')
            current_label = sign_name
            prob_text = f"CONFIDENCE: {round(float(probability)*100, 1)}%"
            current_color = get_sign_color(class_index)
            
            if sign_name != last_spoken:
                threading.Thread(target=speak, args=(sign_name,), daemon=True).start()
                last_spoken = sign_name
        else:
            current_label = "Awaiting Target..."
            prob_text = ""
            current_color = (255, 255, 255)
            last_spoken = ""

    # Build the HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.line(frame, (0, 80), (640, 80), current_color, 2)
    cv2.putText(frame, current_label, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, current_color, 2)
    cv2.putText(frame, prob_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'Q' to Shutdown", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Road Sign AI Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Shutdown sequence
threading.Thread(target=speak, args=("Shutting down.",), daemon=True).start()
cap.release()
cv2.destroyAllWindows()