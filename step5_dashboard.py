import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

# --- 1. LOAD MODEL & LABELS ---
print("Loading AI Model...")
model = tf.keras.models.load_model('my_road_sign_model.keras')

labels = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing (over 3.5t)', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'STOP', 15: 'No vehicles', 
    16: 'Vehicles over 3.5t prohibited', 17: 'No entry', 18: 'General caution', 
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed/passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End of no passing (over 3.5t)'
}

# --- 2. VOICE ENGINE ---
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()

# --- 3. UI SETUP (LIGHT MODE) ---
# Colors
BG_COLOR = "#F0F4F8"      # Light gray-blue background
PANEL_BG = "#FFFFFF"      # Pure white panels
TEXT_COLOR = "#102A43"    # Dark navy text
ACCENT_COLOR = "#334E68"  # Slate blue accents
HIGHLIGHT = "#0078D4"     # Bright blue for titles

root = tk.Tk()
root.title("AI Road Sign Recognition Dashboard")
root.geometry("1100x700")
root.configure(bg=BG_COLOR)

# Fonts
title_font = font.Font(family="Segoe UI", size=20, weight="bold")
header_font = font.Font(family="Segoe UI", size=12, weight="bold")
data_font = font.Font(family="Segoe UI", size=11)
result_font = font.Font(family="Segoe UI", size=24, weight="bold")

# Title Bar
title_label = tk.Label(root, text="AI ROAD SIGN DETECTION & RECOGNITION SYSTEM", 
                       font=title_font, bg=BG_COLOR, fg=TEXT_COLOR, pady=15)
title_label.pack(side="top", fill="x")

# Main Container
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill="both", expand=True, padx=20, pady=5)

# Left Column (Camera + Final Output)
left_frame = tk.Frame(main_frame, bg=BG_COLOR)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

# Camera Panel
cam_panel = tk.Frame(left_frame, bg=PANEL_BG, bd=1, relief="solid")
cam_panel.pack(side="top", fill="both", expand=True)
cam_title = tk.Label(cam_panel, text="Live Camera Feed", font=header_font, bg=PANEL_BG, fg=HIGHLIGHT, anchor="w")
cam_title.pack(fill="x", padx=10, pady=5)
cam_label = tk.Label(cam_panel, bg="black") 
cam_label.pack(padx=10, pady=(0, 10))

# Bottom Banner (Final Result)
result_panel = tk.Frame(left_frame, bg=PANEL_BG, bd=1, relief="solid")
result_panel.pack(side="bottom", fill="x", pady=(10, 0))
result_title = tk.Label(result_panel, text="Detected Sign", font=header_font, bg=PANEL_BG, fg=HIGHLIGHT, anchor="w")
result_title.pack(fill="x", padx=10, pady=(5, 0))
result_display = tk.Label(result_panel, text="Awaiting Target...", font=result_font, bg=PANEL_BG, fg="#102A43", pady=15)
result_display.pack()

# Right Column (Side Panels)
right_frame = tk.Frame(main_frame, bg=BG_COLOR, width=300)
right_frame.pack(side="right", fill="y")

# Right Panel 1: Cropped View
crop_panel = tk.Frame(right_frame, bg=PANEL_BG, bd=1, relief="solid")
crop_panel.pack(side="top", fill="x", pady=(0, 10))
crop_title = tk.Label(crop_panel, text="Targeting Scanner View", font=header_font, bg=PANEL_BG, fg=HIGHLIGHT, anchor="w")
crop_title.pack(fill="x", padx=10, pady=5)
crop_label = tk.Label(crop_panel, bg="black", width=200, height=200)
crop_label.pack(padx=10, pady=(0, 10))

# Right Panel 2: AI Vision (Normalized)
vision_panel = tk.Frame(right_frame, bg=PANEL_BG, bd=1, relief="solid")
vision_panel.pack(side="top", fill="x", pady=(0, 10))
vision_title = tk.Label(vision_panel, text="AI Vision Matrix (32x32)", font=header_font, bg=PANEL_BG, fg=HIGHLIGHT, anchor="w")
vision_title.pack(fill="x", padx=10, pady=5)
vision_label = tk.Label(vision_panel, bg="black", width=200, height=200)
vision_label.pack(padx=10, pady=(0, 10))

# Right Panel 3: History Log
history_panel = tk.Frame(right_frame, bg=PANEL_BG, bd=1, relief="solid")
history_panel.pack(side="top", fill="both", expand=True)
history_title = tk.Label(history_panel, text="Detection Log", font=header_font, bg=PANEL_BG, fg=HIGHLIGHT, anchor="w")
history_title.pack(fill="x", padx=10, pady=5)
history_text = tk.Text(history_panel, font=data_font, bg=PANEL_BG, fg=TEXT_COLOR, wrap="word", height=5, width=25, bd=0)
history_text.pack(padx=10, pady=(0, 10), fill="both", expand=True)

# --- 4. VIDEO & AI LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

box_size = 200
x_start = int(640/2 - box_size/2)
y_start = int(480/2 - box_size/2)
x_end = x_start + box_size
y_end = y_start + box_size

frame_count = 0
last_spoken = ""
history_list = []

def update_gui():
    global frame_count, last_spoken, history_list
    success, frame = cap.read()
    
    if success:
        frame_count += 1
        
        # Draw Targeting Box
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 200, 0), 2)
        cv2.putText(frame, "TARGETING SCANNER", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Extract the region inside the box
        roi = frame[y_start:y_end, x_start:x_end]

        if frame_count % 3 == 0:
            # Prepare image for AI
            img_resized = cv2.resize(roi, (32, 32))
            img_normalized = img_resized / 255.0
            img_input = img_normalized.reshape(1, 32, 32, 3)

            # Predict
            prediction = model.predict_on_batch(img_input)
            class_index = np.argmax(prediction)
            probability = np.max(prediction)

            if probability > 0.95:
                sign_name = labels.get(class_index, 'Unknown')
                prob_percent = round(float(probability) * 100, 1)
                
                # Update Bottom Banner
                result_display.config(text=f"{sign_name} ({prob_percent}%)", fg="#00A300")
                
                # Handle Voice and History Log
                if sign_name != last_spoken:
                    threading.Thread(target=speak, args=(sign_name,), daemon=True).start()
                    last_spoken = sign_name
                    
                    # Update text log
                    history_list.insert(0, f"• {sign_name}")
                    if len(history_list) > 5:
                        history_list.pop()
                    history_text.delete(1.0, tk.END)
                    history_text.insert(tk.END, "\n".join(history_list))
            else:
                result_display.config(text="Awaiting Target...", fg="#102A43")

            # Update Side Panels (Cropped & AI Vision)
            try:
                # 1. Show raw cropped image
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_pil = Image.fromarray(roi_rgb).resize((200, 200))
                roi_tk = ImageTk.PhotoImage(image=roi_pil)
                crop_label.imgtk = roi_tk
                crop_label.configure(image=roi_tk)

                # 2. Show AI Normalized Vision
                vision_display = (img_normalized * 255).astype(np.uint8)
                vision_rgb = cv2.cvtColor(vision_display, cv2.COLOR_BGR2RGB)
                vision_pil = Image.fromarray(vision_rgb).resize((200, 200), Image.NEAREST)
                vision_tk = ImageTk.PhotoImage(image=vision_pil)
                vision_label.imgtk = vision_tk
                vision_label.configure(image=vision_tk)
            except Exception as e:
                pass # Prevents crashing if the box goes out of bounds

        # Update Main Camera View
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        cam_label.imgtk = frame_tk
        cam_label.configure(image=frame_tk)

    # Loop this function every 15 milliseconds
    root.after(15, update_gui)

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the loop
update_gui()
root.mainloop()