import cv2
import os
import numpy as np

# 1. Setup our variables
data_path = r"dataset\Train"
num_classes = 43         # Total number of different road signs
image_size = (32, 32)    # The uniform size we want for our AI

images = [] # This will hold all our picture data
labels = [] # This will hold the answers (e.g., "Folder 1")

print("Starting Data Preprocessing... This will process thousands of images, please wait!")

# 2. Loop through all 43 folders (0 to 42)
for class_id in range(num_classes):
    folder_path = os.path.join(data_path, str(class_id))
    
    # Get all the image files in the current folder
    image_names = os.listdir(folder_path)
    print(f"Loading Folder {class_id} / 42...")
    
    # 3. Process every image inside this folder
    for img_name in image_names:
        img_path = os.path.join(folder_path, img_name)
        
        # Read the image
        img = cv2.imread(img_path)
        
        # Sometime files get corrupted, this skips any empty files
        if img is None:
            continue
            
        # Resize the image to 32x32
        img = cv2.resize(img, image_size)
        
        # Add the image and its label to our lists
        images.append(img)
        labels.append(class_id)

# 4. Convert our standard lists into NumPy math arrays
print("\nConverting data into NumPy arrays...")
images = np.array(images)
labels = np.array(labels)

# 5. Normalize the pixel values to be between 0 and 1
print("Normalizing colors...")
images = images / 255.0

print("\nSUCCESS! Data is perfectly formatted for AI.")
print(f"Total images loaded: {len(images)}")
print(f"Shape of image array: {images.shape}")