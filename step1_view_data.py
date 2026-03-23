import cv2
import os

print("Waking up OpenCV...")

# 1. Define where your dataset is saved
# We are going to look inside folder '1', which contains 'Speed Limit 30' signs
folder_path = r"dataset\Train\1" 

# 2. Get a list of all the image files inside that folder
image_names = os.listdir(folder_path)

# 3. Grab the name of the very first image in that list
first_image_name = image_names[0] 

# 4. Create the exact Windows path to that specific image file
image_path = os.path.join(folder_path, first_image_name)
print(f"Successfully found image at: {image_path}")

# 5. Use OpenCV to read the image file from your hard drive into your computer's memory
img = cv2.imread(image_path)

# 6. Show the image in a popup window on your screen
cv2.imshow("My First AI Road Sign", img)

# 7. Tell Python to keep the window open until you press any key
print("SUCCESS! Press any key on your keyboard to close the image window...")
cv2.waitKey(0) 

# 8. Clean up and close the window safely
cv2.destroyAllWindows()