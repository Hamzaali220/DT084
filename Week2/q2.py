import cv2
import numpy as np

# Load the image
image_path = "./images/19021.png"  # Change this if needed
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(image, None)

# Draw keypoints on the image
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Display the result
cv2.imshow("ORB Feature Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
