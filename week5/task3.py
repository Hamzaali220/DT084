import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two consecutive images (convert to grayscale)
img1 = cv2.imread("./images/img2.jpg")  # First image
img2 = cv2.imread("./images/img1.jpg")  # Second image with object slightly moved
testImage = img2.copy()
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(21, 21),
                 maxLevel=5,  
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))
# Detect good features
features = cv2.goodFeaturesToTrack(gray1, 
                                   maxCorners=500, 
                                   qualityLevel=0.2,  
                                   minDistance=5, 
                                   blockSize=7)

# Compute optical flow using Lucas-Kanade method
new_points, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, features, None, **lk_params)

# Filter good points
good_old = features[status == 1]
good_new = new_points[status == 1]

# Visualize the optical flow
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    output = cv2.arrowedLine(testImage, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

# Show the result
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image 1")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title("Original Image 2")
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axes[2].set_title("Optical Flow Vectors (Lucas-Kanade)")
axes[2].axis('off')

plt.show()
