import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Load image 
image = cv2.imread('./images/test5.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
small_image = cv2.resize(image, (300, 200))

#feature vector combining color and spatial position
h, w, c = small_image.shape
X = []

for i in range(h):
    for j in range(w):
        r, g, b = small_image[i, j]
        X.append([r, g, b, i, j])

X = np.array(X,dtype=np.float32)

# Estimate bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=500)
print(f"Estimated bandwidth: {bandwidth}")

#Mean Shift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_

# Reshape segmented image
segmented_image = np.reshape(labels, (h, w))

# Display output
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(small_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='nipy_spectral')
plt.title('Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()

