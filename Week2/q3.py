import cv2
import numpy as np

def harris_corner_detector(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not read the image {image_path}.")
        return None

    # Step 1: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 1)

    # Step 2: Compute Gradients
    Ix = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Step 3: Compute Gradient Products
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Step 4: Apply Gaussian Smoothing
    Ixx = cv2.GaussianBlur(Ixx, (5, 5), 1)
    Iyy = cv2.GaussianBlur(Iyy, (5, 5), 1)
    Ixy = cv2.GaussianBlur(Ixy, (5, 5), 1)

    # Step 5: Compute Harris Response
    k = 0.04  # Empirical constant
    detM = (Ixx * Iyy) - (Ixy * Ixy)
    traceM = Ixx + Iyy
    harris_response = detM - k * (traceM ** 2)

    # Step 6: Thresholding
    threshold = 0.01 * harris_response.max()
    harris_corners = np.zeros_like(harris_response)
    harris_corners[harris_response > threshold] = 255
    
    # Step 7: Mark the corners on the original image
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result[harris_corners == 255] = [0, 0, 255]  # Red color for corners
    
    return result
# Apply Harris Corner Detector on both images
image1_result = harris_corner_detector("./images/227092.png")
image2_result = harris_corner_detector("./images/19021.png")

# Show results
if image1_result is not None and image2_result is not None:
    cv2.imshow("Harris Corners - Image 1", image1_result)
    cv2.imshow("Harris Corners - Image 2", image2_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
