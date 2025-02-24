import cv2
import numpy as np
import matplotlib.pyplot as plt

def feature_matching_kdtree_ransac(image1_path, image2_path):
    # Load images in grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Could not read images.")
        return

    # Step 1: Detect Features using ORB
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Step 2: Use KD-Tree with FLANN Matcher
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the two nearest neighbors for each descriptor
    matches = flann.knnMatch(des1, des2, k=2)

    # Step 3: Apply Lowe’s Ratio Test to filter weak matches
    good_matches = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe’s ratio test
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # Step 4: Apply RANSAC to remove outliers
    if len(pts1) >= 4:  # RANSAC needs at least 4 points
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        mask = mask.ravel().tolist()
    else:
        print("Not enough matches for RANSAC.")
        return

    # Step 5: Draw only inlier matches
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i] == 1]
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display result
    plt.figure(figsize=(12, 8))  # Adjust width and height
    plt.imshow(match_img)
    plt.axis('off')  # Hide axis for better visualization
    plt.show()

# Example usage
feature_matching_kdtree_ransac("./images/img1.jpg", "./images/img2.jpg")
