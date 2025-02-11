import cv2
import numpy as np
import matplotlib.pyplot as plt

def high_pass_filter(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not read the image.")
        return

    # Get image dimensions
    rows, cols = img.shape

    # Apply FFT and shift zero frequency to center
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    # Create a High-Pass Filter
    mask = np.ones((rows, cols), np.uint8)
    center_x, center_y = rows // 2, cols // 2
    radius = 30  # Controls how much low frequency is removed
    mask[center_x - radius:center_x + radius, center_y - radius:center_y + radius] = 0

    # Apply the high-pass filter
    dft_filtered = dft_shift * mask

    # Shift back and apply inverse FFT
    dft_ishift = np.fft.ifftshift(dft_filtered)
    img_filtered = np.fft.ifft2(dft_ishift)
    img_filtered = np.abs(img_filtered)

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title("Original Image")
    plt.subplot(1, 2, 2), plt.imshow(img_filtered, cmap='gray'), plt.title("High-Pass Filtered Image")
    plt.show()

# Run the function with an image
high_pass_filter("./images/227092.png")
