import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks

# Given points for set (a)
points_a = np.array([(2,2), (3,1.5), (6,0)])

# Hough Transform Function
def hough_transform(points):
    theta_range = np.linspace(-np.pi / 2, np.pi / 2, 360)  # Theta from -90 to 90 degrees
    r_values = []
    
    plt.figure(figsize=(12, 5))
    
    # Plot Hough space
    plt.subplot(1, 2, 1)
    for (x, y) in points:
        r = x * np.cos(theta_range) + y * np.sin(theta_range)
        r_values.append(r)
        plt.plot(theta_range, r, label=f'({x}, {y})')
    
    plt.xlabel(r'$\theta$ (radians)')
    plt.ylabel(r'$r$')
    plt.title('Hough Space')
    plt.legend()
    
    # Convert points to a binary edge map (for Hough transform)
    edge_map = np.zeros((10, 10))  # Creating a small grid
    for (x, y) in points:
        edge_map[int(y), int(x)] = 1
    
    # Compute the Hough transform using skimage
    h, theta, d = hough_line(edge_map)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # Take the strongest detected line
    best_theta = angles[0]
    best_r = dists[0]
    
    # Convert to Cartesian space: y = (-cos(theta)/sin(theta)) * x + (r/sin(theta))
    x_vals = np.array([0, 7])
    y_vals = (-np.cos(best_theta) / np.sin(best_theta)) * x_vals + (best_r / np.sin(best_theta))
    
    # Plot Cartesian space with best-fit line
    plt.subplot(1, 2, 2)
    plt.scatter(points[:, 0], points[:, 1], color='red', label='Points')
    plt.plot(x_vals, y_vals, 'b-', label='Best Fit Line')
    plt.xlim(0, 7)
    plt.ylim(-1, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cartesian Space')
    plt.legend()
    plt.show()

# Run for set A
hough_transform(points_a)
# Run for set B
points_b = np.array([(2,2), (5,3), (6,0)])
hough_transform(points_b)