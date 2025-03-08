import cv2
import numpy as np

# Load image and check OpenCV version
print("OpenCV version:", cv2.__version__)
# img = cv2.imread('pics_rob/R_g_y_r_b.jpg')

# # Convert to HSV color space
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# #Define HSV ranges for yellow, red, blue, and green (H: 0-180, S: 0-255, V: 0-255)
# yellow_lower = np.array([22, 150, 150])  # Increase saturation and value thresholds
# yellow_upper = np.array([35, 255, 255])

# # red_lower = np.array([0, 100, 100])  # First red range
# # red_upper = np.array([10, 255, 255])
# # red_lower2 = np.array([170, 100, 100])  # Second red range (as red wraps around hue 0/180)
# # red_upper2 = np.array([180, 255, 255])

# # blue_lower = np.array([90, 50, 70])
# # blue_upper = np.array([130, 255, 255])

# # green_lower = np.array([50, 50, 50])
# # green_upper = np.array([90, 255, 255])

# # Create masks for each color
# yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
# # red_mask1 = cv2.inRange(hsv, red_lower, red_upper)
# # red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
# # red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # Combine both red masks
# # blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
# # green_mask = cv2.inRange(hsv, green_lower, green_upper)

# # Clean up masks using morphological operations
# kernel = np.ones((5,5), np.uint8)
# yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
# # red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
# # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
# # green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

# # Function to draw contours
# def draw_contours(mask, color, name):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 100:  # Filter small noise
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(img, (x, y), (x+w, y+h), color, 10)
#             cv2.putText(img, name, (x, y-5), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 3, color, 7)

# # Draw detected regions on original image
# draw_contours(yellow_mask, (0, 255, 255), 'YELLOW')
# # draw_contours(red_mask, (0, 0, 255), 'RED')
# # draw_contours(blue_mask, (255, 0, 0), 'BLUE')
# # draw_contours(green_mask, (0, 255, 0), 'GREEN')

# # Display results
# cv2.namedWindow('Cube Detection with Color', cv2.WINDOW_NORMAL)  # Allows resizing
# cv2.resizeWindow('Cube Detection with Color', 800, 600)  # Set width and height

# cv2.imshow('Cube Detection with Color', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load image
# # img = cv2.imread('pics_rob/P_y.jpg')
# img = cv2.imread('pics_rob/R_g_y_r_b_NW.jpg')

# # Convert to HSV color space
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# hsv = cv2.GaussianBlur(hsv, (5,5), 0)  # Reduce noise

# # Refined HSV ranges for yellow
# # yellow_lower = np.array([22, 150, 150])  # Adjusted thresholds
# # yellow_upper = np.array([35, 255, 255])
# yellow_lower = np.array([20, 80, 80])  # Lower value
# yellow_upper = np.array([40, 255, 255])

# # Create a binary mask
# yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

# # Apply erosion and morphological closing
# kernel = np.ones((5,5), np.uint8)
# yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)  # Reduce false detections
# yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

# # Function to draw contours with a higher area threshold
# def draw_contours(mask, color, name):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:  # Increase threshold to avoid multiple labels
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)  # Reduce box thickness
#             cv2.putText(img, name, (x, y-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)  # Reduce font size

# # Detect and label yellow objects
# draw_contours(yellow_mask, (0, 255, 255), 'YELLOW')

# # Display results
# cv2.namedWindow('Cube Detection with Color', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Cube Detection with Color', 800, 600)
# cv2.imshow('Cube Detection with Color', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load image
img = cv2.imread('pics_rob/R_y_N.jpg')

# Convert to HSV and apply CLAHE
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
v = clahe.apply(v)
hsv = cv2.merge((h, s, v))

# Dynamic HSV Adjustment
def adjust_hsv_range(hsv_img, base_lower, base_upper):
    avg_v = np.mean(hsv_img[:, :, 2])
    if avg_v < 100:  
        lower = np.array([base_lower[0], base_lower[1] - 40, base_lower[2] - 40])
        upper = np.array([base_upper[0], base_upper[1] - 20, base_upper[2] - 20])
    elif avg_v > 180:
        lower = np.array([base_lower[0], base_lower[1] + 20, base_lower[2] + 20])
        upper = np.array([base_upper[0], base_upper[1] + 30, base_upper[2] + 30])
    else:
        lower, upper = base_lower, base_upper
    return lower, upper

yellow_lower = np.array([10, 100, 100])
yellow_upper = np.array([35, 255, 255])
yellow_lower, yellow_upper = adjust_hsv_range(hsv, yellow_lower, yellow_upper)

# Create mask
yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

# Apply Edge Filtering
edges = cv2.Canny(img, 50, 150)
yellow_mask[edges > 0] = 0

# Draw Contours
def draw_contours(mask, color, name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if area > 500 and 0.8 < aspect_ratio < 1.2:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

draw_contours(yellow_mask, (0, 255, 255), 'YELLOW')

# Display results
cv2.namedWindow('Cube Detection with Color', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cube Detection with Color', 800, 600)
cv2.imshow('Cube Detection with Color', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
