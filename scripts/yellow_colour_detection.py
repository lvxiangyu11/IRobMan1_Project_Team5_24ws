import cv2
import numpy as np

# Loading image
img = cv2.imread('pics_rob/R_y_N.jpg')

# Converting to HSV and applying CLAHE
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

# Creating mask
yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

# Applying Edge Filtering
edges = cv2.Canny(img, 50, 150)
yellow_mask[edges > 0] = 0

# Drawing Contours
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

# Displaying results
cv2.namedWindow('Cube Detection with Color', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cube Detection with Color', 800, 600)
cv2.imshow('Cube Detection with Color', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
