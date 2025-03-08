import cv2
import numpy as np

# Loading the image
img = cv2.imread('pics_rob/R_r_SE.jpg')

# Converting it to HSV and applying CLAHE
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

# Defining HSV ranges for red (two ranges due to hue wrapping)
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 100])
red_upper2 = np.array([180, 255, 255])

# Adjusting red HSV dynamically based on lighting
red_lower1, red_upper1 = adjust_hsv_range(hsv, red_lower1, red_upper1)
red_lower2, red_upper2 = adjust_hsv_range(hsv, red_lower2, red_upper2)

# Creating masks for both red ranges and combining them
red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # Combine both red masks

# Applying Edge Filtering
edges = cv2.Canny(img, 50, 150)
red_mask[edges > 0] = 0

# Drawing Contours
def draw_contours(mask, color, name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if area > 500 and 0.8 < aspect_ratio < 1.2:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)

# Detecting red objects
draw_contours(red_mask, (0, 0, 255), 'RED')

# Displaying results
cv2.namedWindow('Red Cube Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Red Cube Detection', 800, 600)
cv2.imshow('Red Cube Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
