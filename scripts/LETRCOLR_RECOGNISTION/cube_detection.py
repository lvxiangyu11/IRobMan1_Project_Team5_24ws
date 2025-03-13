import cv2
import numpy as np

# Load image
img = cv2.imread('pics_rob/E_g_NE.jpg')  # Change to your image filename

# Convert to HSV and apply CLAHE for contrast enhancement
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
v = clahe.apply(v)
hsv = cv2.merge((h, s, v))

# Function to adjust HSV range dynamically based on brightness
def adjust_hsv_range(hsv_img, base_lower, base_upper):
    avg_v = np.mean(hsv_img[:, :, 2])  # Get average brightness
    if avg_v < 100:  
        lower = np.array([base_lower[0], base_lower[1] - 40, base_lower[2] - 40])
        upper = np.array([base_upper[0], base_upper[1] - 20, base_upper[2] - 20])
    elif avg_v > 180:
        lower = np.array([base_lower[0], base_lower[1] + 20, base_lower[2] + 20])
        upper = np.array([base_upper[0], base_upper[1] + 30, base_upper[2] + 30])
    else:
        lower, upper = base_lower, base_upper
    return lower, upper

# Define broad HSV range for green
green_lower = np.array([35, 50, 50])   # Lower bound for green
green_upper = np.array([85, 255, 255]) # Upper bound for green

# Adjust dynamically based on lighting
green_lower, green_upper = adjust_hsv_range(hsv, green_lower, green_upper)

# Create a binary mask for green color
green_mask = cv2.inRange(hsv, green_lower, green_upper)

# Apply edge filtering to remove high-contrast noise
edges = cv2.Canny(img, 50, 150)
green_mask[edges > 0] = 0  # Remove edges from detected areas

# Function to draw bounding boxes around detected objects
def draw_contours(mask, color, name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if area > 500 and 0.8 < aspect_ratio < 1.2:  # Filter only square-like objects
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            return x, y, w, h  # Return the bounding box of the detected cube
    return None

# Detect and label green objects
cube_bbox = draw_contours(green_mask, (0, 255, 0), 'GREEN')

if cube_bbox:
    x, y, w, h = cube_bbox
    roi = img[y:y+h, x:x+w]  # Extract the region of interest (ROI) where the cube is detected

    # Convert the ROI to HSV
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create a binary mask for the letter color
    letter_mask = cv2.inRange(roi_hsv, green_lower, green_upper)

    # Find contours in the letter mask
    contours, _ = cv2.findContours(letter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Filter out small areas
            lx, ly, lw, lh = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (lx, ly), (lx+lw, ly+lh), (255, 0, 0), 2)
            cv2.putText(roi, 'LETTER', (lx, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("Detected letter color: Green")

    # Replace the ROI in the original image with the processed ROI
    img[y:y+h, x:x+w] = roi

# Display the output image
cv2.namedWindow('Green Cube Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Green Cube Detection', 800, 600)
cv2.imshow('Green Cube Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()