import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import easyocr
import cv2

class ImageRecognizer:
    def __init__(self, top_dir="cubes/"):
        self.top_dir = top_dir
        self.cubes_images = {}
        self.rotated_test_images = {}
        self._load_images()
        
    def _load_images(self):
        """Load all images from the folder"""
        first_right = True
        for i, folder in enumerate(sorted(os.listdir(self.top_dir))):
            folder_path = os.path.join(self.top_dir, folder)
            if os.path.isdir(folder_path):
                self.cubes_images[i] = {}
                for file in sorted(os.listdir(folder_path)):
                    # if file.endswith(".png"):
                    if file.endswith(".png") :
                        if ("right" in file.lower()):
                            if first_right:
                                first_right = False
                                # read one right 
                            else:
                                continue
                        
                        file_name = os.path.splitext(file)[0]
                        file_path = os.path.join(folder_path, file)
                        image = Image.open(file_path)
                        self.cubes_images[i][file_name] = image

    def rotate_image(self, image, angles=[0, 90, 180, 270]):
        """Rotate the input image"""
        rotated_images = {}
        for angle in angles:
            rotated_images[angle] = np.array(image.rotate(angle, resample=Image.Resampling.BICUBIC))
        return rotated_images

    def recognize_image(self, test_image_np):
        """Recognize the input image and return the most similar image's related information"""
        test_image = Image.fromarray(test_image_np)
        rotated_test_images = self.rotate_image(test_image)

        similarity_results = []

        for i in self.cubes_images:
            for j in self.cubes_images[i]:
                current_image = np.array(self.cubes_images[i][j])
                height, width, _ = test_image_np.shape
                current_image_resized = np.array(
                    Image.fromarray(current_image).resize((width, height), Image.Resampling.LANCZOS)
                )

                best_similarity = -1
                best_angle = 0

                for angle, rotated_test_image in rotated_test_images.items():
                    total_similarity = 0
                    for c in range(3):  # Calculate for RGB channels separately
                        similarity, _ = ssim(current_image_resized[:, :, c], rotated_test_image[:, :, c], full=True)
                        total_similarity += similarity

                    average_similarity = total_similarity / 3
                    if average_similarity > best_similarity:
                        best_similarity = average_similarity
                        best_angle = angle

                similarity_results.append((best_similarity, i, j, best_angle))

        # Sort the results by similarity in descending order
        similarity_results.sort(key=lambda x: x[0], reverse=True)

        return similarity_results[:10]
    
    def match_features_with_orb(self, test_image_np):
        """
        Use feature matching (ORB) algorithm to recognize the input color image, returning the most similar image's related information.
        """
        # Convert the input image to OpenCV format (BGR)
        # test_image = cv2.cvtColor(test_image_np, cv2.COLOR_RGB2BGR)
        test_image = test_image_np
        # Create ORB feature detector (adjust parameters)
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)

        # Compute feature points and descriptors for the input test image
        keypoints_test, descriptors_test = orb.detectAndCompute(test_image, None)

        if descriptors_test is None:
            raise ValueError("No descriptors found for the test image.")

        # Store matching results
        matching_results = []

        # Iterate through each image in the database
        for i in self.cubes_images:
            for j in self.cubes_images[i]:
                # Get the current image and resize it to match the input image
                current_image_np = np.array(self.cubes_images[i][j])
                if current_image_np.shape[2] == 4:  # Detect 4 channels
                    current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_RGBA2BGR)
                if len(current_image_np.shape) != 3 or current_image_np.shape[2] != 3:
                    print(f"Invalid image format for {i}-{j}. Skipping.")
                    continue

                current_image_resized = cv2.resize(
                    current_image_np, 
                    (test_image_np.shape[1], test_image_np.shape[0]), 
                    interpolation=cv2.INTER_AREA
                )

                # Image enhancement
                enhanced_image = cv2.equalizeHist(cv2.cvtColor(current_image_resized, cv2.COLOR_BGR2GRAY))
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

                # Compute feature points and descriptors for the current image
                keypoints_current, descriptors_current = orb.detectAndCompute(enhanced_image, None)

                if descriptors_current is None:
                    print(f"No descriptors found for image {i}-{j}. Skipping.")
                    continue

                # Feature point matching
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(descriptors_test, descriptors_current)

                # Sort the matches by distance, take the top 50 best matches
                matches = sorted(matches, key=lambda x: x.distance)[:50]

                # Calculate matching score (more matches and shorter distances lead to higher scores)
                match_score = sum([1 / (match.distance + 1e-5) for match in matches])

                # Save the matching results
                matching_results.append((match_score, i, j, len(matches)))

        # Sort the results by matching score in descending order
        matching_results.sort(key=lambda x: x[0], reverse=True)

        return matching_results[:10]
    

    def match_features_with_sift(self, test_image_np):
        """
        Use SIFT feature matching algorithm to recognize the input color image, returning the most similar image's related information.
        """
        # Use the input image directly as the test image
        test_image = test_image_np

        # Create SIFT feature detector (adjust parameters)
        sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=20, sigma=1.0)

        # Extract feature points and descriptors from the test image
        keypoints_test, descriptors_test = sift.detectAndCompute(test_image, None)

        if descriptors_test is None:
            raise ValueError("No descriptors found for the test image.")

        # Store matching results
        matching_results = []

        # Iterate through each image in the database
        for i in self.cubes_images:
            for j in self.cubes_images[i]:
                # Get the current image and resize it to match the input image
                current_image = np.array(self.cubes_images[i][j])

                # Check the channel count and convert to 3 channels (if necessary)
                if current_image.shape[2] == 4:  # Convert RGBA to BGR
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGBA2BGR)

                if len(current_image.shape) != 3 or current_image.shape[2] != 3:
                    print(f"Invalid image format for {i}-{j}. Skipping.")
                    continue

                current_image_resized = cv2.resize(
                    current_image,
                    (test_image.shape[1], test_image.shape[0]),
                    interpolation=cv2.INTER_AREA
                )

                # Image enhancement: contrast enhancement or sharpening (based on need)
                gray = cv2.cvtColor(current_image_resized, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(gray)
                enhanced_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

                # Extract feature points and descriptors from the current image
                keypoints_current, descriptors_current = sift.detectAndCompute(enhanced_image, None)

                if descriptors_current is None:
                    print(f"No descriptors found for image {i}-{j}. Skipping.")
                    continue

                # Feature point matching
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # SIFT uses L2 distance
                matches = bf.match(descriptors_test, descriptors_current)

                # Sort the matches by distance, take the top 50 best matches
                matches = sorted(matches, key=lambda x: x.distance)[:50]

                # Calculate matching score (more matches and shorter distances lead to higher scores)
                match_score = sum([1 / (match.distance + 1e-5) for match in matches])

                # Save the matching results
                matching_results.append((match_score, i, j, len(matches)))

        # Sort the results by matching score in descending order
        matching_results.sort(key=lambda x: x[0], reverse=True)

        return matching_results[:10]

    def get_image_from_result(self, result, show=False):
        """
        Extract and return the corresponding image from the result (including rotated images).
        :param result: A tuple containing (score, folder, image_name, angle).
        :return: Return the image related to the recognition result (numpy array), including the radians of rotation.
        """
        score, folder, image_name, angle = result
        
        # Get the original image
        original_image = self.cubes_images[folder][image_name]
        
        # Convert the angle to radians
        angle_radians = angle * np.pi / 180.0

        if show:
            plt.figure(figsize=(6, 6))
            plt.imshow(original_image)
            plt.title(f"Image: {image_name}\nAngle: {angle_radians:.2f} radians")
            plt.axis("off")
            plt.show()        

        # Return the image and radians
        return np.array(original_image), angle_radians

    def display_results(self, test_image_np, results, max_display=10):
        """Visualize the recognition results"""
        width, height, _ = test_image_np.shape
        plt.figure(figsize=(15, 6))

        # Display the test image
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(test_image_np)
        plt.title("Test Image")
        plt.axis("off")

        # Display significant images
        for idx, (score, folder, image_name, angle) in enumerate(results[:max_display]):
            current_image = np.array(self.cubes_images[folder][image_name])
            current_image_resized = np.array(
                Image.fromarray(current_image).resize((width, height), Image.Resampling.LANCZOS)
            )

            plt.subplot(1, len(results) + 1, idx + 2)
            plt.imshow(current_image_resized)
            plt.title(f"Folder {folder}\nImage {image_name}\nAngle {angle}°\nSim: {score:.4f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def selected_best_based_on_CNN(self, results, test_image_np):
        """
        Select the best match from the recognition results.
        :param results: A list of results from the recognize_image() method
        :param test_image_np: Numpy array of the test image
        :return: The final matched result (similarity, folder number, file name, rotation angle)
        """
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'], gpu=True)

        # Convert the test image to PIL image
        test_image = Image.fromarray(test_image_np)

        # Initialize the best similarity result
        best_result = results[0]

        for result in results[:10]:  # Only try the top 10 results
            _, folder, image_name, angle = result

            # Get the original image and rotate by angle
            original_image = self.cubes_images[folder][image_name]
            rotated_image = original_image.rotate(angle, resample=Image.Resampling.BICUBIC)
            rotated_image_np = np.array(rotated_image)

            # Rotate the test image by angle
            test_rotated = test_image.rotate(angle, resample=Image.Resampling.BICUBIC)
            test_rotated_np = np.array(test_rotated)

            # OCR text extraction
            test_text = reader.readtext(test_rotated_np, detail=0)
            recognized_text = reader.readtext(rotated_image_np, detail=0)
            # print(test_text, recognized_text)

            if not test_text or not recognized_text:
                continue  # Skip if OCR result is empty

            # Convert to string
            test_text_str = " ".join(test_text)
            recognized_text_str = " ".join(recognized_text)

            # Check if it contains English characters
            if not test_text_str.isalpha() or not recognized_text_str.isalpha():
                continue

            # If the text of the test image matches the current candidate image's text, return this result
            if test_text_str.strip().lower() == recognized_text_str.strip().lower():
                # print("find it !")
                return result

        # If no match found, return the highest similarity result
        return best_result


if __name__ == "__main__":
    # Example usage
    recognizer = ImageRecognizer(top_dir="/opt/ros_ws/src/franka_zed_gazebo/scripts/mycode/2_perception/cubes/")
    test_image_np = np.array(Image.open("/opt/ros_ws/src/franka_zed_gazebo/scripts/mycode/2_perception/test_0.png"))

    # Recognize the image and display results
    results = recognizer.recognize_image(test_image_np)
    print(results[0])
    recognizer.display_results(test_image_np, results[:10])

    results = recognizer.match_features_with_orb(test_image_np)
    print(results[0])
    recognizer.display_results(test_image_np, results[:10])

    results = recognizer.match_features_with_sift(test_image_np)
    print(results[0])

    # Get and display the first result's image
    image_from_result = recognizer.get_image_from_result(results[0])
    # plt.imshow(image_from_result)
    # plt.title(f"Result Image: {results[0][2]} at {results[0][3]}°")
    # plt.axis("off")
    # plt.show()

    recognizer.display_results(test_image_np, results[:10])

    # Use CNN and text to filter the best match
    # best_result = recognizer.selected_best_based_on_CNN(results, test_image_np)
    # print("Best Result:", best_result)
    # img, angle = recognizer.get_image_from_result(best_result, True)
    # print(angle)
