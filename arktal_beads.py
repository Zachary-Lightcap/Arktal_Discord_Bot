# Imports
import pandas as pd
import numpy as np
import cv2
import colour
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

# Initialize global vars
alpha = None

# Import Arktal Beads Dataset
df = pd.read_excel("Arktal Beads.xlsx", usecols="A:D")

# Extract values
rgb_values = df[['Red', 'Green', 'Blue']].values

# Reshape to match opencv format
rgb_values = rgb_values.reshape(-1, 1, 3).astype(np.uint8)

# Convert to LAB
lab_values = cv2.cvtColor(rgb_values, cv2.COLOR_RGB2LAB)

# Add LAB values to dataframe
df['L'] = lab_values[:, 0, 0]
df['A'] = lab_values[:, 0, 1]
df['B'] = lab_values[:, 0, 2]


# Add a test to check the values of the RGB so they make sense in the documentation
def test_df():
    red_good = ((0 <= df['Red']) & (df['Red'] <= 255)).all()
    green_good = ((0 <= df['Green']) & (df['Green'] <= 255)).all()
    blue_good = ((0 <= df['Blue']) & (df['Blue'] <= 255)).all()

    if not red_good or not green_good or not blue_good:
        print("Values are out of range!")
        print("R: ", red_good)
        print("G: ", green_good)
        print("B: ", blue_good)
    else:
        print("All RGB values are in range!")


# Metric for finding distance between colors
def delta_e_distance(pixel1, pixel2):
    # Rescale pixel1 from OpenCV's 0-255 range to standard CIELAB ranges
    L1_true = pixel1[0] * 100 / 255
    a1_true = pixel1[1] - 128
    b1_true = pixel1[2] - 128
    scaled_pixel1 = np.array([L1_true, a1_true, b1_true])

    # Rescale pixel2 from OpenCV's 0-255 range to standard CIELAB ranges
    L2_true = pixel2[0] * 100 / 255
    a2_true = pixel2[1] - 128
    b2_true = pixel2[2] - 128
    scaled_pixel2 = np.array([L2_true, a2_true, b2_true])

    # Use colour.delta_e with method='CIE2000' for Delta E 2000 calculation
    return colour.delta_E(scaled_pixel1, scaled_pixel2, method='CIE2000')

# X and y split
X = df[['L', 'A', 'B']]
y = df['Color Code']

# Initialize the two models w/ delta_e_distance metric
model = KNeighborsClassifier(n_neighbors=1, metric=delta_e_distance)
larger_model = KNeighborsClassifier(n_neighbors=5, metric=delta_e_distance)
model.fit(X, y)
larger_model.fit(X, y)

# Take in an image and adjust it for further processing
def preprocess_image(img):
    # Get alpha value
    global alpha
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)

    # Convert image to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Add back in the alpha value
    img = cv2.merge((img, alpha))

    # Resize the image
    img = cv2.resize(img, (16, 16), cv2.INTER_LINEAR)

    return img

# Function to find unique pixels in an image
def find_unique_pixels(img):
    # Initialize the unique row with the first row
    unique_pixels = np.unique(img[0], axis=0)

    # Loop through and find unique pixels per row
    for i in range(1, img.shape[0]):
        new_pixels = np.unique(img[i], axis=0)
        unique_pixels = np.concatenate([unique_pixels, new_pixels])

    # Pull out all unique pixels
    unique_pixels = np.unique(unique_pixels, axis=0)

    # Remove any pixels with transparency of 0
    unique_pixels = unique_pixels[unique_pixels[:, -1] != 0]

    return unique_pixels


def predict_image(img):
    # Convert back to RGB
    color_code_to_rgb = df.set_index('Color Code')[['Red', 'Green', 'Blue']].apply(lambda x: x.tolist(), axis=1).to_dict()

    # Initialize variables of predicted image
    height, width, _ = img.shape
    predicted_color_codes_array = np.empty((height, width), dtype=object)
    predicted_img_rgba = np.zeros((height, width, 4), dtype=np.uint8)

   # Perform the prediction loop
    for r in range(height):
        for c in range(width):
            pixel_lab_alpha = img[r, c]
            pixel_lab = pixel_lab_alpha[:3]  # L, a, b values
            alpha_val = pixel_lab_alpha[3]  # Alpha value

            if alpha_val == 0:
                # Keep the pixel transparent if original alpha is 0
                predicted_img_rgba[r, c] = [0, 0, 0, 0]
                predicted_color_codes_array[r, c] = 'Transparent'  # Assign a placeholder for transparent pixels
            else:
                # Predict the color code for non-transparent pixels
                predicted_color_code = model.predict(np.array([pixel_lab]))[0]
                predicted_color_codes_array[r, c] = predicted_color_code

                # Get RGB values for the predicted color code
                predicted_rgb = color_code_to_rgb[predicted_color_code]

                pixel = predicted_rgb + [alpha_val]

                if any(v > 255 or v < 0 for v in pixel):
                    print("Found a bad pixel!")
                    print("predicted_rgb:", predicted_rgb)
                    print("alpha:", alpha_val)
                    print("pixel:", pixel)
                # Assign to the new image with original alpha
                predicted_img_rgba[r, c] = predicted_rgb + [alpha_val]



    # Convert to bgra for CV2
    predicted_img_bgra = cv2.cvtColor(predicted_img_rgba, cv2.COLOR_RGBA2BGRA)

    # Resize the image
    block_size = 60
    scaled_image = cv2.resize(predicted_img_bgra, (width * block_size, height * block_size),
                              interpolation=cv2.INTER_NEAREST)

    # Second loop to add on the predicted label text
    for r in range(height):
        for c in range(width):
            if predicted_color_codes_array[r, c] != 'Transparent':
                # Add text label
                (text_w, text_h), baseline = cv2.getTextSize(
                    predicted_color_codes_array[r, c],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    2
                )

                x = c * block_size + (block_size - text_w) // 2
                y = r * block_size + (block_size + text_h) // 2

                cv2.putText(
                    scaled_image,
                    predicted_color_codes_array[r, c],
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_8
                )

    return scaled_image

def analyze_image(img):
    # Run through and process/analyze the image
    img = preprocess_image(img)
    print("Image preprocessing has completed")
    predicted_img = predict_image(img)
    print("Image prediction has completed")
    return predicted_img







