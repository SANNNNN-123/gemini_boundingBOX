import streamlit as st
import json
from PIL import Image, ImageDraw, ImageFont
import io
import random
import os
import cv2
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

def process_image_for_digits(image, min_area=100, max_area_percent=10, min_aspect=0.1, max_aspect=4.0):
    """
    Process the image to detect orange regions and find contours of digits.
    Returns the processed image and bounding boxes for digits.
    """
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)

    # Define the range for detecting orange color
    lower_orange = np.array([10, 110, 110])
    upper_orange = np.array([25, 255, 255])
    
    # Create a mask for orange color
    orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    orange_regions = cv2.bitwise_and(opencv_image, opencv_image, mask=orange_mask)
    
    # Convert the orange regions to grayscale
    gray_orange_only = cv2.cvtColor(orange_regions, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to detect digits
    thresh = cv2.threshold(gray_orange_only, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    inverted_thresh = cv2.bitwise_not(thresh)

    # Apply an "opening" morphological operation to disconnect components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(inverted_thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio to identify likely digits
    digit_bboxes = {}
    height, width = opencv_image.shape[:2]
    max_area = width * height * (max_area_percent / 100)
    
    for i, contour in enumerate(contours):
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours: Must be reasonable size
        area = w * h
        aspect_ratio = w / float(h) if h != 0 else 0
        
        # Apply the filters using the parameters
        if area > min_area and area < max_area and min_aspect < aspect_ratio < max_aspect:
            # Draw the contour on the original image for visualization
            cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Convert to the format expected by draw_bounding_boxes
            # [ymin, xmin, ymax, xmax] in 0-1000 scale
            ymin = int(y * 1000 / height)
            xmin = int(x * 1000 / width)
            ymax = int((y + h) * 1000 / height)
            xmax = int((x + w) * 1000 / width)
            
            digit_bboxes[f"digit_{i}"] = [ymin, xmin, ymax, xmax]
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    return processed_image, digit_bboxes

def resize_image(image, max_size=800):
    """
    Resize the image maintaining the aspect ratio. If either dimension exceeds max_size, scale it down.
    """
    width, height = image.size
    if width > height:
        if width > max_size:
            height = int((height * max_size) / width)
            width = max_size
    else:
        if height > max_size:
            width = int((width * max_size) / height)
            height = max_size
    return image.resize((width, height))

def generate_random_color():
    """
    Generate a random color in hexadecimal format.
    """
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def get_font(size=20):
    """
    Get a font object for drawing text. Attempts to load NotoSansCJK-Regular.ttc.
    Falls back to default font if unavailable.
    """
    font_files = ["NotoSansCJK-Regular.ttc"]
    
    for font_file in font_files:
        if os.path.exists(font_file):
            try:
                return ImageFont.truetype(font_file, size)
            except IOError:
                continue
    
    return ImageFont.load_default()

def draw_text_with_outline(draw, text, position, font, text_color, outline_color):
    """
    Draw text with an outline on the image.
    """
    x, y = position
    # Draw outline
    draw.text((x-1, y-1), text, font=font, fill=outline_color)
    draw.text((x+1, y-1), text, font=font, fill=outline_color)
    draw.text((x-1, y+1), text, font=font, fill=outline_color)
    draw.text((x+1, y+1), text, font=font, fill=outline_color)
    # Draw text
    draw.text(position, text, font=font, fill=text_color)

def draw_bounding_boxes(image, bboxes):
    """
    Draw bounding boxes on the image using the coordinates provided in the bboxes dictionary.
    Labels are positioned above the boxes and show just the number.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    font = get_font(20)
    
    # Sort bboxes by x position to number them left to right
    sorted_bboxes = sorted(bboxes.items(), key=lambda x: x[1][1])  # Sort by xmin
    
    for i, (label, bbox) in enumerate(sorted_bboxes, 1):
        color = generate_random_color()
        ymin, xmin, ymax, xmax = [coord / 1000 * dim for coord, dim in zip(bbox, [height, width, height, width])]
        
        # Draw the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # Prepare the label (just the number)
        label_text = str(i)
        
        # Calculate text size
        label_bbox = font.getbbox(label_text)
        label_width = label_bbox[2] - label_bbox[0] + 10  # Adding padding
        label_height = label_bbox[3] - label_bbox[1] + 10  # Adding padding
        
        # Position label above the box
        label_x = xmin + (xmax - xmin - label_width) / 2  # Center horizontally
        label_y = ymin - label_height - 5  # Place above box with 5px gap
        
        # Ensure label stays within image bounds
        label_y = max(0, label_y)
        
        # Draw label background and text
        draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height], fill=color)
        draw_text_with_outline(draw, label_text, (label_x + 5, label_y + 5), font, text_color="white", outline_color="black")
    
    return image

def extract_bounding_boxes(text):
    """
    Extract bounding boxes from the given text, which is expected to be in JSON format.
    """
    try:
        bboxes = json.loads(text)
        return bboxes
    except json.JSONDecodeError:
        import re
        pattern = r'"([^"]+)":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, text)
        return {label: list(map(int, coords)) for label, *coords in matches}

def main():
    st.title("Digit Detection in Orange Regions")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Add slider controls for fine-tuning
    st.sidebar.header("Detection Parameters")
    min_area = st.sidebar.slider("Minimum Area", 50, 500, 100)
    max_area_percent = st.sidebar.slider("Maximum Area (%)", 1, 30, 10)
    min_aspect = st.sidebar.slider("Min Aspect Ratio", 0.1, 1.0, 0.1)
    max_aspect = st.sidebar.slider("Max Aspect Ratio", 1.0, 10.0, 4.0)
    
    if st.button("Process") and uploaded_file is not None:
        try:
            # Validate and open the uploaded image file
            try:
                original_image = Image.open(uploaded_file)
            except IOError:
                st.error("Uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file.")
                return
            
            resized_image = resize_image(original_image)
            
            # Process image for digits in orange regions
            processed_image, digit_bboxes = process_image_for_digits(
                resized_image,
                min_area=min_area,
                max_area_percent=max_area_percent,
                min_aspect=min_aspect,
                max_aspect=max_aspect
            )
            
            # Display the processed image with contours
            st.subheader("Processed Image (Detected Digits)")
            st.image(processed_image, caption="Processed Image", use_container_width=True)
            
            if digit_bboxes:
                image_with_boxes = draw_bounding_boxes(resized_image.copy(), digit_bboxes)
                
                # Display the image with bounding boxes
                st.subheader("Image with Bounding Boxes")
                st.image(image_with_boxes, caption="Image with Bounding Boxes", use_container_width=True)
                
                # Display the detected bounding boxes
                st.subheader("Detected Digit Bounding Boxes")
                st.json(digit_bboxes)
                
                buffered = io.BytesIO()
                image_with_boxes.save(buffered, format="PNG")
                st.download_button(
                    label="Download Image with Bounding Boxes",
                    data=buffered.getvalue(),
                    file_name="image_with_digit_boxes.png",
                    mime="image/png"
                )
            else:
                st.warning("No digits detected in the orange regions.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 