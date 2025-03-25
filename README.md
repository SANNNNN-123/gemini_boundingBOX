# Advanced Digit Detection System

This Streamlit application combines two powerful detection methods:
1. Orange Digit Detection - Specialized detection for digits in orange regions
2. Gemini API Detection - General purpose object detection using Google's Gemini API

You can use either method independently or combine them for comprehensive detection results.



## Features

- **Multiple Detection Methods:**
  - Orange Digit Detection: Specialized algorithm for detecting digits in orange-colored regions
  - Gemini API Detection: Flexible object detection using Google's Gemini API
  - Combined Detection: Use both methods simultaneously for comprehensive results

- **Image Processing:**
  - Upload images in JPG, JPEG, or PNG format
  - Automatic image resizing for optimal processing
  - Customizable detection parameters
  - Real-time visualization of detection results

- **Orange Digit Detection Features:**
  - Adjustable parameters for area, aspect ratio, and color sensitivity
  - Automatic numbering of detected digits from left to right
  - Clear visualization of detected regions with colored bounding boxes

- **Gemini API Features:**
  - Customizable detection prompts
  - Multiple Gemini model options
  - Flexible object detection capabilities

- **Output Options:**
  - Download processed images with bounding boxes
  - View JSON data of detected coordinates
  - Separate visualization for each detection method

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.9 or later
- A valid Gemini API key (required for Gemini API detection)

### Required Python Packages

Install the necessary Python packages using `pip`:

```bash
pip install streamlit Pillow google-generativeai opencv-python numpy
```

## Running the Application

```bash
streamlit run bbox.py 

or

streamlit run main.py 

```
