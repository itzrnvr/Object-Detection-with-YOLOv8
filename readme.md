# Object Detection with YOLOv8

This project demonstrates real-time object detection using the YOLOv8 (You Only Look Once version 8) model. The script processes a video file, identifies objects in each frame, and displays the video with bounding boxes, class labels, and confidence scores for the detected objects.

## Features

*   **Real-time Object Detection:** Utilizes the powerful YOLOv8 model for efficient object detection.
*   **Multiple Object Classes:** Can detect a wide range of objects based on the COCO dataset classes (e.g., person, car, bicycle, dog, etc.).
*   **Visual Feedback:** Draws bounding boxes around detected objects.
*   **Information Display:** Shows the class name and confidence score for each detected object.
*   **Performance Monitoring:** Calculates and prints the Frames Per Second (FPS) to the console.
*   **Video File Processing:** Currently configured to process a local video file.

## Directory Structure

```
.
├── CarCounter/
│   └── Carcounter.py       # Main Python script for object detection
├── Video/
│   ├── bikes.mp4           # Sample video file
│   ├── cars.mp4            # Sample video file
│   ├── motorbikes-1.mp4    # Sample video file
│   ├── people.mp4          # Sample video file (currently used by the script)
│   ├── ppe-1-1.mp4         # Sample video file
│   ├── ppe-2-1.mp4         # Sample video file
│   └── ppe-3-1.mp4         # Sample video file
├── Yolo-Weights/
│   └── yolov8l.pt          # Pre-trained YOLOv8 Large model weights
└── readme.md               # This README file
```

## Prerequisites

Before running the script, ensure you have Python installed. You will also need the following Python libraries:

*   **ultralytics:** The official package for YOLO models.
*   **opencv-python:** For video processing and display.
*   **cvzone:** For utility functions like drawing styled rectangles and text.
*   **math:** Standard Python math library (used for ceiling confidence values).
*   **time:** Standard Python time library (used for FPS calculation).

## Setup and Installation

This project uses `uv` for package management, which is a fast Python package installer and resolver, written in Rust.

1.  **Clone the Repository (Optional):**
    If you have this project as a Git repository, clone it:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install `uv`:**
    If you don't have `uv` installed, you can install it using `pipx` (recommended) or `pip`:
    *   Using `pipx`:
        ```bash
        pipx install uv
        ```
    *   Using `pip`:
        ```bash
        pip install uv
        ```
    Verify the installation:
    ```bash
    uv --version
    ```

3.  **Create and Activate a Virtual Environment:**
    It is highly recommended to use a virtual environment. Navigate to the project root directory and run:
    *   Create the virtual environment (e.g., named `.venv`):
        ```bash
        uv venv .venv
        ```
    *   Activate the virtual environment:
        *   On macOS/Linux:
            ```bash
            source .venv/bin/activate
            ```
        *   On Windows (PowerShell):
            ```powershell
            .venv\Scripts\Activate.ps1
            ```
        *   On Windows (CMD):
            ```batch
            .venv\Scripts\activate.bat
            ```

4.  **Install Dependencies:**
    With the virtual environment activated, install the required Python libraries from [`requirements.txt`](requirements.txt):
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **YOLOv8 Model Weights:**
    The project expects the YOLOv8 Large model weights (`yolov8l.pt`) to be present in the `Yolo-Weights/` directory. This file is included in the provided project structure. If you need to download it manually, you can usually obtain it when `ultralytics` is first used or from the official Ultralytics YOLO repository.

## Usage

1.  **Navigate to the Project Root Directory:**
    Open your terminal or command prompt and navigate to the main project directory (the one containing `CarCounter/`, `Video/`, etc.).

2.  **Run the Script:**
    Execute the Python script using the following command:
    ```bash
    python CarCounter/Carcounter.py
    ```

3.  **Output:**
    *   A window titled "Image" will open, displaying the video stream with detected objects highlighted.
    *   The FPS will be printed to the console.
    *   Press 'q' or the ESC key (depending on your OpenCV setup) while the "Image" window is active to close it and terminate the script.

4.  **Changing the Input Video:**
    The script currently uses `../Video/people.mp4` as its input. To change the video source, modify line 11 in [`CarCounter/Carcounter.py`](CarCounter/Carcounter.py:11):
    ```python
    # cap = cv2.VideoCapture(1) # For webcam input (currently commented out and overridden)
    # ...
    cap = cv2.VideoCapture("../Video/your_video_file.mp4") # Change this line
    ```
    Replace `"../Video/your_video_file.mp4"` with the path to your desired video file (relative to the `CarCounter` directory, or provide an absolute path).

5.  **Using Webcam Input (Experimental):**
    The script contains a commented-out section for webcam input (lines 7-9):
    ```python
    # cap = cv2.VideoCapture(1)  # Use 0 for the default webcam, 1 for an external webcam, etc.
    # cap.set(3, 1280) # Set width
    # cap.set(4, 720)  # Set height
    ```
    To use a webcam, you would uncomment these lines and comment out or remove line 11 (`cap = cv2.VideoCapture("../Video/people.mp4")`). Note that the script immediately reassigns `cap` on line 11, so that line must be handled if webcam input is desired.

## Key Files

*   **[`CarCounter/Carcounter.py`](CarCounter/Carcounter.py):** The core Python script that performs object detection using YOLOv8, processes video frames, and displays the results.
*   **[`Yolo-Weights/yolov8l.pt`](Yolo-Weights/yolov8l.pt):** The pre-trained YOLOv8 Large model weights file used for object detection.
*   **`Video/` directory:** Contains various sample video files (`.mp4`) that can be used as input for the detection script.

## How It Works

1.  **Initialization:**
    *   Imports necessary libraries.
    *   Initializes the video capture. It attempts to set up a webcam feed but then immediately overrides this by loading the `people.mp4` video file.
    *   Loads the pre-trained YOLOv8 model (`yolov8l.pt`).
    *   Defines a list of class names corresponding to the objects the model can detect.

2.  **Frame Processing Loop:**
    *   Continuously reads frames from the video source.
    *   For each frame:
        *   The frame is passed to the YOLO model for inference (`model(img, stream=True)`).
        *   The script iterates through the detection results.
        *   For each detected object:
            *   Bounding box coordinates are extracted.
            *   A styled rectangle (`cvzone.cornerRect`) is drawn around the object.
            *   The confidence score of the detection is calculated.
            *   The class of the detected object is determined.
            *   The class name and confidence score are displayed near the bounding box using `cvzone.putTextRect`.
    *   Calculates the FPS based on the time taken to process the frame.
    *   Displays the processed frame in an OpenCV window.
    *   Waits for a key press; the loop continues until the window is closed or the video ends.

## Potential Improvements / Future Work

*   **Command-Line Arguments:** Allow specifying the input video file, webcam ID, or other parameters via command-line arguments instead of hardcoding.
*   **Actual Counting Logic:** The script is named `Carcounter.py`, suggesting an intention to count objects (e.g., cars). This functionality is not yet implemented but could be added (e.g., by defining a detection line and counting objects that cross it).
*   **Output Video Saving:** Add an option to save the processed video with detections to a file.
*   **Configuration File:** For more complex settings (e.g., model path, confidence thresholds, class filters), a configuration file could be used.
*   **Error Handling:** Implement more robust error handling (e.g., if a video file is not found).
*   **GUI:** Develop a simple Graphical User Interface (GUI) for easier interaction.