# Project Abstract: Object Detection with YOLOv8

This project implements a real-time object detection system utilizing the YOLOv8 (You Only Look Once version 8) model. The primary goal is to process video input, identify various objects within each frame, and provide visual feedback.

Key functionalities include:
*   **Real-time Detection:** Leverages the YOLOv8 model for efficient object identification in video streams.
*   **Multi-Class Identification:** Capable of detecting a diverse range of objects based on the COCO dataset (e.g., persons, cars, bicycles).
*   **Visual Output:** Displays the processed video with bounding boxes drawn around detected objects, along with their class labels and confidence scores.
*   **Performance Metrics:** Calculates and shows Frames Per Second (FPS) to monitor processing speed.

The system is currently configured to process a local video file (specifically `Video/people.mp4`) using the main script located at `CarCounter/Carcounter.py`. The object detection relies on pre-trained YOLOv8 Large model weights stored in `Yolo-Weights/yolov8l.pt`. While named `Carcounter.py`, the current implementation focuses on general object detection rather than specific counting logic, which is noted as a potential area for future development.