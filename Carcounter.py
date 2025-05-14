from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import sys # For sys.exit()
import numpy as np

# --- Configuration Variables ---
# Set these variables to configure the script's behavior.

# --- Core Detection Settings ---
VIDEO_PATH = "Video/bikes.mp4"  # Path to the input video file.
WEIGHTS_PATH = "Yolo-Weights/yolov8l.pt"  # Path to YOLOv8 model weights file.
USE_WEBCAM = False  # Set to True to use webcam as input source instead of video file.
WEBCAM_ID = 0  # Webcam ID to use if USE_WEBCAM is True.
WEBCAM_WIDTH = 1280  # Desired webcam frame width. Set to None to use webcam default.
WEBCAM_HEIGHT = 720  # Desired webcam frame height. Set to None to use webcam default.
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence threshold for detections.

# --- FPS Display Settings ---
HIDE_FPS_VIDEO = False  # Set to True to hide FPS display on the output video window.
HIDE_FPS_CONSOLE = False  # Set to True to hide FPS display in the console.

# --- Image Processing Demonstrations (Course Topics) ---
# 1. Gaussian Blur (Image Smoothing - Pre-processing)
APPLY_GAUSSIAN_BLUR = False  # If True, applies Gaussian blur to the full frame before detection.
GAUSSIAN_KERNEL_SIZE = 5  # Kernel size for Gaussian blur (must be odd).

# 2. Canny Edge Detection on Largest ROI (Post-processing Visualization)
SHOW_CANNY_EDGE_LARGEST_ROI = True  # If True, shows Canny edges for the largest detected object's ROI.
CANNY_THRESHOLD_1 = 50  # Lower threshold for Canny edge detection.
CANNY_THRESHOLD_2 = 150  # Higher threshold for Canny edge detection.

# 3. HSV Color Conversion of Largest ROI (Post-processing Visualization)
SHOW_HSV_LARGEST_ROI = False  # If True, shows the HSV version of the largest detected object's ROI.

# --- End of Configuration Variables ---

def main():
    cap = None
    user_quit = False

    # --- Video/Webcam Initialization ---
    if USE_WEBCAM:
        print(f"Attempting to use webcam ID: {WEBCAM_ID}")
        cap = cv2.VideoCapture(WEBCAM_ID)
        if WEBCAM_WIDTH and WEBCAM_HEIGHT:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    else:
        print(f"Attempting to load video: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap or not cap.isOpened():
        print("Error: Could not open video source.")
        print("Please check VIDEO_PATH or WEBCAM_ID settings.")
        sys.exit(1)

    # --- YOLO Model Initialization ---
    print(f"Loading YOLO model from: {WEIGHTS_PATH}")
    try:
        model = YOLO(WEIGHTS_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}. Check WEIGHTS_PATH: {WEIGHTS_PATH}")
        if cap: cap.release()
        sys.exit(1)

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    prev_frame_time = 0
    
    # Create named windows for largest ROI processing if enabled
    if SHOW_CANNY_EDGE_LARGEST_ROI:
        cv2.namedWindow("Canny Edge of Largest ROI", cv2.WINDOW_AUTOSIZE)
    if SHOW_HSV_LARGEST_ROI:
        cv2.namedWindow("HSV of Largest ROI", cv2.WINDOW_AUTOSIZE)


    print("Processing video... Press 'q' or ESC to quit, or Ctrl+C to interrupt.")

    try:
        while True:
            new_frame_time = time.time()
            success, frame = cap.read()
            if not success:
                print("End of video or cannot read frame.")
                break
            
            processed_frame = frame.copy()

            # --- 1. Gaussian Blur (Pre-processing) ---
            if APPLY_GAUSSIAN_BLUR:
                blur_kernel_size = GAUSSIAN_KERNEL_SIZE
                if blur_kernel_size % 2 == 0: # Ensure kernel size is odd and positive
                    blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size > 0 else 1
                else: # Kernel is already odd
                    blur_kernel_size = blur_kernel_size if blur_kernel_size > 0 else 1
                
                if blur_kernel_size > 0: # Apply blur if kernel size is valid
                   processed_frame = cv2.GaussianBlur(processed_frame, (blur_kernel_size, blur_kernel_size), 0)

            # --- YOLO Detection ---
            results = model(processed_frame, stream=True, conf=CONFIDENCE_THRESHOLD)
            
            largest_roi_area = 0
            largest_roi_original = None

            all_detections_for_frame = [] 

            for r in results:
                for box_data in r.boxes: 
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = box_data.conf[0] # Get confidence score
                    cls_idx = int(box_data.cls[0]) # Get class index
                    
                    all_detections_for_frame.append(((x1, y1, w, h), conf, cls_idx))

                    if w > 0 and h > 0: # Ensure valid ROI
                        current_area = w * h
                        if current_area > largest_roi_area:
                            largest_roi_area = current_area
                            largest_roi_original = frame[y1:y2, x1:x2].copy() 
            
            # --- Draw all detections on the main frame ---
            for (x1_draw, y1_draw, w_draw, h_draw), conf_draw, cls_idx_draw in all_detections_for_frame:
                cvzone.cornerRect(frame, (x1_draw, y1_draw, w_draw, h_draw), l=9, rt=2, colorR=(255,0,255))
                conf_val = math.ceil((conf_draw.item() * 100)) / 100 # .item() to get Python number from tensor
                label = f'{classNames[cls_idx_draw] if 0 <= cls_idx_draw < len(classNames) else f"Unknown({cls_idx_draw})"} {conf_val}'
                cvzone.putTextRect(frame, label, (max(0, x1_draw), max(35, y1_draw)), 
                                   scale=1, thickness=1, colorT=(255,255,255), colorR=(255,0,255), offset=5)

            # --- Process and Display Largest ROI ---
            if largest_roi_original is not None:
                # --- 2. Canny Edge Detection on Largest ROI ---
                if SHOW_CANNY_EDGE_LARGEST_ROI and largest_roi_original.size > 0:
                    gray_roi = cv2.cvtColor(largest_roi_original, cv2.COLOR_BGR2GRAY)
                    edges_roi = cv2.Canny(gray_roi, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
                    cv2.imshow("Canny Edge of Largest ROI", edges_roi)

                # --- 3. HSV Color Conversion of Largest ROI ---
                if SHOW_HSV_LARGEST_ROI and largest_roi_original.size > 0:
                    hsv_roi = cv2.cvtColor(largest_roi_original, cv2.COLOR_BGR2HSV)
                    cv2.imshow("HSV of Largest ROI", hsv_roi)
            else: 
                # If no objects detected, display blank images in ROI windows if they are enabled
                if SHOW_CANNY_EDGE_LARGEST_ROI:
                    # Create a small blank image, e.g., 100x100
                    blank_canny = np.zeros((100, 100, 1), dtype=np.uint8)
                    cv2.putText(blank_canny, "No Detections", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    cv2.imshow("Canny Edge of Largest ROI", blank_canny)
                if SHOW_HSV_LARGEST_ROI:
                    blank_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
                    cv2.putText(blank_hsv, "No Detections", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    cv2.imshow("HSV of Largest ROI", blank_hsv)


            # --- FPS Calculation and Display ---
            if prev_frame_time > 0:
                fps_val = 1 / (new_frame_time - prev_frame_time)
                if not HIDE_FPS_CONSOLE: print(f"FPS: {fps_val:.2f}", end='\r')
                if not HIDE_FPS_VIDEO:
                    cv2.putText(frame, f"FPS: {fps_val:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            prev_frame_time = new_frame_time

            cv2.imshow("Object Detection - YOLOv8 (Main)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q or ESC
                print(f"\n'{chr(key)}' pressed, initiating exit.")
                user_quit = True
                break
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
        user_quit = True
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if cap: cap.release()
        cv2.destroyAllWindows() 
        print("Resources released.")

    if user_quit:
        print("Exiting script now.")
        sys.exit(0)
    else:
        print("Script finished or encountered an issue.")

if __name__ == "__main__":
    main()
