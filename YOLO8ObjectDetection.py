import cv2

from ultralytics import YOLO


# Load the YOLOv8 model
# model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model (nano version whic is lightweight)
# model = YOLO("yolov8s.pt")  # Load a pre-trained YOLOv8 model (small version)
# model = YOLO("yolov8m.pt")  # Load a pre-trained YOLOv8 model (medium version)
# model = YOLO("yolov8l.pt")  # Load a pre-trained YOLOv8 model (large version)
model = YOLO("yolov8x.pt")  # Load a pre-trained YOLOv8 model (extra-large version)

# Initialize the Camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()