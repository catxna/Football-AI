from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('models/best_pro.pt')


# Custom function to draw bounding boxes with smaller, transparent labels
def draw_custom_box(image, bbox, label, color=(0, 255, 0), thickness=2, alpha=0.4):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]  # Ensure coordinates are integers

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare label background (transparent)
    overlay = image.copy()
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    label_height = label_size[1] + 6

    label_x1 = x1  # Align with  the bounding box
    label_x2 = x1 + label_size[0] + 4  # Extend to accommodate the label text
    label_y1 = y2  # Position below the bounding box
    label_y2 = y2 + label_height

    # Draw a transparent rectangle for the label background
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Add the label text
    cv2.putText(image, label, (x1 + 2, label_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Run YOLO inference on a video
results = model.predict('input_vids/pro_short3.mp4', save=False)

# Open the video
cap = cv2.VideoCapture('input_vids/pro_short3.mp4')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_vids/annotated_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frame by frame
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get detections for the current frame
    detections = results[frame_index].boxes
    for box in detections:
        bbox = box.xyxy[0]
        cls = int(box.cls[0])  # Make sure it's an integer
        confidence = box.conf[0]

        # if model.names[cls].lower() == "ball":  # Only proceed if the object is "Ball"
        label = f"{model.names[cls]} {confidence:.2f}"
        draw_custom_box(frame, bbox, label)


    # Write the processed frame to the output video
    out.write(frame)
    frame_index += 1

cap.release()
out.release()

print(f"Annotated video saved to {output_path}")