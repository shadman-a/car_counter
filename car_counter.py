import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')
layer_names = net.getLayerNames()

# Fix for IndexError by correctly indexing layers
unconnected_layers = net.getUnconnectedOutLayers()

# Handle both cases: when unconnected_layers is a scalar or an array
if isinstance(unconnected_layers, np.ndarray) and unconnected_layers.ndim > 1:
    output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
elif isinstance(unconnected_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_layers]
else:
    output_layers = [layer_names[unconnected_layers - 1]]

# Load COCO class labels
with open('models/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Extract detection results
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-max suppression to avoid overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and count cars
    car_count = 0
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            car_count += 1

    # Display car count
    cv2.putText(frame, f'Car Count: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow('Car Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
