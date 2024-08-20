import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("models/coco.names").read().strip().split("\n")

def process_frame(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize counts for different objects
    counts = {'car': 0, 'person': 0, 'bicycle': 0, 'bus': 0, 'truck': 0}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                if label in counts:
                    counts[label] += 1
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green for now, can be customized per class
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Add label and confidence score
                cv2.putText(frame, f'{label} {int(confidence * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, counts
