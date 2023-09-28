import cv2
import numpy as np
from flask import Flask, jsonify

app = Flask(__name__)

# Load YOLOv3 model for people detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Replace with the path to your video file

@app.route("/people_count", methods=['GET'])
def get_people_count():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"people_count": 0})  # If the video is finished, return 0 people.

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize variables for bounding boxes and class IDs
    class_ids = []
    confidences = []
    boxes = []

    # Extract information from the detection outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 represents 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count people
    people_count = len(indexes)

    return jsonify({"people_count": people_count})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
