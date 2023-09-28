import cv2
import numpy as np

# Define the line's position (you can adjust these values)
line_position = 400  # Adjust this based on your video frame dimensions

# Initialize the video capture device (0 for the default camera)
cap = cv2.VideoCapture("rail 2.mp4")

# Create a variable to store the previous frame
prev_frame = None

# Initialize a flag to indicate motion detection after the line
detected_after_line = False

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize prev_frame if it's the first frame
    if prev_frame is None:
        prev_frame = gray_frame
        continue

    # Compute the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(prev_frame, gray_frame)

    # Threshold the difference image to create a binary image
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False  # Initialize a flag to indicate motion detection

    # Loop through the contours and check for motion before and after the line
    for contour in contours:
        if cv2.contourArea(contour) > line_position:  # You can adjust the area threshold
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate the center of the detected object
            object_center_x = x + w // 2
            object_center_y = y + h // 2
            # Check if the center is to the left of the line
            if object_center_x < line_position:
                # Draw a rectangle around the moving object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if not detected_after_line:
                    detected_after_line = True
                    print("Detected")
            motion_detected = True

    # Draw the vertical line on the frame
    cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Motion Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Update the previous frame
    prev_frame = gray_frame

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
