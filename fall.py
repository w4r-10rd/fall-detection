import cv2
import matplotlib.pyplot as plt

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Initialize HOG descriptor for people detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize variables for fall detection
prev_height = 0
fall_detected = False

plt.ion() # Enable interactive mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people in the frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # Process each detected person
    for (x, y, w, h) in boxes:
        # Compare the height of the detected person to the previous height
        if prev_height > 0 and h < prev_height * 0.5:  # Example threshold for fall detection
            fall_detected = True
            cv2.putText(frame, "Fall Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Update the previous height
        prev_height = h
        
        # Draw rectangle around detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Convert BGR frame to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the video feed with detections
    #cv2.imshow('Fall Detection', frame)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.pause(0.01) # Pause to update the display

    # Check for fall alert
    if fall_detected:
        print("Alert: Fall detected!")
        fall_detected = False  # Reset for the next frame

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
#cv2.destroyAllWindows()
plt.close()
