import cv2
import mediapipe as mp
import numpy as np
import logging

# Initialize logging
logging.basicConfig(filename='fall_detection.log', level=logging.INFO)

# Initialize MediaPipe pose estimation components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define thresholds for fall detection (adjust as needed)
FALL_THRESHOLD = 0.1  # Height difference threshold
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold for pose detection

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    angle = np.degrees(np.arctan2(bc[1], bc[0]) - np.arctan2(ab[1], ab[0]))
    return angle + 360 if angle < 0 else angle

def process_frame(frame):
    """Process each video frame to detect falls."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        results = pose.process(rgb_frame)

        # Draw the skeleton if pose landmarks are detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks for fall detection
            landmarks = results.pose_landmarks.landmark
            
            # Check confidence
            confidence = np.mean([landmark.visibility for landmark in landmarks])
            if confidence < CONFIDENCE_THRESHOLD:
                cv2.putText(frame, "Insufficient confidence!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Get positions of key points (hips and shoulders)
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                # Calculate heights and angle
                hip_height = (left_hip.y + right_hip.y) / 2
                shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
                hip_angle = calculate_angle(
                    (left_shoulder.x, left_shoulder.y),
                    (left_hip.x, left_hip.y),
                    (left_knee.x, left_knee.y)
                )

                # Fall detection logic
                if hip_height > shoulder_height + FALL_THRESHOLD and hip_angle > 150:
                    cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    logging.info("Fall detected at hip height: %.2f, shoulder height: %.2f", hip_height, shoulder_height)
                else:
                    cv2.putText(frame, "Standing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No pose detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for performance
    frame = cv2.resize(frame, (640, 480))

    # Process the current frame
    processed_frame = process_frame(frame)

    # Display the video feed with detections
    cv2.imshow('Fall Detection with Bone Structure', processed_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
