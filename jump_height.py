import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video Capture
video = cv2.VideoCapture(0)

# Calibration Variables
baseline_y = None
max_displacement = 0
jumping = False
jump_count = 0
jump_heights = []

AVERAGE_HUMAN_HEIGHT_CM = 150  # Used to estimate pixel-to-cm scale dynamically

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Mid-hip point
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        mid_hip_y = (left_hip.y + right_hip.y) / 2

        # Estimate body height from NOSE to LEFT_ANKLE
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        height_norm = abs(nose.y - ankle.y)

        H = frame.shape[0]
        mid_hip_y_px = mid_hip_y * H
        pixel_height = height_norm * H

        # Avoid errors for very small people in frame
        if pixel_height > 100:
            pixel_to_cm = AVERAGE_HUMAN_HEIGHT_CM / pixel_height

            # Set baseline when person is standing
            if baseline_y is None:
                baseline_y = mid_hip_y_px

            # Jump detection
            threshold_px = 10
            if baseline_y - mid_hip_y_px > threshold_px:
                jumping = True
                displacement = baseline_y - mid_hip_y_px
                max_displacement = max(max_displacement, displacement)
            else:
                if jumping:
                    jump_height_cm = max_displacement * pixel_to_cm
                    jump_heights.append(jump_height_cm)
                    jump_count += 1
                    print(f"Jump {jump_count}: {jump_height_cm:.2f} cm")
                    jumping = False
                    max_displacement = 0

        # Draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show jump data
    cv2.putText(frame, f"Jumps: {jump_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if jump_heights:
        cv2.putText(frame, f"Last Jump: {jump_heights[-1]:.2f} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Jump Height Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()