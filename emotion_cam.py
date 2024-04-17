import cv2
from deepface import DeepFace
import logging
import time

# Configure logging
logging.basicConfig(
    filename="log/emotions_cam.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)


# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start capturing video
cap = cv2.VideoCapture(0)

frame_skip = 5  # Skip every 5 frames
frame_count = 0
processed_frames = 0


start_time = time.time()

while processed_frames < 10:  # Process only the first 10 frames
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % frame_skip != 0:
        continue  # Skip this frame

    processed_frames += 1  # Increment the processed frames counter

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y : y + h, x : x + w]

        # Perform analysis on the face ROI for age, gender, race, and emotion
        result = DeepFace.analyze(
            face_roi,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False,
        )

        # Check if result is a list and access the first element if it is
        if isinstance(result, list):
            result = result[0]

        # Log the detected parameters along with additional data points
        logging.info(
            "Face detected at position X: %s, Y: %s, Width: %s, Height: %s, "
            "Gender: %s,  Dominant Emotion: %s",
            x,
            y,
            w,
            h,
            result["gender"],
            result["dominant_emotion"],
        )

        # Draw rectangle around face and label with predicted parameters
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = f"{result['dominant_emotion']} - {result['gender']}"

        cv2.putText(
            frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Display the resulting frame
    cv2.imshow("Real-time Face Analysis", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Calculate the total time taken
total_time = time.time() - start_time

# Calculate the time taken for 1 frame
time_per_frame = total_time / processed_frames

print(f"Total time for {processed_frames} frames: {total_time} seconds")
print(f"Time per frame: {time_per_frame} seconds")


# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
