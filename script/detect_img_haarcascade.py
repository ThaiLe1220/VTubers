import cv2
from deepface import DeepFace
import logging
import os
import re
import time  # Import the time module

# Configure logging
logging.basicConfig(
    filename="log/emotions_static.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)
logging.info("\n")

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "weights/haarcascade_frontalface_default.xml"
)

# Directory containing images
img_dir = "_img/"


# Helper function to convert text to natural number for sorting
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split("(\d+)", text)]


# Get all jpg files in the directory, sorted naturally
file_list = [filename for filename in os.listdir(img_dir) if filename.endswith(".jpeg")]
file_list.sort(key=natural_keys)

start_time = time.time()  # Start timing for the entire process

# Iterate through all files in the directory
for filename in file_list:
    img_process_start = time.time()  # Start timing for this image

    try:
        # Construct the full path to the image file
        img_path = os.path.join(img_dir, filename)

        # Read the image
        img = cv2.imread(img_path)
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y : y + h, x : x + w]

            # Optionally perform DeepFace analysis on the face ROI
            result = DeepFace.analyze(
                face_roi,
                actions=["gender", "emotion"],
                enforce_detection=False,
            )
            # Check if result is a list and access the first element if it is
            if isinstance(result, list):
                result = result[0]

            # Format the display text for gender and emotion probabilities
            gender_text = ", ".join(
                [f"{g}: {p:.1f}%" for g, p in result["gender"].items() if p > 1]
            )
            emotion_text = ", ".join(
                [f"{e}: {p:.1f}%" for e, p in result["emotion"].items() if p > 1]
            )
            combined_text = f"{gender_text}; {emotion_text}"

            # text_position = (x, y - 10 if y - 10 > 10 else y + h + 20)
            text_position = (x, y + h + 10)

            # Draw the text on the image in multiple lines if needed
            for j, line in enumerate(combined_text.split("; ")):
                cv2.putText(
                    img,
                    line,
                    (text_position[0], text_position[1] + (j * 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

        # Save the detected face region to a new file
        face_filename = f"{os.path.splitext(filename)[0]}_d.png"
        face_path = os.path.join("_img_detection", face_filename)
        cv2.imwrite(face_path, img)

    except Exception as e:
        logging.error("Error processing %s: %s", filename, e)
        continue

    img_process_end = time.time()  # End timing for this image
    logging.info(
        "Time taken to process %s: %s seconds",
        filename,
        img_process_end - img_process_start,
    )


end_time = time.time()  # End timing for the entire process
logging.info(
    "Total time taken to process all images: %s seconds", end_time - start_time
)


print("Face detection and saving process completed.")
