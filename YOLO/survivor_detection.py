import os
import csv
import cv2
import time
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
VIDEO_PATH = "Aerial View 1.mp4"       # Change this
OUTPUT_VIDEO = "survivor_detection.mp4"
STATS_FILE = "detection_statistics.csv"

MODEL_NAME = "yolo11n.pt"
CONF_THRESHOLD = 0.10
IMAGE_SIZE = 640

# =========================
# CHECK INPUT
# =========================
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Put the video in this folder or change VIDEO_PATH."
    )


print("Loading YOLO...")
model = YOLO(MODEL_NAME)
print("Model loaded.\n")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30.0

print("Video information")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")
print(f"Duration   : {total_frames / fps:.2f} seconds\n")


ret, frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Could not read first frame.")

print("Testing first frame...")
results = model(frame, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False)

first_frame_detections = 0

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # COCO class 0 = person
        if class_id == 0 and confidence >= CONF_THRESHOLD:
            first_frame_detections += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"Survivor {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

cv2.imwrite("first_frame_detection.jpg", frame)

print(f"Survivors in first frame: {first_frame_detections}")
print("Saved first_frame_detection.jpg\n")

# Reset video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO, fourcc, fps, (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

frame_number = 0
total_detections = 0
frames_with_survivors = 0
max_survivors = 0
statistics = []

start_time = time.time()

print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1
    survivors_this_frame = 0

    results = model(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Only detect people
            if class_id != 0 or confidence < CONF_THRESHOLD:
                continue

            survivors_this_frame += 1
            total_detections += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Survivor {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    if survivors_this_frame > 0:
        frames_with_survivors += 1

    max_survivors = max(max_survivors, survivors_this_frame)

    # Information displayed on video
    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Survivors: {survivors_this_frame}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    out.write(frame)

    statistics.append([
        frame_number,
        survivors_this_frame
    ])

    if frame_number % 30 == 0:
        percent = 100 * frame_number / total_frames
        print(f"Processed {frame_number}/{total_frames} ({percent:.1f}%)")

cap.release()
out.release()

elapsed = time.time() - start_time

with open(STATS_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Survivors_Detected"])
    writer.writerows(statistics)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Frames processed            : {frame_number}")
print(f"Frames containing survivors : {frames_with_survivors}")
print(f"Total detections            : {total_detections}")
print(f"Maximum survivors/frame     : {max_survivors}")

if frame_number:
    print(
        f"Frames with survivors (%)   : "
        f"{100 * frames_with_survivors / frame_number:.2f}%"
    )

print(f"Processing time              : {elapsed:.2f} seconds")
print(f"Average processing FPS       : {frame_number / elapsed:.2f}")
print(f"Output video                 : {OUTPUT_VIDEO}")
print(f"Statistics                   : {STATS_FILE}")
print("First-frame test             : first_frame_detection.jpg")
print("=" * 60)
