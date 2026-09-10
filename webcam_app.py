import cv2
import easyocr
import os
from collections import Counter
from ultralytics import YOLO

# Load your custom-trained plate detector
# NOTE: update this filename once you have your new v2/frozen best.pt
model = YOLO('best.pt')

# Load EasyOCR reader
reader = easyocr.Reader(['en'])

# Make sure the log folder exists
os.makedirs("detected_plates", exist_ok=True)


def save_plate(plate_text):
    with open("detected_plates/plates_log.txt", "a") as f:
        f.write(plate_text + "\n")


def normalize_plate(text):
    """
    Collapse small OCR variations into one consistent form so the same
    plate isn't treated as a 'different' reading every frame.
    - Removes separators (. and -) entirely
    - Fixes common look-alike misreads (I/1, O/0)
    """
    text = text.replace(" ", "").replace(".", "").replace("-", "").upper()
    text = text.replace("I", "1").replace("O", "0")
    return text


def clean_plate_text(ocr_results):
    """
    Pick the most likely plate number from OCR results.
    Real plate numbers contain digits (e.g. '1J-9829'), so we filter out
    pure-letter words like 'HONDA' or 'KAMPONG' that OCR also picks up.
    Returns the normalized form so frame-to-frame variants match up.
    """
    best_match = None
    best_conf = 0

    for (bbox, text, conf) in ocr_results:
        cleaned = text.replace(" ", "").upper()

        has_digit = any(c.isdigit() for c in cleaned)
        has_enough_chars = len(cleaned) >= 5

        if has_digit and has_enough_chars and conf > best_conf:
            best_match = normalize_plate(cleaned)
            best_conf = conf

    return best_match


# --- Stability tracking to avoid flickering / spamming the log ---
recent_readings = []
STABILITY_WINDOW = 8   # look at the last 8 frames
MIN_AGREEMENT = 5      # require the same reading at least 5 times in that window
last_saved_plate = None

# Only run the heavy OCR step every N frames — detection still runs every frame,
# OCR is the slow part, so we don't need to repeat it 20-30 times per second
OCR_EVERY_N_FRAMES = 8
frame_count = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

# Lower the capture resolution — most webcams default higher than needed,
# and a smaller frame means less work for YOLO on every single frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # imgsz=320 tells YOLO to work at a smaller internal resolution —
    # noticeably faster than the default (640), small accuracy trade-off
    results = model(frame, imgsz=320, verbose=False)
    display_frame = frame.copy()
    frame_count += 1

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf > 0.5:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cropped_plate = frame[y1:y2, x1:x2]

            # Only run OCR periodically — it's the slow step, detection box
            # above still updates every frame so the video stays smooth
            if cropped_plate.size > 0 and frame_count % OCR_EVERY_N_FRAMES == 0:
                # Upscale the crop so OCR has more detail to work with —
                # small/blurry text is a common cause of misreads (e.g. J read as 1, 4, 7)
                scale_factor = 4
                cropped_plate_big = cv2.resize(
                    cropped_plate, None,
                    fx=scale_factor, fy=scale_factor,
                    interpolation=cv2.INTER_CUBIC
                )

                # Convert to grayscale, then use CLAHE (adaptive contrast) instead of a
                # hard black/white threshold — a global threshold can wipe out text
                # entirely under uneven lighting/glare, CLAHE is much gentler and safer
                gray = cv2.cvtColor(cropped_plate_big, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Debug: save what OCR is actually seeing, so we can visually check
                # if the crop/preprocessing looks right when something seems off
                cv2.imwrite("debug_last_crop.jpg", enhanced)

                # allowlist restricts OCR to letters/numbers only — skips time
                # wasted trying to recognize punctuation/symbols that won't appear
                ocr_result = reader.readtext(
                    enhanced,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                plate_text = clean_plate_text(ocr_result)

                if plate_text:
                    # Track this reading for stability checking
                    recent_readings.append(plate_text)
                    if len(recent_readings) > STABILITY_WINDOW:
                        recent_readings.pop(0)

                    most_common, count = Counter(recent_readings).most_common(1)[0]

                    # Always show the current best guess on screen
                    live_message = f"Reading: {plate_text}"
                    cv2.putText(display_frame, live_message, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                    # Only confirm + log once it's stable across multiple frames
                    if count >= MIN_AGREEMENT and most_common != last_saved_plate:
                        message = f"Access Granted - License #{most_common}"
                        cv2.putText(display_frame, message, (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        save_plate(most_common)
                        last_saved_plate = most_common
                        print(message)

    cv2.imshow("Parking Gate Reader", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()