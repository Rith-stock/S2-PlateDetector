import streamlit as st
import cv2
import easyocr
import os
from collections import Counter
from ultralytics import YOLO

st.set_page_config(page_title="Parking Gate Reader", layout="wide")


# --- Cache the heavy models so they only load once, not on every rerun ---
@st.cache_resource
def load_models():
    model = YOLO('best.pt')
    reader = easyocr.Reader(['en'])
    return model, reader


model, reader = load_models()

os.makedirs("detected_plates", exist_ok=True)
LOG_PATH = "detected_plates/plates_log.txt"


def save_plate(plate_text):
    with open(LOG_PATH, "a") as f:
        f.write(plate_text + "\n")


def normalize_plate(text):
    """
    Collapse small OCR variations into one consistent form so the same
    plate isn't treated as a 'different' reading every frame.
    """
    text = text.replace(" ", "").replace(".", "").replace("-", "").upper()
    text = text.replace("I", "1").replace("O", "0")
    return text


def clean_plate_text(ocr_results):
    """
    Pick the most likely plate number from OCR results, filtering out
    pure-letter words (province names, brand names, etc.).
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


# --- Page layout ---
st.title("🚗 Parking Gate Reader")
st.caption("Group 4 — License Plate Detection & OCR")

col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox("Start Webcam")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Status")
    status_placeholder = st.empty()

    st.subheader("Recent Detections")
    if st.button("Clear Log"):
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        st.rerun()
    log_placeholder = st.empty()

# --- Stability tracking (same logic as the local app) ---
recent_readings = []
STABILITY_WINDOW = 8
MIN_AGREEMENT = 5
last_saved_plate = None
OCR_EVERY_N_FRAMES = 8
frame_count = 0

# Detection confidence threshold — raised from 0.5 to reduce false positives
# (e.g. the model mistaking a face/background for a plate)
DETECTION_CONFIDENCE = 0.65

if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("Webcam not accessible")
    else:
        status_placeholder.info("Webcam running — scanning for plates...")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            results = model(frame, imgsz=320, verbose=False)
            display_frame = frame.copy()
            frame_count += 1

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf > DETECTION_CONFIDENCE:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cropped_plate = frame[y1:y2, x1:x2]

                    if cropped_plate.size > 0 and frame_count % OCR_EVERY_N_FRAMES == 0:
                        scale_factor = 4
                        cropped_plate_big = cv2.resize(
                            cropped_plate, None,
                            fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_CUBIC
                        )

                        gray = cv2.cvtColor(cropped_plate_big, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)

                        ocr_result = reader.readtext(
                            enhanced,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        )
                        plate_text = clean_plate_text(ocr_result)

                        if plate_text:
                            recent_readings.append(plate_text)
                            if len(recent_readings) > STABILITY_WINDOW:
                                recent_readings.pop(0)

                            most_common, count = Counter(recent_readings).most_common(1)[0]

                            cv2.putText(display_frame, f"Reading: {plate_text}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                            if count >= MIN_AGREEMENT and most_common != last_saved_plate:
                                message = f"Access Granted - License #{most_common}"
                                cv2.putText(display_frame, message, (x1, y2 + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                save_plate(most_common)
                                last_saved_plate = most_common
                                status_placeholder.success(message)

            # Streamlit expects RGB, OpenCV gives BGR — convert before displaying
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB")

            # Show the log file's contents so far
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r") as f:
                    log_placeholder.text(f.read())
            else:
                log_placeholder.text("(no detections yet)")

        cap.release()
else:
    status_placeholder.info("Webcam is stopped. Check the box above to start.")