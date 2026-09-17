import streamlit as st
import cv2
import numpy as np
import easyocr
import os
import threading
import time
import csv
import pandas as pd
from datetime import datetime
from collections import Counter
from ultralytics import YOLO

st.set_page_config(page_title="Parking Gate Reader", layout="wide")


@st.cache_resource
def load_models():
    model = YOLO('best.pt')
    reader = easyocr.Reader(['en'])
    return model, reader


model, reader = load_models()

os.makedirs("detected_plates", exist_ok=True)
LOG_PATH = "detected_plates/plates_log.csv"

DETECTION_CONFIDENCE = 0.4
STABILITY_WINDOW = 8
MIN_AGREEMENT = 5
OCR_EVERY_N_DETECTIONS = 3
MIN_OCR_CONFIDENCE = 0.45


def save_detection(location, plate_text, time_in):
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["location", "plate_num", "time_in"])
        writer.writerow([location, plate_text, time_in])


def normalize_plate(text):
    text = text.replace(" ", "").replace(".", "").replace("-", "").upper()
    text = text.replace("I", "1").replace("O", "0")
    return text


# =====================================================================
# LIVE WEBCAM PATH — unchanged from before, used only by detection_worker
# =====================================================================

def clean_plate_text(ocr_results):
    candidates = []
    for (bbox, text, conf) in ocr_results:
        cleaned = text.replace(" ", "").upper()
        if conf >= MIN_OCR_CONFIDENCE and any(c.isdigit() for c in cleaned):
            left_x = bbox[0][0]
            candidates.append((left_x, cleaned))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    combined = normalize_plate("".join(text for _, text in candidates))

    if len(combined) >= 5:
        return combined
    return None


def extract_location_text(ocr_results):
    location_words = []

    for (bbox, text, conf) in ocr_results:
        cleaned = text.replace(" ", "").upper()
        is_letters_only = cleaned.isalpha()
        has_enough_chars = len(cleaned) >= 4

        if is_letters_only and has_enough_chars:
            left_x = bbox[0][0]
            location_words.append((left_x, text.strip().upper()))

    if location_words:
        location_words.sort(key=lambda w: w[0])
        return " ".join(word for _, word in location_words)
    return None


def read_plate_from_crop(cropped_plate, reader):
    if cropped_plate.size == 0:
        return None, None, []

    scale_factor = 2
    cropped_plate_big = cv2.resize(
        cropped_plate, None,
        fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(cropped_plate_big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    ocr_result = reader.readtext(
        enhanced,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    plate_text = clean_plate_text(ocr_result)
    location_text = extract_location_text(ocr_result)
    return plate_text, location_text, ocr_result


# =====================================================================
# PHOTO UPLOAD PATH — separate, fixed functions, isolated from webcam path
# =====================================================================

UPLOAD_MIN_OCR_CONFIDENCE = 0.6     # bar for the main number group (e.g. "9829") — this text is large & clear
PREFIX_MIN_OCR_CONFIDENCE = 0.5     # bar for the small prefix group (e.g. "1J"). Raised from 0.15 after
                                     # seeing a low-confidence garbage read ("JL" at 0.29) get accepted as
                                     # a real prefix on a photo where OCR simply never read the true prefix.
                                     # A missing prefix (falls back to number-only) is safer than a wrong one.
PREFIX_CROP_WIDTH_RATIO = 0.42      # left portion of the plate crop where the letter/number prefix sits


def pick_best_digit_block(ocr_results, min_conf=UPLOAD_MIN_OCR_CONFIDENCE, min_len=4, max_len=4):
    """
    Finds the single most confident OCR block that looks like the plate's
    main number group (e.g. "9829") — a short run of characters containing
    at least one digit. [PHOTO UPLOAD PATH ONLY]
    """
    best_match, best_conf = None, 0.0
    for (bbox, text, conf) in ocr_results:
        cleaned = text.replace(" ", "").upper()
        has_digit = any(c.isdigit() for c in cleaned)
        length_ok = min_len <= len(cleaned) <= max_len
        if has_digit and length_ok and conf >= min_conf and conf > best_conf:
            best_match, best_conf = cleaned, conf
    return best_match, best_conf


def pick_best_prefix_block(ocr_results, min_conf=PREFIX_MIN_OCR_CONFIDENCE, max_len=3):
    """
    Finds the most confident short (<=3 char) OCR block that looks like
    the letter/digit prefix (e.g. "1J") rather than part of the main
    number. Requires at least one letter so a pure-digit fragment (which
    is far more likely to be a stray piece of the number, e.g. "98")
    never gets mistaken for the prefix. [PHOTO UPLOAD PATH ONLY]
    """
    best_match, best_conf = None, 0.0
    for (bbox, text, conf) in ocr_results:
        cleaned = text.replace(" ", "").upper()
        length_ok = 1 <= len(cleaned) <= max_len
        has_letter = any(c.isalpha() for c in cleaned)
        if length_ok and has_letter and conf >= min_conf and conf > best_conf:
            best_match, best_conf = cleaned, conf
    return best_match, best_conf


def read_plate_from_crop_upload(cropped_plate, reader):
    """
    Photo-upload version of the OCR step, run in two passes:

      1. Full-plate pass, tried at a few preprocessing strengths, to find
         the main number group (e.g. "9829") and the location text. Some
         photos respond very differently to contrast enhancement, so no
         single fixed setting works for every photo.

      2. A dedicated pass on just the LEFT portion of the crop (where the
         short letter/digit prefix like "1J" sits, before the dash),
         upscaled much more aggressively than the full-plate pass. That
         prefix is physically tiny relative to the whole plate, so a
         full-crop OCR pass tends to lose or badly misread it — zooming
         in on just that region gives OCR far more pixels to work with.

    The prefix and number are then stitched together into the full plate.
    [PHOTO UPLOAD PATH ONLY — do not use for live webcam]
    """
    if cropped_plate.size == 0:
        return None, None, []

    all_debug_results = []
    best_number_text, best_number_conf = None, 0.0
    best_prefix_text, best_prefix_conf = None, 0.0
    best_location_text = None

    # ---- Build the list of (image, scale_factor, clahe_clip_limit) passes to run ----
    # First the full-plate crop at a few preprocessing strengths (for the number + location),
    # then a dedicated, more heavily upscaled pass on just the left portion (for the prefix).
    # Every pass is checked for BOTH a number candidate and a prefix candidate — the earlier
    # bug was only checking the dedicated prefix pass for prefixes, missing good prefix reads
    # that showed up in one of the full-plate passes instead.
    passes = [
        (cropped_plate, 4, 2.0),   # strong enhancement
        (cropped_plate, 3, 1.0),   # milder enhancement
        (cropped_plate, 2, None),  # just upscale, no contrast enhancement at all
    ]

    h, w = cropped_plate.shape[:2]
    prefix_width = max(int(w * PREFIX_CROP_WIDTH_RATIO), 1)
    prefix_crop = cropped_plate[:, :prefix_width]
    if prefix_crop.size > 0:
        passes.append((prefix_crop, 6, 2.5))

    for img, scale_factor, clip_limit in passes:
        big = cv2.resize(
            img, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

        if clip_limit is not None:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
        else:
            processed = gray

        ocr_result = reader.readtext(
            processed,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        all_debug_results.extend(ocr_result)

        number_text, number_conf = pick_best_digit_block(ocr_result)
        if number_text and number_conf > best_number_conf:
            best_number_text, best_number_conf = number_text, number_conf

        prefix_text, prefix_conf = pick_best_prefix_block(ocr_result)
        if prefix_text and prefix_conf > best_prefix_conf:
            best_prefix_text, best_prefix_conf = prefix_text, prefix_conf

        location_text = extract_location_text(ocr_result)
        if location_text and not best_location_text:
            best_location_text = location_text

    # ---- Stitch prefix + number together ----
    if best_number_text and best_prefix_text:
        final_plate = normalize_plate(best_prefix_text + best_number_text)
    elif best_number_text:
        final_plate = normalize_plate(best_number_text)
    else:
        final_plate = None

    return final_plate, best_location_text, all_debug_results


def detect_plate_in_image(frame, model, reader, imgsz=640):
    results = model(frame, imgsz=imgsz, verbose=False)
    boxes = results[0].boxes

    display_frame = frame.copy()
    best_plate_text = None
    best_location_text = None
    debug_ocr_results = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf > DETECTION_CONFIDENCE:
            cropped_plate = frame[y1:y2, x1:x2]
            plate_text, location_text, raw_ocr = read_plate_from_crop_upload(cropped_plate, reader)
            debug_ocr_results.extend(raw_ocr)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if plate_text:
                cv2.putText(display_frame, f"Reading: {plate_text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                best_plate_text = plate_text
                best_location_text = location_text

    return display_frame, best_plate_text, best_location_text, debug_ocr_results


# =====================================================================
# LIVE WEBCAM — SharedState and worker, unchanged
# =====================================================================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.boxes = []
        self.reading_text = None
        self.granted_message = None
        self.stop = False


def detection_worker(state, model, reader):
    recent_readings = []
    last_saved_plate = None
    detection_pass_count = 0

    while not state.stop:
        with state.lock:
            frame = None if state.latest_frame is None else state.latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        results = model(frame, imgsz=384, verbose=False)
        boxes = results[0].boxes
        detection_pass_count += 1

        reading_text = None
        granted_message = None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf > DETECTION_CONFIDENCE:
                if detection_pass_count % OCR_EVERY_N_DETECTIONS == 0:
                    cropped_plate = frame[y1:y2, x1:x2]
                    plate_text, location_text, _ = read_plate_from_crop(cropped_plate, reader)

                    if plate_text:
                        reading_text = plate_text
                        recent_readings.append(plate_text)
                        if len(recent_readings) > STABILITY_WINDOW:
                            recent_readings.pop(0)

                        most_common, count = Counter(recent_readings).most_common(1)[0]

                        if count >= MIN_AGREEMENT and most_common != last_saved_plate:
                            granted_message = f"Access Granted - License #{most_common}"
                            time_in = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            save_detection(location_text or "Unknown", most_common, time_in)
                            last_saved_plate = most_common

        with state.lock:
            state.boxes = boxes
            if reading_text:
                state.reading_text = reading_text
            if granted_message:
                state.granted_message = granted_message


def render_log():
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.text("(no detections yet)")


st.title("🚗 Parking Gate Reader")
st.caption("Group 4 — License Plate Detection & OCR")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Status")
    status_placeholder = st.empty()

    st.subheader("Recent Detections")
    if st.button("Clear Log"):
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        st.rerun()
    log_placeholder = st.empty()

with col1:
    mode = st.radio("Input source", ["Live Webcam", "Upload Photo"], horizontal=True)

    if mode == "Live Webcam":
        run = st.checkbox("Start Webcam")
        frame_placeholder = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            if not cap.isOpened():
                st.error("Webcam not accessible")
            else:
                state = SharedState()
                worker = threading.Thread(target=detection_worker, args=(state, model, reader), daemon=True)
                worker.start()

                status_placeholder.info("Webcam running — scanning for plates...")

                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame")
                        break

                    with state.lock:
                        state.latest_frame = frame
                        boxes = state.boxes
                        reading_text = state.reading_text
                        granted_message = state.granted_message

                    display_frame = frame.copy()

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        if conf > DETECTION_CONFIDENCE:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if reading_text:
                                cv2.putText(display_frame, f"Reading: {reading_text}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                            if granted_message:
                                cv2.putText(display_frame, granted_message, (x1, y2 + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if granted_message:
                        status_placeholder.success(granted_message)

                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB")

                    with log_placeholder.container():
                        render_log()

                state.stop = True
                cap.release()
        else:
            status_placeholder.info("Webcam is stopped. Check the box above to start.")
            with log_placeholder.container():
                render_log()

    else:  # Upload Photo
        uploaded_file = st.file_uploader(
            "Upload a photo of the plate",
            type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            # Only re-run detection when a genuinely new file is uploaded —
            # not on every rerun triggered by editing the confirm form below.
            file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            if st.session_state.get("last_uploaded_id") != file_id:
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is None:
                    st.session_state.pending_upload_detection = None
                else:
                    with st.spinner("Detecting plate..."):
                        display_frame, plate_text, location_text, debug_ocr_results = detect_plate_in_image(frame, model, reader)
                    st.session_state.pending_upload_detection = {
                        "display_frame": display_frame,
                        "plate_text": plate_text or "",
                        "location_text": location_text or "",
                        "debug_ocr_results": debug_ocr_results,
                    }
                st.session_state.last_uploaded_id = file_id

            pending = st.session_state.get("pending_upload_detection")

            if pending is None:
                st.error("Could not read that image — try a different file.")
            else:
                rgb_frame = cv2.cvtColor(pending["display_frame"], cv2.COLOR_BGR2RGB)
                st.image(rgb_frame, channels="RGB", use_container_width=True)

                if pending["plate_text"]:
                    status_placeholder.info("Review the detected plate below, then confirm to save it.")
                else:
                    status_placeholder.warning("No plate auto-detected — enter it manually below if visible.")

                # Nothing is saved to the log until the user confirms — OCR on a
                # glare-heavy plate can look confident and still be wrong (e.g.
                # guessing a prefix like "JL" from noise), so this catches that
                # before it becomes a bad row in plates_log.csv.
                with st.form("confirm_upload_detection_form"):
                    edited_plate = st.text_input("Plate number", value=pending["plate_text"])
                    edited_location = st.text_input("Location", value=pending["location_text"])
                    submitted = st.form_submit_button("Confirm & Save")

                if submitted:
                    if edited_plate.strip():
                        time_in = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        final_plate = normalize_plate(edited_plate.strip())
                        final_location = edited_location.strip().upper() or "Unknown"
                        save_detection(final_location, final_plate, time_in)
                        status_placeholder.success(f"Saved — License #{final_plate}")
                    else:
                        st.error("Plate number can't be empty.")

                with st.expander("Debug: raw OCR output", expanded=not pending["plate_text"]):
                    if pending["debug_ocr_results"]:
                        for bbox, text, conf in pending["debug_ocr_results"]:
                            st.text(f"'{text}'  (confidence: {conf:.2f})")
                    else:
                        st.text("EasyOCR returned nothing at all for the detected crop — "
                                 "likely a preprocessing/crop issue rather than a text-parsing issue.")

                with log_placeholder.container():
                    render_log()
        else:
            st.session_state.last_uploaded_id = None
            st.session_state.pending_upload_detection = None
            status_placeholder.info("Upload a photo to run detection.")
            with log_placeholder.container():
                render_log()