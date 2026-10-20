"""Public research-only kidney ultrasound image classifier (STEP06 frozen CNN)."""

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from preprocessing import model_batch, preprocess_for_prediction

st.set_page_config(
    page_title="Kidney Ultrasound | Research Demo",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).resolve().parent / "model.keras"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
EXPECTED_MODEL_SHA256 = "464971dff1d7f584f5ab33a645f4357053754d90f9f4cb2100e3dac8969590df"


@st.cache_resource(show_spinner="Loading frozen research model…")
def load_model():
    # Delayed TF import makes dependency/startup failures easier to communicate.
    import tensorflow as tf

    actual_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("The deployed model file differs from the audited checkpoint.")
    # Model was saved with built-in Keras 3 layers; no arbitrary custom objects.
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False, safe_mode=True)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise RuntimeError("Unexpected saved model input shape.")
    if tuple(model.output_shape[1:]) != (1,):
        raise RuntimeError("Unexpected saved model output shape.")
    return model


def uploaded_image_to_gray(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Please choose a non-empty JPG or PNG up to 10 MB.")
    try:
        with Image.open(BytesIO(data)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 12-megapixel limit.")
            corrected = ImageOps.exif_transpose(img)
            gray = np.asarray(corrected.convert("L"), dtype=np.uint8)
        return gray
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable JPEG or PNG.") from exc


st.title("🔬 Kidney Ultrasound · Research Demo")
st.caption("Exploratory image classification · frozen STEP06 MobileNetV3Large")
st.error(
    "RESEARCH ONLY — NOT FOR CLINICAL USE. This system must not be used for "
    "diagnosis, treatment, triage, reassurance, or medical decisions."
)
st.write(
    "This demonstration shows how a research model assigns a *stone-class score* "
    "to an ultrasound image. It is not a validated measure of kidney stone presence."
)
with st.expander("Research limitations and model details", expanded=False):
    st.markdown(
        "- **Model:** STEP06 MobileNetV3Large CNN; **not** the proposed gated fusion, "
        "simulated federated, XAI-FL, or Grad-CAM++ system.\n"
        "- **Training preprocessing:** ultrasound fan-detection heuristic and central "
        "80% square crop, resized to 224×224; *not* verified kidney segmentation.\n"
        "- **Evidence:** source-related label shortcuts were found in the source "
        "dataset. Patient/site independence and robust external clinical validity "
        "have not been established.\n"
        "- **Score:** uncalibrated model output, not a clinical probability. "
        "No threshold here has been clinically validated."
    )

st.info(
    "Use only non-identifiable, permission-cleared research images. "
    "Do not upload names, dates of birth, IDs, or other patient-identifying data."
)
upload = st.file_uploader("Upload a kidney ultrasound JPG or PNG", type=["jpg", "jpeg", "png"])

if upload is not None:
    try:
        raw = uploaded_image_to_gray(upload.getvalue())
        crop, status = preprocess_for_prediction(raw)
        st.image(crop, caption="Model input: heuristic ultrasound-fan crop (224×224)", clamp=True)
        if status != "fan":
            st.warning(
                f"Preprocessing used '{status}' fallback; this image may be "
                "outside the development distribution."
            )
        with st.spinner("Evaluating the frozen model…"):
            pred = load_model()(model_batch(crop), training=False).numpy()
        score = float(np.asarray(pred).reshape(-1)[0])
        if not np.isfinite(score) or not (0 <= score <= 1):
            raise ValueError("The model returned an invalid score.")
        st.metric("Stone-class model score (not diagnostic)", f"{100 * score:.1f}%")
        st.progress(score)
        st.caption(
            "Values near 0 or 100 indicate the model's internal scoring, not "
            "certainty of a medical finding. False positives and false negatives may occur."
        )
    except (ValueError, OSError) as exc:
        st.warning(f"Image could not be evaluated: {exc}")
    except Exception:
        st.error(
            "The application could not load the model or run inference. "
            "Please check the Streamlit deployment logs and dependency versions."
        )

st.divider()
st.caption("Experimental research demonstration • Not a medical device • No clinical claims")
