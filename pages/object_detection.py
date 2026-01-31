import os
from pathlib import Path
import PIL
import streamlit as st

# Local modules
import settings
import helper
from llm_helper import load_llm

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Disease Detection | AgriVision-Bridge",
    page_icon="🌿",
    layout="wide"
)

st.title("🎯 Crop Disease Detection & Diagnosis")
st.write("Upload a crop leaf image to detect diseases and generate an AI-based diagnostic report.")

# --------------------------------------------------
# SIDEBAR – MODEL CONFIG
# --------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

confidence = float(
    st.sidebar.slider("Detection Confidence Threshold (%)", 25, 100, 45)
) / 100

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return helper.load_model(Path(settings.DETECTION_MODEL))

yolo_model = load_yolo_model()
llm = load_llm()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.sidebar.header("📤 Upload Image")
source_img = st.sidebar.file_uploader(
    "Choose a crop image",
    type=("jpg", "jpeg", "png", "bmp", "webp")
)

col1, col2 = st.columns(2)

# --------------------------------------------------
# LEFT COLUMN – SOURCE IMAGE
# --------------------------------------------------
with col1:
    st.subheader("📷 Input Image")

    if source_img is None:
        if os.path.exists(settings.DEFAULT_IMAGE):
            default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
            st.image(default_image, caption="Default Image", use_container_width=True)
        else:
            st.warning("Default image not found.")
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# --------------------------------------------------
# RIGHT COLUMN – DETECTION + LLM OUTPUT
# --------------------------------------------------
with col2:
    st.subheader("🔍 Detection Result")

    if source_img is None:
        if os.path.exists(settings.DEFAULT_DETECT_IMAGE):
            default_detected = PIL.Image.open(settings.DEFAULT_DETECT_IMAGE)
            st.image(default_detected, caption="Sample Output", use_container_width=True)
        else:
            st.info("Upload an image to view detection results.")
    else:
        if st.sidebar.button("Detect & Diagnose"):
            with st.spinner("Running YOLO detection..."):
                results = yolo_model.predict(uploaded_image, conf=confidence)
                plotted_img = results[0].plot()[:, :, ::-1]
                st.image(plotted_img, caption="Detected Diseases", use_container_width=True)

            # --------------------------------------------------
            # EXTRACT DETECTIONS
            # --------------------------------------------------
            detections = []

            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    cls_name = yolo_model.names[cls_id]
                    conf_score = float(box.conf)

                    detections.append(
                        f"- {cls_name} detected with {conf_score * 100:.2f}% confidence"
                    )

            with st.expander("📦 Detection Metadata"):
                if detections:
                    for d in detections:
                        st.write(d)
                else:
                    st.write("No diseases detected.")

            # --------------------------------------------------
            # LLM DIAGNOSTIC REPORT
            # --------------------------------------------------
            if detections:
                prompt = f"""
You are an agriculture expert.

A YOLO-based computer vision model analyzed a crop image and detected
the following disease conditions:

{' '.join(detections)}

Tasks:
1. Explain the disease(s) in simple, farmer-friendly language.
2. Mention possible causes.
3. Suggest immediate preventive or corrective actions.
4. If confidence is below 60%, politely mention uncertainty.

Keep the response concise and practical.
"""

                with st.spinner("Generating AI diagnostic report..."):
                    llm_output = llm(
                        prompt,
                        max_new_tokens=200,
                        temperature=0.4
                    )

                st.subheader("🧠 AI Diagnostic Report")
                st.write(llm_output[0]["generated_text"])
            else:
                st.info("No disease detected — LLM diagnosis not generated.")
