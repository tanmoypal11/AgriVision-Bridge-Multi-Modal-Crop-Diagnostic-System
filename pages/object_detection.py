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
    page_title="Crop Disease Detection | AgriVision",
    page_icon="🌿",
    layout="centered"   # ✅ single column
)

st.title("🎯 Crop Disease Detection & Diagnosis")
st.write("Upload a crop image to detect disease and get an AI explanation.")

# --------------------------------------------------
# SIDEBAR CONFIG
# --------------------------------------------------
st.sidebar.header("⚙️ Model Settings")
confidence = float(
    st.sidebar.slider("Detection Confidence (%)", 25, 100, 45)
) / 100

st.sidebar.header("📤 Upload Image")
source_img = st.sidebar.file_uploader(
    "Choose an image",
    type=("jpg", "jpeg", "png", "bmp", "webp")
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_yolo():
    return helper.load_model(Path(settings.DETECTION_MODEL))

yolo_model = load_yolo()
llm = load_llm()

# --------------------------------------------------
# IMAGE DISPLAY
# --------------------------------------------------
if source_img is None:
    if os.path.exists(settings.DEFAULT_IMAGE):
        img = PIL.Image.open(settings.DEFAULT_IMAGE)
        st.image(img, caption="Default Image", use_container_width=True)
    else:
        st.info("Upload an image to begin.")
else:
    uploaded_image = PIL.Image.open(source_img)
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# --------------------------------------------------
# DETECTION
# --------------------------------------------------
if source_img and st.button("🔍 Detect & Diagnose"):
    with st.spinner("Running detection..."):
        results = yolo_model.predict(uploaded_image, conf=confidence)
        plotted = results[0].plot()[:, :, ::-1]
        st.image(plotted, caption="Detection Result", use_container_width=True)

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
                f"{cls_name} detected with {conf_score*100:.2f}% confidence"
            )

    with st.expander("📦 Detection Metadata"):
        if detections:
            for d in detections:
                st.write(d)
        else:
            st.write("No objects detected.")

    # --------------------------------------------------
    # 🧠 LLM OUTPUT (THIS WILL SHOW)
    # --------------------------------------------------
    if detections:
        st.subheader("🧠 AI Diagnostic Explanation")

        prompt = f"""
Task: Crop disease diagnosis

Detected disease:
{'; '.join(detections)}

Instructions:
- Explain the disease in simple farmer-friendly language
- Mention possible causes
- Suggest immediate preventive or corrective actions
- If confidence is below 60%, say the diagnosis may be uncertain

Answer clearly in 5–6 short bullet points.
"""

        with st.spinner("Generating AI explanation..."):
            llm_output = llm(prompt, max_new_tokens=200)

        st.write(llm_output[0]["generated_text"])
    else:
        st.info("No disease detected, so AI explanation was not generated.")





