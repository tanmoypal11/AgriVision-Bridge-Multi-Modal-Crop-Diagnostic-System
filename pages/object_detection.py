import os
from pathlib import Path
import PIL.Image
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
    layout="centered"
)

st.title("🎯 Crop Disease Detection & Diagnosis")
st.write("Upload a crop image to detect disease and receive a farmer-friendly AI explanation.")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Detection Settings")
confidence = st.sidebar.slider(
    "Detection Confidence (%)",
    min_value=25,
    max_value=100,
    value=45
) / 100.0

st.sidebar.header("📤 Upload Image")
source_img = st.sidebar.file_uploader(
    "Choose a crop image",
    type=("jpg", "jpeg", "png", "bmp", "webp")
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return helper.load_model(Path(settings.DETECTION_MODEL))

yolo_model = load_yolo_model()
llm = load_llm()

# --------------------------------------------------
# IMAGE DISPLAY
# --------------------------------------------------
if source_img is None:
    if os.path.exists(settings.DEFAULT_IMAGE):
        default_img = PIL.Image.open(settings.DEFAULT_IMAGE)
        st.image(default_img, caption="Default Image", use_container_width=True)
    else:
        st.info("⬅️ Upload a crop image from the sidebar to begin.")
else:
    uploaded_image = PIL.Image.open(source_img)
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# --------------------------------------------------
# DETECTION + DIAGNOSIS
# --------------------------------------------------
if source_img and st.button("🔍 Detect & Diagnose"):
    st.divider()

    # ---------------------------
    # RUN YOLO
    # ---------------------------
    with st.spinner("Running disease detection..."):
        results = yolo_model.predict(uploaded_image, conf=confidence)

    # ---------------------------
    # SHOW IMAGE RESULT
    # ---------------------------
    plotted_img = results[0].plot()[:, :, ::-1]
    st.image(plotted_img, caption="Detection Result", use_container_width=True)

    # ---------------------------
    # EXTRACT DETECTIONS
    # ---------------------------
    detections = []
    structured_detections = []

    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = yolo_model.names[cls_id]
            conf_score = float(box.conf) * 100

            text = f"{cls_name} detected with {conf_score:.2f}% confidence"
            detections.append(text)

            structured_detections.append({
                "disease": cls_name,
                "confidence": conf_score
            })

    # ---------------------------
    # ALWAYS SHOW DETECTION TEXT
    # ---------------------------
    st.subheader("📦 Detection Result")

    if detections:
        for d in detections:
            st.write("•", d)
    else:
        st.warning("No disease detected in the image.")
        st.stop()

    # --------------------------------------------------
    # 🧠 AI DIAGNOSTIC EXPLANATION
    # --------------------------------------------------
    st.divider()
    st.subheader("🧠 AI Diagnostic Explanation")

    detection_lines = "\n".join(
        [f"- {d['disease']} ({d['confidence']:.2f}%)" for d in structured_detections]
    )

    prompt = f"""
You are an agricultural extension expert.

A crop image was analyzed by a computer vision system.
The following disease(s) were detected:

{detection_lines}

STRICT RULES:
- Talk ONLY about crop disease and farming
- Use simple farmer-friendly language
- No animals, people, medicine, or unrelated topics
- No stories, no examples outside agriculture
- No speculation beyond detected disease

Explain clearly:
• What the disease is
• Why it happens
• What the farmer should do immediately
• How to prevent it in future

If confidence is below 60%, clearly say the diagnosis may be uncertain.

Format:
- 5 to 6 short bullet points
- Practical, direct, and actionable
"""

    with st.spinner("Generating AI explanation..."):
        llm_output = llm(
            prompt,
            max_new_tokens=180,
            temperature=0.2,
            top_p=0.9
        )

    st.write(llm_output[0]["generated_text"])
