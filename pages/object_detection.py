import PIL
import streamlit as st
from pathlib import Path
import os

# Local Modules
import settings
import helper
from llm_helper import load_llm   # ✅ LLM import added

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Image Detection | SmartVision AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 YOLOv8 Image Detection")
st.write("Upload an image to detect or segment objects.")

# 2. MODEL CONFIG
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
confidence = float(st.sidebar.slider("Model Confidence", 25, 100, 45)) / 100

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
else:
    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check path: {model_path}")
    st.stop()

# ✅ Load LLM (SAFE)
llm = load_llm()

# 3. IMAGE UPLOAD
st.sidebar.header("Image Upload")
source_img = st.sidebar.file_uploader(
    "Choose an image...",
    type=("jpg", "jpeg", "png", 'bmp', 'webp')
)

col1, col2 = st.columns(2)

# --- LEFT COLUMN: SOURCE ---
with col1:
    st.subheader("📷 Source Image")
    if source_img is None:
        if os.path.exists(str(settings.DEFAULT_IMAGE)):
            default_image = PIL.Image.open(str(settings.DEFAULT_IMAGE))
            st.image(default_image, caption="Default Image", use_container_width=True)
        else:
            st.warning("Default placeholder image not found on server.")
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# --- RIGHT COLUMN: RESULT ---
with col2:
    st.subheader("🔍 Result Image")
    if source_img is None:
        if os.path.exists(str(settings.DEFAULT_DETECT_IMAGE)):
            default_detected_image = PIL.Image.open(str(settings.DEFAULT_DETECT_IMAGE))
            st.image(default_detected_image, caption='Default Result', use_container_width=True)
        else:
            st.info("Upload an image to see detection results.")
    else:
        if st.sidebar.button('Detect Objects'):
            with st.spinner("Analyzing..."):
                res = model.predict(uploaded_image, conf=confidence)

                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Objects', use_container_width=True)

                # ---------------------------
                # Detection Metadata
                # ---------------------------
                detected_labels = []

                with st.expander("Detection Metadata"):
                    if res[0].boxes:
                        for box in res[0].boxes:
                            cls_id = int(box.cls)
                            cls_name = model.names[cls_id]
                            conf_score = float(box.conf)
                            detected_labels.append(
                                f"{cls_name} ({conf_score*100:.2f}%)"
                            )
                            st.write(cls_name, f"{conf_score*100:.2f}%")
                    else:
                        st.write("No objects detected.")

                # ---------------------------
                # 🧠 LLM DIAGNOSTIC REPORT
                # ---------------------------
                if detected_labels:
                    st.subheader("🧠 AI Explanation")

                    prompt = f"""
You are an expert computer vision assistant.

The YOLO model detected the following objects:
{', '.join(detected_labels)}

Explain what these detections mean in simple language.
Keep it short and clear.
"""

                    with st.spinner("Generating explanation..."):
                        llm_output = llm(prompt, max_new_tokens=150)

                    st.write(llm_output[0]["generated_text"])
