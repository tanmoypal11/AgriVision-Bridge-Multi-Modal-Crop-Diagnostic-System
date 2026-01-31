import streamlit as st
import pandas as pd
from PIL import Image
import os

# -------------------------------------------------
# 1. PAGE CONFIG & STABILIZATION CSS
# -------------------------------------------------
st.set_page_config(
    page_title="Model Performance | SmartVision AI",
    layout="wide"
)

st.markdown("""
    <style>
    html {
        overflow-y: scroll;
    }
    .main {
        overflow-x: hidden;
    }
    img {
        max-height: 500px;
        object-fit: contain;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# 2. PATH CONFIGURATION
# -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

def get_img_path(relative_path):
    return os.path.join(root_dir, relative_path)

# -------------------------------------------------
# NEW: SIDEBAR NAVIGATION TO KEEP MODULES SEPARATE
# -------------------------------------------------
st.sidebar.title("SmartVision AI")
module = st.sidebar.radio("Select Module:", ["Object Detection (YOLOv5u)", "CNN Classification"])

# =================================================
# MODULE: OBJECT DETECTION (NEW)
# =================================================
if module == "Object Detection (YOLOv5u)":
    st.title("🎯 Object Detection Performance")
    st.write("Evaluation of **YOLOv5u** trained on Google Colab")
    
    # YOLO Metrics Highlights
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("mAP@50", "90.1%")
    m2.metric("Precision", "97.9%")
    m3.metric("Recall", "85.0%")
    m4.metric("Inference", "3.3 ms")

    st.markdown("---")
    st.subheader("Confusion Matrix – YOLOv5u")
    st.image(get_img_path("images/YOLO/confusion_matrix.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Training Results (Loss & mAP Curves)")
    st.image(get_img_path("images/YOLO/results.png"), use_container_width=True)

# =================================================
# MODULE: CNN CLASSIFICATION (YOUR INTACT CODE)
# =================================================
else:
    # -------------------------------------------------
    # 3. HEADER
    # -------------------------------------------------
    st.title("📊 Model Performance Dashboard")
    st.write("Comprehensive evaluation of CNN models used in **SmartVision AI**")

    # -------------------------------------------------
    # 4. FINAL METRICS TABLE
    # -------------------------------------------------
    data = {
        "Model": ["EfficientNet-B0 (Fully Unlocked)", "EfficientNet-B0 (Partial)", "MobileNet", "ResNet18", "VGG16"],
        "Accuracy (%)": [81.65, 65.38, 64.26, 64.82, 55.38],
        "Precision (%)": [81.88, 64.95, 64.07, 64.92, 59.73],
        "Recall (%)": [81.90, 65.38, 64.26, 64.82, 55.38],
        "F1 Score (%)": [81.86, 64.73, 63.91, 64.55, 53.11],
        "Inference Time": ["0.49 ms", "84.10 sec", "57.52 sec", "65.48 sec", "64.14 sec"],
        "Model Size (MB)": [41.49, 15.70, 9.28, 42.76, 512.58]
    }

    df = pd.DataFrame(data)
    st.subheader("📌 Final Test Metrics Comparison")
    st.dataframe(df, use_container_width=True)

    # -------------------------------------------------
    # 5. BEST MODEL SECTION
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("🏆 EfficientNet-B0 (Fully Unlocked – Colab Trained)")

    col_best1, col_best2 = st.columns([1, 2])
    with col_best1:
        st.markdown("""
        **📊 FINAL EVALUATION**
        - Accuracy: **81.65%**
        - Precision: **81.88%**
        - Recall: **81.90%**
        - F1 Score: **81.86%**
        - Inference Time: **0.49 ms**
        - Model Size: **41.49 MB**
        """)
    with col_best2:
        img_best = Image.open(get_img_path("images/EfficientNet_Colab/Efficientnet_confusion_matrix_colab.png"))
        st.image(img_best, caption="Confusion Matrix – EfficientNet Fully Unlocked", use_container_width=True)

    # -------------------------------------------------
    # 6. HELPER FUNCTION FOR OTHER MODELS
    # -------------------------------------------------
    def display_performance(title, conf_path, curve_path):
        st.markdown("---")
        st.subheader(title)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Confusion Matrix**")
            st.image(get_img_path(conf_path), use_container_width=True)
        with c2:
            st.write("**Loss & Accuracy Curve**")
            st.image(get_img_path(curve_path), use_container_width=True)

    # -------------------------------------------------
    # 7. REMAINING MODELS
    # -------------------------------------------------
    display_performance(
        "🔵 EfficientNet-B0 (Partially Frozen)", 
        "images/EfficientNet/EfficientNet_confusion_matrix.png", 
        "images/EfficientNet/EfficientNet_loss_accuracy_curve.png"
    )

    display_performance(
        "🟢 MobileNet", 
        "images/MobileNet/MobileNet_confusion_matrix.png", 
        "images/MobileNet/MobileNet_loss_accuracy.png"
    )

    display_performance(
        "🟣 ResNet18", 
        "images/Resnet18/Confusion_matrix_Resnet18.png", 
        "images/Resnet18/loss_accuracy_Resnet18.png"
    )

    display_performance(
        "🔴 VGG16", 
        "images/VGG16/VGG16_confusion_matrix.png", 
        "images/VGG16/VGG16_accuracy_loss_curve.png"
    )

    # -------------------------------------------------
    # 8. FINAL OBSERVATIONS
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Final Observations")
    st.markdown("""
    - 🏆 **EfficientNet-B0 (Fully Unlocked)** is the top performer, benefiting from full weight updates.
    - 🟢 **MobileNet** remains the most efficient for mobile/edge scenarios.
    - 🔴 **VGG16** shows the limits of older architectures with much larger parameter sizes.
    """)
