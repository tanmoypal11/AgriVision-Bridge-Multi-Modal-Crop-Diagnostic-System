import streamlit as st
from groq import Groq
from PIL import Image
from pathlib import Path
import helper  # Assuming your helper.py handles YOLO loading
import settings

# --- INITIALIZATION ---
st.set_page_config(page_title="AgriVision AI", page_icon="🌿", layout="wide")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: Upload & Confidence ---
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence (%)", 25, 100, 45) / 100.0
source_img = st.sidebar.file_uploader("Upload Leaf", type=("jpg", "jpeg", "png"))

# Load Model
@st.cache_resource
def load_yolo():
    return helper.load_model(Path(settings.DETECTION_MODEL))

yolo_model = load_yolo()

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Image Analysis")
    if source_img:
        uploaded_image = Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded", use_container_width=True)
        
        if st.button("Detect Disease"):
            results = yolo_model.predict(uploaded_image, conf=confidence)
            plotted_img = results[0].plot()[:, :, ::-1]
            st.image(plotted_img, caption="Result")

            # Extract results for the LLM
            detections = []
            for box in results[0].boxes:
                cls_name = yolo_model.names[int(box.cls)]
                conf = float(box.conf) * 100
                detections.append(f"{cls_name} ({conf:.2f}%)")

            if detections:
                detection_text = ", ".join(detections)
                # Create the custom prompt for the AI
                ai_prompt = f"The system detected: {detection_text}. \nTasks:\n1. Explain the disease in farmer-friendly language.\n2. Mention causes.\n3. Suggest immediate actions.\n4. If confidence is below 60%, mention uncertainty."
                
                # Push this to session state so the chat picks it up
                st.session_state.messages.append({"role": "user", "content": ai_prompt})
            else:
                st.warning("No disease detected.")

with col2:
    st.subheader("💬 AI Plant Doctor")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input (if user wants to ask follow-up questions)
    if chat_input := st.chat_input("Ask more about the treatment..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)

    # Generate response if the last message is from 'user'
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                stream=True
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
