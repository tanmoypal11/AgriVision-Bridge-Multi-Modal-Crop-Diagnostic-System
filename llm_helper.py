import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_llm():
    """
    Load lightweight text-generation model for explanations.
    Cached to avoid reload on every interaction.
    """
    generator = pipeline(
        task="text-generation",
        model="google/flan-t5-small",
        max_new_tokens=128
    )
    return generator


def generate_insight(llm, detected_objects):
    """
    Generate agricultural insight based on detected objects
    """
    if not detected_objects:
        return "No crops or plant diseases detected in the image."

    prompt = (
        "You are an agriculture expert. "
        "Based on the following detected objects, "
        "provide crop health insight and suggestions:\n\n"
        f"{', '.join(detected_objects)}"
    )

    response = llm(prompt)[0]["generated_text"]
    return response
