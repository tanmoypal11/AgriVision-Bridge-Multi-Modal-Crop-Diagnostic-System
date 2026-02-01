import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_llm():
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-small"
    )
