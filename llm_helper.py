from collections import Counter

# ----------------------------------------
# Crop-specific advisory knowledge base
# ----------------------------------------
CROP_ADVICE = {
    "leaf_blight": {
        "problem": "Leaf blight detected",
        "cause": "Fungal infection caused by high humidity",
        "solution": [
            "Remove infected leaves immediately",
            "Apply recommended fungicide",
            "Avoid overhead irrigation"
        ]
    },
    "rust": {
        "problem": "Rust disease detected",
        "cause": "Airborne fungal spores",
        "solution": [
            "Use resistant crop varieties",
            "Apply sulfur-based fungicide",
            "Ensure good air circulation"
        ]
    },
    "healthy": {
        "problem": "Crop appears healthy",
        "cause": "No visible disease symptoms",
        "solution": [
            "Continue regular monitoring",
            "Maintain proper irrigation",
            "Follow nutrient schedule"
        ]
    }
}

# ----------------------------------------
# Generate human-friendly explanation
# ----------------------------------------
def generate_explanation(predictions):
    """
    Convert YOLO predictions into farmer-friendly advice
    """
    if not predictions:
        return "No crop issues detected. The plant appears healthy."

    detected_classes = [p["class"] for p in predictions]
    class_counts = Counter(detected_classes)

    explanations = []

    for crop_class, count in class_counts.items():
        key = crop_class.lower().replace(" ", "_")

        if key in CROP_ADVICE:
            info = CROP_ADVICE[key]
            explanations.append(
                f"🔍 **{info['problem']}**\n"
                f"- Occurrences: {count}\n"
                f"- Likely Cause: {info['cause']}\n"
                f"- Recommended Action:\n"
                + "\n".join([f"  • {s}" for s in info["solution"]])
            )
        else:
            explanations.append(
                f"🔍 **{crop_class} detected**\n"
                f"- Occurrences: {count}\n"
                f"- Recommendation: Consult local agriculture expert."
            )

    return "\n\n".join(explanations)


# ----------------------------------------
# LLM-ready prompt generator (optional)
# ----------------------------------------
def build_llm_prompt(predictions):
    """
    Build prompt for GPT / Gemini / LLaMA
    """
    summary = Counter([p["class"] for p in predictions])

    prompt = f"""
You are an agricultural expert.

The crop image analysis detected:
{dict(summary)}

Explain:
1. What diseases are present
2. How severe the condition is
3. Preventive and corrective actions
4. Farmer-friendly advice in simple language

Keep response concise and actionable.
"""
    return prompt.strip()
