from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from config import MODEL_PATH, CLASS_INFO
from model_utils import (
    get_inference_transform,
    load_model,
    predict_topk,
    generate_gradcam,
    GRADCAM_AVAILABLE,
)

st.set_page_config(page_title="Solar Panel Defect Detection", layout="wide")


@st.cache_resource
def load_artifacts():
    model, class_names, device = load_model(MODEL_PATH)
    transform = get_inference_transform()
    return model, class_names, device, transform


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def show_confidence_message(top3_results: list[tuple[str, float]]) -> None:
    """
    Shows a practical confidence interpretation based on:
    1. Top-1 confidence
    2. Gap between top-1 and top-2 probabilities
    """
    if len(top3_results) < 2:
        st.warning("Prediction available, but confidence interpretation is limited.")
        return

    top1_class, top1_prob = top3_results[0]
    top2_class, top2_prob = top3_results[1]

    top1_pct = top1_prob * 100
    top2_pct = top2_prob * 100
    gap = top1_pct - top2_pct

    confusing_classes = {"crack", "finger", "thick_line"}

    if top1_pct < 50:
        st.error("⚠️ Low confidence prediction — the model is uncertain.")
    elif top1_pct < 70 or gap < 10:
        st.warning("⚠️ Prediction uncertain — visually similar defect classes detected.")
    else:
        st.success("✅ High confidence prediction.")

    st.caption(f"Top-1 vs Top-2 confidence gap: {gap:.2f}%")

    if top1_class in confusing_classes and top2_class in confusing_classes:
        st.info(
            "Note: The top predictions belong to line-type defects "
            "(crack, finger, thick_line), which are known to be visually similar."
        )


def main():
    st.title("Solar Panel Defect Detection using Deep Learning")
    st.write(
        "Upload an electroluminescence (EL) image to classify the defect type, "
        "view the top-3 predictions, and inspect the model attention using Grad-CAM."
    )

    with st.sidebar:
        st.header("About this app")
        st.write("- Model: ResNet18")
        st.write("- Input size: 224 x 224")
        st.write("- Output: 8 defect classes")
        st.write("- Explainability: Grad-CAM")

    try:
        model, class_names, device, transform = load_artifacts()
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a solar panel EL image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    if uploaded_file is None:
        st.info("Upload an image to begin prediction.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        return

    try:
        predicted_class, confidence, top3_results = predict_topk(
            image=image,
            model=model,
            transform=transform,
            class_names=class_names,
            device=device,
            k=3,
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded image")
        st.image(image, caption="Input image", use_container_width=True)

    with col2:
        st.subheader("Prediction result")
        st.success(f"Predicted defect: {predicted_class}")
        st.write(f"Confidence: {format_percent(confidence)}")

        if predicted_class in CLASS_INFO:
            st.caption(CLASS_INFO[predicted_class])

        show_confidence_message(top3_results)

        st.subheader("Top-3 predictions")
        top3_df = pd.DataFrame(top3_results, columns=["Class", "Probability"])
        top3_df["Probability (%)"] = top3_df["Probability"].apply(lambda x: round(x * 100, 2))
        st.dataframe(
            top3_df[["Class", "Probability (%)"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(top3_df.set_index("Class")["Probability"])

    st.divider()
    st.subheader("Grad-CAM explainability")

    if not GRADCAM_AVAILABLE:
        st.warning(
            "Grad-CAM library is not installed. Add 'grad-cam' to your environment to enable this section."
        )
        return

    try:
        gradcam_result = generate_gradcam(
            image=image,
            model=model,
            transform=transform,
            class_names=class_names,
            device=device,
        )
    except Exception as exc:
        st.error(f"Grad-CAM generation failed: {exc}")
        return

    gc1, gc2 = st.columns(2)
    with gc1:
        st.image(
            gradcam_result["original_image"],
            caption="Resized input",
            use_container_width=True,
        )
    with gc2:
        st.image(
            gradcam_result["gradcam_overlay"],
            caption=(
                f"Grad-CAM overlay | {gradcam_result['predicted_class']} "
                f"({format_percent(gradcam_result['confidence'])})"
            ),
            use_container_width=True,
        )

    st.info(
        "Interpret Grad-CAM carefully: strong attention on a region does not always mean "
        "the model is using the exact defect boundary. This matters especially for confusing "
        "classes like crack, finger, and thick_line."
    )


if __name__ == "__main__":
    main()