import streamlit as st
import torch
import numpy as np
from streamlit_drawable_canvas import st_canvas
from inference import load_model, predict_drawing, convert_canvas_to_strokes

# Set page config
st.set_page_config(page_title="Drawing Recognition", layout="wide")

# Model weights path
MODEL_PATH = "kaggle-quickdraw-weights.pth"

@st.cache_resource
def load_cached_model():
    """Load and cache the model in memory."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(MODEL_PATH, device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure the model weights file exists in the repository.")
        st.stop()

# Title
st.title("Drawing Recognition App")
st.write("Draw something in the canvas below and the model will try to recognize it!")

# Create canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#ffffff",
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Recognize Drawing"):
    if canvas_result.image_data is not None:
        with st.spinner("Processing..."):
            # Convert canvas data to strokes
            drawing = convert_canvas_to_strokes(canvas_result.json_data)
            
            if drawing:
                try:
                    # Load model and make prediction
                    model, device = load_cached_model()
                    predictions = predict_drawing(model, drawing, device)
                    
                    # Display results
                    st.subheader("Top 5 Predictions:")
                    for pred in predictions:
                        st.write(f"{pred['class']}: {pred['probability']*100:.2f}%")
                except Exception as e:
                    st.error(f"Error processing drawing: {str(e)}")
                    st.error("Please try drawing again with simpler strokes.")
            else:
                st.warning("No drawing detected. Please draw something on the canvas.") 