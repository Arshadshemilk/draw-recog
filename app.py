import streamlit as st
import torch
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import json
import os
import gdown
from model import strokes_to_seresnext50_32x4d, process_single_drawing

# Set page config
st.set_page_config(page_title="Drawing Recognition", layout="wide")

# Model weights URL (Google Drive)
MODEL_WEIGHTS_URL = "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID"  # You'll need to replace this with your Google Drive file ID
MODEL_WEIGHTS_PATH = "kaggle-quickdraw-weights.pth"

def download_model_weights():
    """Download the model weights if they don't exist."""
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        st.info("Downloading model weights... This may take a few minutes.")
        try:
            gdown.download(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH, quiet=False)
            st.success("Model weights downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model weights: {str(e)}")
            return False
    return True

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

# Initialize model
@st.cache_resource
def load_model():
    if not download_model_weights():
        st.stop()
    model = strokes_to_seresnext50_32x4d(img_size=32, window=64, num_classes=340)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    return model

def convert_canvas_to_strokes(canvas_result):
    if canvas_result.json_data is not None:
        # Extract stroke data
        strokes = []
        current_stroke = []
        
        for obj in canvas_result.json_data["objects"]:
            if "path" in obj:
                points = []
                for cmd in obj["path"]:
                    if cmd[0] in ["M", "L"]:  # Move to or Line to
                        points.append([cmd[1], cmd[2]])
                if points:
                    points = np.array(points)
                    strokes.append(points)
        
        if strokes:
            # Convert to Quick Draw format
            drawing = []
            for stroke in strokes:
                x = stroke[:, 0].tolist()
                y = stroke[:, 1].tolist()
                drawing.append([x, y])
            
            return drawing
    return None

if st.button("Recognize Drawing"):
    if canvas_result.image_data is not None:
        with st.spinner("Processing..."):
            # Convert canvas data to strokes
            drawing = convert_canvas_to_strokes(canvas_result)
            
            if drawing:
                try:
                    # Process the drawing
                    points, indices = process_single_drawing(drawing)
                    
                    # Load model and make prediction
                    model = load_model()
                    
                    # Convert to tensor and add batch dimension
                    points = torch.FloatTensor(points).unsqueeze(0)
                    indices = torch.LongTensor(indices).unsqueeze(0)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = model(points, indices)
                        prob = torch.softmax(output, dim=1)
                        top_p, top_class = torch.topk(prob, k=5)
                        
                    # Display results
                    st.subheader("Top 5 Predictions:")
                    for p, c in zip(top_p[0], top_class[0]):
                        st.write(f"Class {c.item()}: {p.item()*100:.2f}%")
                except Exception as e:
                    st.error(f"Error processing drawing: {str(e)}")
            else:
                st.warning("No drawing detected. Please draw something on the canvas.") 