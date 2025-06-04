import streamlit as st
import torch
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import json
import os
from model import strokes_to_seresnext50_32x4d, process_single_drawing

# Set page config
st.set_page_config(page_title="Drawing Recognition", layout="wide")

# Model weights path
MODEL_PATH = "kaggle-quickdraw-weights.pth"

@st.cache_resource
def load_model():
    """Load and cache the model in memory."""
    try:
        # Initialize model with correct parameters
        model = strokes_to_seresnext50_32x4d(
            img_size=256,  # Increased image size
            window=64,     # Window size for temporal convolution
            num_classes=340
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure the model weights file exists in the repository.")
        st.stop()

# Title
st.title("Drawing Recognition App")
st.write("Draw something in the canvas below and the model will try to recognize it!")

# Initialize session state for stroke timing
if 'stroke_start_time' not in st.session_state:
    st.session_state.stroke_start_time = None
if 'current_stroke_points' not in st.session_state:
    st.session_state.current_stroke_points = []
if 'strokes' not in st.session_state:
    st.session_state.strokes = []

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

def normalize_drawing(strokes):
    """Normalize the drawing to fit in a unit square."""
    if not strokes:
        return strokes
    
    # Find min and max coordinates
    all_points = np.concatenate([np.array(stroke[:2]).T for stroke in strokes])
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    
    # Calculate scale and offset
    scale = max(max_coords - min_coords)
    if scale == 0:
        scale = 1
    
    normalized_strokes = []
    for stroke in strokes:
        x = ((np.array(stroke[0]) - min_coords[0]) / scale)
        y = ((np.array(stroke[1]) - min_coords[1]) / scale)
        t = np.array(stroke[2]) if len(stroke) > 2 else np.arange(len(x))
        normalized_strokes.append([x.tolist(), y.tolist(), t.tolist()])
    
    return normalized_strokes

def convert_canvas_to_strokes(canvas_result):
    if canvas_result.json_data is not None:
        strokes = []
        current_time = 0
        
        for obj in canvas_result.json_data["objects"]:
            if "path" in obj:
                # Extract stroke properties from the object
                stroke = {
                    "x": [],
                    "y": [],
                    "t": [],
                    "color": obj.get("stroke", "#000000"),
                    "width": obj.get("strokeWidth", 3)
                }
                
                for i, cmd in enumerate(obj["path"]):
                    if cmd[0] in ["M", "L"]:  # Move to or Line to
                        stroke["x"].append(float(cmd[1]))
                        stroke["y"].append(float(cmd[2]))
                        stroke["t"].append(current_time + i * 16)  # Match JS 60fps sampling (16ms)
                
                if stroke["x"]:  # Only add strokes with points
                    # Normalize coordinates to [0, 1] range
                    x_min, x_max = min(stroke["x"]), max(stroke["x"])
                    y_min, y_max = min(stroke["y"]), max(stroke["y"])
                    x_range = x_max - x_min if x_max > x_min else 1
                    y_range = y_max - y_min if y_max > y_min else 1
                    
                    stroke["x"] = [(x - x_min) / x_range for x in stroke["x"]]
                    stroke["y"] = [(y - y_min) / y_range for y in stroke["y"]]
                    
                    # Adjust time points to be relative to first point
                    first_time = stroke["t"][0]
                    stroke["t"] = [t - first_time for t in stroke["t"]]
                    
                    strokes.append(stroke)
                    current_time = stroke["t"][-1] + 100  # Add gap between strokes
        
        if strokes:
            # Convert to the format expected by process_single_drawing
            drawing = []
            for stroke in strokes:
                drawing.append([
                    stroke["x"],
                    stroke["y"],
                    stroke["t"]
                ])
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
                    st.error("Please try drawing again with simpler strokes.")
            else:
                st.warning("No drawing detected. Please draw something on the canvas.") 
