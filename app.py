import streamlit as st
import torch
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import json
import os
from model import strokes_to_seresnext50_32x4d, process_single_drawing
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Drawing Recognition", layout="wide")

# Model weights path
MODEL_PATH = "kaggle-quickdraw-weights.pth"

# Class names (same as in code.txt)
CLASSES = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'airplane', 'alarm_clock', 'ambulance',
           'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana',
           'bandage', 'barn', 'baseball_bat', 'baseball', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear',
           'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry',
           'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli',
           'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar',
           'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat',
           'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock',
           'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon',
           'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin',
           'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant',
           'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant',
           'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower',
           'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden_hose', 'garden', 'giraffe', 'goatee',
           'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
           'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital',
           'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house_plant', 'house', 'hurricane', 'ice_cream',
           'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop', 'leaf', 'leg',
           'light_bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map',
           'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito',
           'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose',
           'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda',
           'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin',
           'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car',
           'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio',
           'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'river', 'roller_coaster', 'rollerskates',
           'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver',
           'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull',
           'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman',
           'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel',
           'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry',
           'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword',
           't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger',
           'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light',
           'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase',
           'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle',
           'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

idx_to_class = {i: cls for i, cls in enumerate(CLASSES)}

@st.cache_resource
def load_model():
    """Load and cache the model in memory."""
    try:
        # Initialize model with correct parameters (matching code.txt)
        model = strokes_to_seresnext50_32x4d(
            img_size=32,    # Changed from 256 to 32 (matching code.txt)
            window=2,       # Changed from 64 to 2 (matching code.txt)
            num_classes=340
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'), strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure the model weights file exists in the repository.")
        st.stop()

def convert_canvas_to_strokes_advanced(canvas_result):
    """
    Advanced conversion that tries to extract more detailed stroke information.
    This attempts to capture the actual drawing path more accurately.
    """
    if canvas_result.json_data is not None:
        strokes = []
        
        for obj in canvas_result.json_data["objects"]:
            if obj.get("type") == "path":
                # Try to extract from the path data
                path_data = obj.get("path", [])
                if not path_data:
                    continue
                
                stroke_x = []
                stroke_y = []
                stroke_t = []
                time_counter = 0
                
                current_x, current_y = 0, 0
                
                for cmd in path_data:
                    if len(cmd) < 3:
                        continue
                        
                    cmd_type = cmd[0]
                    
                    if cmd_type == "M":  # Move to
                        current_x, current_y = float(cmd[1]), float(cmd[2])
                        stroke_x.append(current_x)
                        stroke_y.append(current_y)
                        stroke_t.append(time_counter)
                        time_counter += 16
                        
                    elif cmd_type == "L":  # Line to
                        end_x, end_y = float(cmd[1]), float(cmd[2])
                        
                        # Interpolate points between current and end position
                        distance = np.sqrt((end_x - current_x)**2 + (end_y - current_y)**2)
                        num_points = max(2, int(distance / 5))  # Point every ~5 pixels
                        
                        for i in range(1, num_points + 1):
                            t = i / num_points
                            x = current_x + t * (end_x - current_x)
                            y = current_y + t * (end_y - current_y)
                            stroke_x.append(x)
                            stroke_y.append(y)
                            stroke_t.append(time_counter)
                            time_counter += 8
                        
                        current_x, current_y = end_x, end_y
                        
                    elif cmd_type == "Q":  # Quadratic curve
                        if len(cmd) >= 5:
                            ctrl_x, ctrl_y = float(cmd[1]), float(cmd[2])
                            end_x, end_y = float(cmd[3]), float(cmd[4])
                            
                            # Sample points along quadratic curve
                            for t in np.linspace(0.1, 1.0, 10):
                                x = (1-t)**2 * current_x + 2*(1-t)*t * ctrl_x + t**2 * end_x
                                y = (1-t)**2 * current_y + 2*(1-t)*t * ctrl_y + t**2 * end_y
                                stroke_x.append(x)
                                stroke_y.append(y)
                                stroke_t.append(time_counter)
                                time_counter += 8
                            
                            current_x, current_y = end_x, end_y
                
                if len(stroke_x) > 1:
                    stroke = np.array([stroke_x, stroke_y, stroke_t])
                    strokes.append(stroke)
        
        return strokes
    return None

def convert_canvas_to_strokes(canvas_result):
    """Convert canvas data to stroke format matching the original dataset."""
    # First try the advanced method
    strokes = convert_canvas_to_strokes_advanced(canvas_result)
    
    # If that doesn't work, fall back to the basic method
    if not strokes and canvas_result.json_data is not None:
        strokes = []
        current_time = 0
        
        for obj in canvas_result.json_data["objects"]:
            if "path" in obj:
                stroke_x = []
                stroke_y = []
                stroke_t = []
                
                # Process SVG path commands to extract all points
                for i, cmd in enumerate(obj["path"]):
                    if cmd[0] == "M":  # Move to
                        stroke_x.append(float(cmd[1]))
                        stroke_y.append(float(cmd[2]))
                        stroke_t.append(current_time)
                        current_time += 16  # Simulate ~60fps (16ms intervals)
                    elif cmd[0] == "L":  # Line to
                        stroke_x.append(float(cmd[1]))
                        stroke_y.append(float(cmd[2]))
                        stroke_t.append(current_time)
                        current_time += 16
                    elif cmd[0] == "Q":  # Quadratic curve - extract intermediate points
                        # For quadratic curves, we have control point and end point
                        if len(cmd) >= 5:  # Q cx cy x y
                            # Add some intermediate points for smoother curves
                            start_x, start_y = stroke_x[-1] if stroke_x else cmd[3], stroke_y[-1] if stroke_y else cmd[4]
                            ctrl_x, ctrl_y = float(cmd[1]), float(cmd[2])
                            end_x, end_y = float(cmd[3]), float(cmd[4])
                            
                            # Sample points along the quadratic curve
                            for t in np.linspace(0.2, 1.0, 4):  # Skip t=0 as it's the start point
                                # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                                x = (1-t)**2 * start_x + 2*(1-t)*t * ctrl_x + t**2 * end_x
                                y = (1-t)**2 * start_y + 2*(1-t)*t * ctrl_y + t**2 * end_y
                                stroke_x.append(x)
                                stroke_y.append(y)
                                stroke_t.append(current_time)
                                current_time += 8  # Faster sampling for curves
                
                if len(stroke_x) > 1:  # Only add strokes with multiple points
                    # Convert to numpy arrays
                    stroke = np.array([stroke_x, stroke_y, stroke_t])
                    strokes.append(stroke)
                
                # Add gap between strokes
                current_time += 100
        
        return strokes
    return strokes

def visualize_strokes(strokes, title="Stroke Data Visualization"):
    """Create a matplotlib plot of the stroke data."""
    if not strokes:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: X-Y coordinates
    colors = plt.cm.tab10(np.linspace(0, 1, len(strokes)))
    for i, stroke in enumerate(strokes):
        if stroke.shape[1] > 1:
            ax1.plot(stroke[0], stroke[1], 'o-', color=colors[i], 
                    linewidth=2, markersize=3, label=f'Stroke {i+1}')
    
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Drawing Path (X-Y)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot 2: Time series data
    for i, stroke in enumerate(strokes):
        if stroke.shape[1] > 1:
            ax2.plot(stroke[2], stroke[0], 'o-', color=colors[i], 
                    linewidth=2, markersize=3, label=f'Stroke {i+1} - X')
            ax2.plot(stroke[2], stroke[1], 's--', color=colors[i], 
                    linewidth=2, markersize=3, alpha=0.7, label=f'Stroke {i+1} - Y')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Coordinate Value')
    ax2.set_title('Coordinates vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def parse_stroke_input(text_input):
    """Parse stroke data from text input. Supports multiple formats."""
    try:
        # Remove extra whitespace and newlines
        text_input = text_input.strip()
        
        if not text_input:
            return None
            
        # Try to parse as JSON first
        try:
            data = json.loads(text_input)
            if isinstance(data, list):
                # Convert to numpy arrays
                strokes = []
                for stroke in data:
                    if isinstance(stroke, list) and len(stroke) == 3:
                        strokes.append(np.array(stroke))
                return strokes if strokes else None
        except json.JSONDecodeError:
            pass
        
        # Try to parse as Python list format
        try:
            data = eval(text_input)  # Be careful with eval in production!
            if isinstance(data, list):
                strokes = []
                for stroke in data:
                    if isinstance(stroke, list) and len(stroke) == 3:
                        strokes.append(np.array(stroke))
                return strokes if strokes else None
        except:
            pass
            
        return None
        
    except Exception as e:
        st.error(f"Error parsing stroke data: {str(e)}")
        return None

def format_strokes_for_display(strokes):
    """Format strokes data for display in the text area."""
    if not strokes:
        return ""
    
    formatted_strokes = []
    for stroke in strokes:
        formatted_stroke = [
            stroke[0].tolist(),  # X coordinates
            stroke[1].tolist(),  # Y coordinates  
            stroke[2].tolist()   # Time coordinates
        ]
        formatted_strokes.append(formatted_stroke)
    
    return json.dumps(formatted_strokes, indent=2)

def predict_drawing(drawing):
    """Make prediction on drawing data."""
    if not drawing or len(drawing) == 0:
        return None, None
        
    try:
        # Process the drawing using the corrected function
        points, indices = process_single_drawing(
            drawing, 
            out_size=2048,      # Match the original parameters
            actual_points=256,
            padding=16
        )
        
        # Load model and make prediction
        model = load_model()
        
        # Convert to tensor and add batch dimension
        points_tensor = points.unsqueeze(0)  # points is already a tensor
        indices_tensor = indices.unsqueeze(0)  # indices is already a tensor
        
        # Make prediction
        with torch.no_grad():
            output = model(points_tensor, indices_tensor)
            
            # Handle single sample output
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            # Get top 5 predictions
            prob = torch.softmax(output, dim=1)
            top_p, top_class = torch.topk(prob, k=5)
            
        return top_p[0], top_class[0]
        
    except Exception as e:
        st.error(f"Error processing drawing: {str(e)}")
        return None, None

# Title
st.title("Enhanced Drawing Recognition App")
st.write("Draw something in the canvas below or input raw stroke data for recognition!")

# Debug section (can be hidden in production)
if st.checkbox("🔍 Show Debug Information"):
    if canvas_result.json_data is not None:
        st.subheader("Canvas JSON Data Structure")
        st.json(canvas_result.json_data)
        
        st.subheader("Raw Path Analysis")
        for i, obj in enumerate(canvas_result.json_data.get("objects", [])):
            st.write(f"**Object {i+1}:**")
            st.write(f"- Type: {obj.get('type', 'unknown')}")
            if "path" in obj:
                st.write(f"- Path commands: {len(obj['path'])}")
                st.write(f"- First few commands: {obj['path'][:5]}")

# Create two columns for the main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Drawing Canvas")
    
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
    
    if st.button("Recognize Canvas Drawing", key="canvas_predict"):
        if canvas_result.image_data is not None:
            with st.spinner("Processing canvas drawing..."):
                # Convert canvas data to strokes
                drawing = convert_canvas_to_strokes(canvas_result)
                
                if drawing and len(drawing) > 0:
                    # Store strokes in session state for display
                    st.session_state['current_strokes'] = drawing
                    
                    # Make prediction
                    top_p, top_class = predict_drawing(drawing)
                    
                    if top_p is not None and top_class is not None:
                        st.session_state['predictions'] = (top_p, top_class)
                    else:
                        st.error("Failed to make prediction.")
                else:
                    st.warning("No drawing detected. Please draw something on the canvas.")

with col2:
    st.subheader("Raw Stroke Data Input")
    
    # Example format information
    with st.expander("ℹ️ Stroke Data Format"):
        st.write("""
        **Format:** List of strokes, where each stroke is [x_coords, y_coords, time_coords]
        
        **Example:**
        ```json
        [
          [[0, 10, 20], [0, 10, 20], [0, 16, 32]],
          [[25, 35], [15, 25], [48, 64]]
        ]
        ```
        
        - Each stroke has 3 arrays: X coordinates, Y coordinates, and time stamps
        - Coordinates should be numeric values
        - You can copy the stroke data from the visualization below
        """)
    
    # Text area for raw stroke input
    stroke_input = st.text_area(
        "Paste stroke data here:",
        height=200,
        placeholder="Paste your stroke data in JSON format here...",
        key="stroke_input"
    )
    
    if st.button("Recognize Raw Stroke Data", key="raw_predict"):
        if stroke_input.strip():
            with st.spinner("Processing raw stroke data..."):
                # Parse the input
                drawing = parse_stroke_input(stroke_input)
                
                if drawing and len(drawing) > 0:
                    # Store strokes in session state for display
                    st.session_state['current_strokes'] = drawing
                    
                    # Make prediction
                    top_p, top_class = predict_drawing(drawing)
                    
                    if top_p is not None and top_class is not None:
                        st.session_state['predictions'] = (top_p, top_class)
                    else:
                        st.error("Failed to make prediction.")
                else:
                    st.error("Invalid stroke data format. Please check the example format above.")
        else:
            st.warning("Please enter stroke data to analyze.")

# Display stroke visualization and predictions
st.markdown("---")

# Create columns for stroke visualization and predictions
viz_col1, viz_col2 = st.columns([2, 1])

with viz_col1:
    st.subheader("Stroke Data Visualization")
    
    # Display current strokes if available
    if 'current_strokes' in st.session_state and st.session_state['current_strokes']:
        strokes = st.session_state['current_strokes']
        
        # Show stroke statistics
        st.write(f"**Number of strokes:** {len(strokes)}")
        for i, stroke in enumerate(strokes):
            st.write(f"- Stroke {i+1}: {stroke.shape[1]} points")
        
        # Create and display the plot
        fig = visualize_strokes(strokes)
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Prevent memory leaks
        
        # Display raw stroke data
        with st.expander("📋 Raw Stroke Data (Copy/Paste)"):
            formatted_data = format_strokes_for_display(strokes)
            st.code(formatted_data, language="json")
    else:
        st.info("Draw something on the canvas or input stroke data to see visualization.")

with viz_col2:
    st.subheader("Predictions")
    
    # Display predictions if available
    if 'predictions' in st.session_state:
        top_p, top_class = st.session_state['predictions']
        
        st.write("**Top 5 Predictions:**")
        for i, (p, c) in enumerate(zip(top_p, top_class)):
            class_name = idx_to_class.get(c.item(), f"Class_{c.item()}")
            confidence = p.item() * 100
            
            # Create a progress bar for confidence
            st.write(f"**{i+1}. {class_name}**")
            st.progress(confidence/100)
            st.write(f"{confidence:.2f}%")
            st.write("")
    else:
        st.info("Make a prediction to see results here.")

# Clear button
if st.button("🗑️ Clear All Data"):
    # Clear session state
    if 'current_strokes' in st.session_state:
        del st.session_state['current_strokes']
    if 'predictions' in st.session_state:
        del st.session_state['predictions']
    st.rerun()
