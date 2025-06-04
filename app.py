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

def convert_canvas_to_strokes(canvas_result):
    """Convert canvas data to stroke format matching the original dataset."""
    if canvas_result.json_data is not None:
        strokes = []
        
        for obj in canvas_result.json_data["objects"]:
            if "path" in obj:
                stroke_x = []
                stroke_y = []
                stroke_t = []
                
                for i, cmd in enumerate(obj["path"]):
                    if cmd[0] in ["M", "L"]:  # Move to or Line to
                        stroke_x.append(float(cmd[1]))
                        stroke_y.append(float(cmd[2]))
                        stroke_t.append(i * 16)  # Simulate 60fps timing
                
                if len(stroke_x) > 1:  # Only add strokes with multiple points
                    # Convert to numpy arrays for easier processing
                    stroke = np.array([stroke_x, stroke_y, stroke_t])
                    strokes.append(stroke)
        
        return strokes
    return None

if st.button("Recognize Drawing"):
    if canvas_result.image_data is not None:
        with st.spinner("Processing..."):
            # Convert canvas data to strokes
            drawing = convert_canvas_to_strokes(canvas_result)
            
            if drawing and len(drawing) > 0:
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
                        
                    # Display results
                    st.subheader("Top 5 Predictions:")
                    for i, (p, c) in enumerate(zip(top_p[0], top_class[0])):
                        class_name = idx_to_class.get(c.item(), f"Class_{c.item()}")
                        confidence = p.item() * 100
                        st.write(f"{i+1}. {class_name}: {confidence:.2f}%")
                        
                except Exception as e:
                    st.error(f"Error processing drawing: {str(e)}")
                    st.error("Please try drawing again with simpler strokes.")
                    # Debug information
                    st.write("Debug info:")
                    if 'drawing' in locals():
                        st.write(f"Number of strokes: {len(drawing)}")
                        for i, stroke in enumerate(drawing):
                            st.write(f"Stroke {i} shape: {stroke.shape}")
            else:
                st.warning("No drawing detected. Please draw something on the canvas.")
