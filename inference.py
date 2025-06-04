import torch
import numpy as np
from model import strokes_to_seresnext50_32x4d
from classes import CLASSES, get_class_mapping

def process_stroke_data(drawing, out_size=2048, actual_points=256, padding=16):
    """Process a drawing into the format expected by the model."""
    data = [np.array(s) for s in drawing]
    
    # Normalize and scale
    minimums = np.stack([s.min(1) for s in data]).min(0)
    maximums = np.stack([s.max(1) for s in data]).max(0)
    scale = maximums - minimums
    scale[scale == 0] = 1
    data = [(s - minimums[:, None]) / scale[:, None] for s in data]
    data = [np.clip(s*255, 0, 255) for s in data]

    # Create points and indices arrays
    points = np.zeros((3, out_size), dtype=np.float32)
    indices = np.full(actual_points, padding, dtype=np.int64)
    cursor_points = 0
    cursor_indices = 0
    
    for s in data:
        remaining_space_points = out_size - cursor_points
        remaining_space_indices = actual_points - cursor_indices
        padded_s = np.pad(s, [[0, 0], [padding, padding]], mode='edge')
        keep_new = min(padded_s.shape[1], remaining_space_points)
        points[:, cursor_points:cursor_points+keep_new] = padded_s[:, :keep_new]
        cursor_points += keep_new

        num_points = s.shape[1]
        indices_new_start = max(padding, cursor_points - padding - keep_new)
        keep_new_indices = min(num_points, remaining_space_indices)
        indices[cursor_indices:cursor_indices+keep_new_indices] = np.arange(indices_new_start, indices_new_start + keep_new_indices)
        cursor_indices += keep_new_indices

    # Normalize points
    drawing_max = points.max(axis=1)
    drawing_min = points.min(axis=1)
    size = drawing_max - drawing_min
    largest_dimension = size[:2].max()
    xy_scale = max(largest_dimension // 2, 1)
    time_scale = max(size[2], 1)
    middle = drawing_min + size / 2
    points = (points - middle.reshape((3, 1))) / np.array([[xy_scale], [xy_scale], [time_scale]])

    return torch.FloatTensor(points), torch.LongTensor(indices)

def load_model(model_path, device='cpu'):
    """Load the model with weights."""
    model = strokes_to_seresnext50_32x4d(
        img_size=32,     # Original image size
        window=2,        # Original window size
        num_classes=340  # Number of classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_drawing(model, drawing, device='cpu', top_k=5):
    """Predict classes for a drawing."""
    # Process the drawing
    points, indices = process_stroke_data(drawing)
    
    # Add batch dimension and move to device
    points = points.unsqueeze(0).to(device)
    indices = indices.unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        output = model(points, indices)
        prob = torch.softmax(output, dim=1)
        top_p, top_class = torch.topk(prob, k=top_k)
    
    # Get class names and probabilities
    idx_to_class = get_class_mapping()
    predictions = []
    for p, c in zip(top_p[0], top_class[0]):
        predictions.append({
            'class': idx_to_class[c.item()],
            'probability': p.item()
        })
    
    return predictions

def convert_canvas_to_strokes(canvas_data):
    """
    Convert canvas drawing data to stroke format.
    
    Args:
        canvas_data: JSON data from streamlit-drawable-canvas
        
    Returns:
        List of strokes, where each stroke is [x_coords, y_coords, time_coords]
    """
    if not canvas_data or "objects" not in canvas_data:
        return None
        
    strokes = []
    current_time = 0
    
    for obj in canvas_data["objects"]:
        if "path" in obj:
            # Initialize stroke arrays
            x_coords = []
            y_coords = []
            time_coords = []
            
            for i, cmd in enumerate(obj["path"]):
                if cmd[0] in ["M", "L"]:  # Move to or Line to
                    x_coords.append(float(cmd[1]))
                    y_coords.append(float(cmd[2]))
                    time_coords.append(current_time + i * 16)  # 60fps sampling (16ms)
            
            if x_coords:  # Only add strokes with points
                # Normalize coordinates to [0, 1] range
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_range = x_max - x_min if x_max > x_min else 1
                y_range = y_max - y_min if y_max > y_min else 1
                
                x_coords = [(x - x_min) / x_range for x in x_coords]
                y_coords = [(y - y_min) / y_range for y in y_coords]
                
                # Adjust time points to be relative to first point
                first_time = time_coords[0]
                time_coords = [t - first_time for t in time_coords]
                
                strokes.append([x_coords, y_coords, time_coords])
                current_time = time_coords[-1] + 100  # Add gap between strokes
    
    return strokes if strokes else None

if __name__ == "__main__":
    # Example usage
    model_path = "kaggle-quickdraw-weights.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path, device)
    
    # Example canvas data
    example_canvas_data = {
        "objects": [
            {
                "path": [
                    ["M", 100, 100],
                    ["L", 200, 200],
                    ["L", 300, 100]
                ]
            },
            {
                "path": [
                    ["M", 400, 400],
                    ["L", 500, 500]
                ]
            }
        ]
    }
    
    # Convert canvas data to strokes
    drawing = convert_canvas_to_strokes(example_canvas_data)
    
    if drawing:
        # Get predictions
        predictions = predict_drawing(model, drawing, device)
        
        # Print results
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['class']}: {pred['probability']*100:.2f}%")
    else:
        print("No valid drawing detected") 