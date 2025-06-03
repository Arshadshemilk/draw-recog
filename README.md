# Drawing Recognition App

This is a Streamlit application that uses a deep learning model to recognize drawings. The model is based on the SEResNeXt architecture and was trained on the Quick Draw dataset.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

The model weights will be automatically downloaded when you first run the application.

## Usage

1. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
2. Wait for the model weights to download (first run only)
3. Use the canvas to draw something
4. Click the "Recognize Drawing" button
5. The model will process your drawing and show the top 5 predictions

## Features

- Real-time drawing canvas
- Stroke-based drawing recognition
- Top 5 predictions with confidence scores
- Clean and intuitive user interface

## Technical Details

The application uses:
- Streamlit for the web interface
- PyTorch for the deep learning model
- SEResNeXt50 architecture with custom modifications for stroke processing
- streamlit-drawable-canvas for the drawing interface

## Model Architecture

The model processes drawings in the following way:
1. Captures drawing strokes as sequences of points
2. Resamples the strokes to a fixed number of points
3. Converts the strokes into a format suitable for the neural network
4. Uses a SEResNeXt50 architecture to classify the drawing

## Notes

- The model works best with clear, simple drawings
- Each stroke is processed separately and then combined
- The canvas size is fixed at 400x400 pixels for optimal performance 