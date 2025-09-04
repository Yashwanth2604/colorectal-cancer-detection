# Colorectal Cancer Detection using Pre-Trained Ensemble Algorithms

This project uses pre-trained deep learning models (ResNet, EfficientNet) combined through ensemble techniques to classify histopathological images for colorectal cancer detection with high accuracy and explainability.

## Features

- Ensemble of ResNet and EfficientNet for robust classification
- Explainable AI using Grad-CAM
- High accuracy on medical image datasets
- (Optional) Web interface using Django or Streamlit

## Tech Stack

- Python, PyTorch, Scikit-learn
- OpenCV, NumPy, Pandas
- Grad-CAM for model explainability
- (Optional) Django/Streamlit for UI

## Project Structure

```
.
├── src/                 # Scripts (train, predict, utils)
├── models/              # Saved models
├── data/                # Dataset samples (if public)
├── templates/           # UI templates (if Django)
├── requirements.txt     # Dependencies
├── .gitignore
└── README.md
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Run inference
python src/predict.py --image sample.jpg
```

## Results

- Accuracy: 95%+
- F1-Score: 94.7%
- ROC-AUC: 0.96

## Visualizations

![Grad-CAM example](static/gradcam_output.jpg)

## License

MIT License
