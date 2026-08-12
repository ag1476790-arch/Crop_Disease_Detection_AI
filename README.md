# Crop Disease Detection

A Flask-based web application that uses deep learning to detect diseases in crop leaves. Upload an image of a plant leaf and get instant disease detection results with treatment recommendations.

## Features

- 🌿 Real-time crop disease detection using TensorFlow/Keras
- 📊 38 different plant disease classes supported
- 💾 Automatic history tracking of predictions
- 📈 Confidence scores for each prediction
- 🔧 Treatment guidance for detected diseases
- 🎨 Clean, responsive web interface

## Supported Crops

- Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/Crop_Desease_Detection.git
cd Crop_Desease_Detection
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application
```bash
python app.py
```

2. Open your browser and navigate to
```
http://127.0.0.1:5000
```

3. Upload a leaf image and click "Detect Disease"

4. View your prediction history at `/history`

## Project Structure

```
.
├── app.py                          # Main Flask application
├── crop_disease_model.keras    # Trained TensorFlow model
├── requirements.txt                # Python dependencies
├── template/                       # HTML templates
│   ├── index.html                 # Home page
│   ├── history.html               # History page
│   └── report.html                # Individual report view
└── static/                        # Static files
    ├── css/
    │   └── style.css              # Styling
    └── uploads/                   # Uploaded images
```

## Technologies Used

- **Backend**: Flask, Python
- **ML Framework**: TensorFlow, Keras
- **Frontend**: HTML5, CSS3
- **Database**: SQLite
- **Image Processing**: Pillow, NumPy

## Model Details

- **Model Architecture**: CNN (Convolutional Neural Network)
- **Input Size**: 224x224 pixels
- **Training Dataset**: PlantVillage Dataset
- **Classes**: 38 plant-disease combinations

## Features Explained

### Disease Detection
Upload any plant leaf image and the model will classify it into one of 38 categories, providing confidence scores.

### Treatment Guidance
Each detected disease comes with specific treatment recommendations based on agricultural best practices.

### History Tracking
All predictions are automatically saved to a local SQLite database, allowing you to review past detections.

## API Routes

- `GET /` - Home page with upload form
- `POST /predict` - Submit image for prediction
- `GET /history` - View all past predictions
- `GET /report/<id>` - View specific prediction report

## Future Enhancements

- [ ] Mobile app version
- [ ] Real-time camera feed support
- [ ] Additional crop types
- [ ] Model retraining with new data
- [ ] Multi-language support
- [ ] Export reports as PDF

## Contributing

Contributions are welcome! Feel free to fork and submit pull requests.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues or questions, please open an issue on GitHub.

---

**Note**: The model file (`crop_disease_model (1).keras`) should be downloaded separately if not included.
