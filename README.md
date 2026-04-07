# Face Detection & Recognition App
**COMP 6721 – Applied Artificial Intelligence | Winter 2026 | Concordia University**

---

## Project Description

This project is a real-time face detection and recognition application developed as part of the Applied AI course at Concordia University.

The application uses a webcam to detect faces in live video, recognize specific individuals it has been trained on, and label anyone it cannot confidently identify as **Unknown**.

The project covers the full AI pipeline: collecting a custom image dataset of at least 4-5 people (minimum 30 images each), preprocessing the images through cropping, resizing, normalization, and data augmentation, and training a Convolutional Neural Network (CNN) using TensorFlow/Keras. A face detector (such as OpenCV or MTCNN) handles real-time face localization, feeding cropped faces into the CNN which outputs a predicted identity and a confidence score. If the score falls below a chosen threshold, the person is marked as Unknown.

The final application displays a live webcam feed with bounding boxes around detected faces, the predicted name or "Unknown", and the associated confidence score.

---

## Dependencies

Before running the code, install the required Python packages:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn websockets nest_asyncio h5py seaborn
```

**Version requirements:**
- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.5+
- NumPy 1.20+

---

## Project Structure

```
Face-Detection-And-Recognition-App/
├── Local_Notebook.ipynb          # Main Jupyter notebook (preprocessing, training, testing)
├── frontend.html                 # Web-based frontend for real-time recognition
├── face_model.h5                 # Trained CNN model (added after training)
├── class_names.json              # Mapping of class indices to names
├── processed/                    # Preprocessed dataset (train/val/test splits)
│   ├── train/
│   ├── val/
│   └── test/
├── dataset/                      # Raw dataset folder (⚠️ NOT PUSHED TO GITHUB)
└── README.md
```

**Important:** The `dataset/` folder is listed in `.gitignore` and will not be pushed to GitHub. You must provide your own dataset or download it separately.

---

## How to Run

### Step 1: Prepare Your Dataset

1. Create a `dataset/` folder in the project root
2. Organize images by person:
   ```
   dataset/
   ├── Aafreen/
   │   ├── Aafreen_1.jpg
   │   ├── Aafreen_2.jpg
   │   └── ...
   ├── Syeda/
   │   ├── Syeda_1.jpg
   │   └── ...
   └── Taha/
       ├── Taha_1.jpg
       └── ...
   ```
   - Each person should have at least 30 images
   - Images can be `.jpg`, `.jpeg`, or `.png`

### Step 2: Open the Notebook

Open `Local_Notebook.ipynb` in Jupyter or VS Code and run cells sequentially:

#### **Section 1: Preprocessing**
```python
# Cell 2: Edit configuration if needed
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Then run cells 3+ to start preprocessing
```

#### **Section 2: Training**
```python
# Run cells to build CNN and train
# NUM_CLASSES = 6 (Aafreen, Harsha, Syeda, Arjun, Taha, Unknown)
# Best model auto-saves to face_model.h5
```

#### **Section 3: Testing & Evaluation**
```python
# Run cells to evaluate model on test set
# Includes confidence threshold analysis (default: 0.90)
```

### Step 3: Run the Backend Server

Run the WebSocket server cell to start the backend:

```python
# Run the backend cell and keep it running
# Server starts on ws://localhost:8765
```

Server output:
```
[*] Model loaded successfully.
  WebSocket server  ->  ws://localhost:8765
  Threshold         ->  0.90
[*] Open frontend.html in your browser
```

### Step 4: Open the Frontend

Open `frontend.html` in your web browser:

1. Click **Start Camera** button
2. Allow webcam access
3. The app will stream frames to the backend
4. Detection results display in real-time with:
   - **Green boxes** = Known person (confidence ≥ 0.90)
   - **Pink boxes** = Unknown person (confidence < 0.90)
   - Bounding boxes, labels, and confidence scores

---

## Running Backend Separately

Once `face_model.h5` and `class_names.json` are saved, run the backend server cell independently to start the WebSocket server without retraining.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model loads with wrong confidence (e.g., 37%) | Keras version mismatch. The notebook will auto-fallback using h5py. Check Cell 31 output logs. |
| Frontend shows "Unknown" for known faces | Lower the confidence threshold in `frontend.html` (line with `_THRESHOLD`) or retrain with more data. |
| No bounding boxes appear | Ensure WebSocket server is running and check browser console for errors. |
| Camera permission denied | Grant webcam access when browser prompts. |
| Out of memory during training | Reduce `BATCH_SIZE` in Cell 11 (e.g., from 16 to 8). |

---

## Notes

- Full preprocessing, training, and evaluation pipeline in one notebook
- Real-time WebSocket backend for live recognition
- Automatic data augmentation (7 variants per training image)
