# Face Detection & Recognition App
**COMP 6721 – Applied Artificial Intelligence | Winter 2026 | Concordia University**

---

## Project Description

This project is a real-time face detection and recognition application developed as part of the Applied AI course at Concordia University.

The application uses a webcam to detect faces in live video, recognize specific individuals it has been trained on, and label anyone it cannot confidently identify as **Unknown**.

The project covers the full AI pipeline: collecting a custom image dataset of at least 5 people (minimum 30 images each), preprocessing the images through cropping, resizing, normalization, and data augmentation, and training a Convolutional Neural Network (CNN) using either Keras or PyTorch. A face detector (such as OpenCV or MTCNN) handles real-time face localization, feeding cropped faces into the CNN which outputs a predicted identity and a confidence score. If the score falls below a chosen threshold, the person is marked as Unknown.

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

#### **Section 1: Preprocessing** (Cells 2-9)
- Runs face detection on raw images
- Crops, resizes, and augments images
- Splits data into train/val/test sets (70% / 15% / 15%)
- Output: Preprocessed images in `processed/` folder

```python
# Cell 2: Edit configuration if needed
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)

# Cells 3-9: Run preprocessing pipeline
```

#### **Section 2: Training** (Cells 10-20)
- Loads preprocessed data
- Builds and trains a 4-block CNN
- Applies early stopping and learning rate reduction
- Saves best model to `face_model.h5`
- Saves class mapping to `class_names.json`

```python
# Cells 10-20: Run training (typically 10-30 minutes)
# Best model is automatically saved and monitored via validation accuracy
```

#### **Section 3: Testing** (Cells 26-30)
- Loads the trained model
- Tests on validation/test sets
- Displays classification reports and confusion matrix
- Tests different confidence thresholds

```python
# Cells 26-30: Run evaluation (optional but recommended)
```

### Step 3: Run the Backend Server

Run Cell 31 (Debug + WebSocket Server) to start the real-time recognition backend:

```python
# Cell 31: Run this cell LAST and keep it running
# The server will start on ws://localhost:8765
```

You should see:
```
[*] Classes: {'0': 'Aafreen', '1': 'Syeda', '2': 'Taha'}
[*] Loading model from face_model.h5 ...
[✓] Model loaded successfully
───────────────────────────────────────────────────────
  WebSocket server  ->  ws://localhost:8765
  Classes           ->  ['Aafreen', 'Syeda', 'Taha']
  Threshold         ->  0.75
  Image size        ->  (224, 224)
───────────────────────────────────────────────────────
[*] Open frontend.html in your browser. Ctrl-C to stop.
```

### Step 4: Open the Frontend

Open `frontend.html` in your web browser:

1. Click **Start Camera** button
2. Allow webcam access
3. The app will stream frames to the backend
4. Detection results display in real-time with:
   - **Green boxes** = Known person (confidence ≥ 0.75)
   - **Pink boxes** = Unknown person (confidence < 0.75)
   - Bounding boxes, labels, and confidence scores

---

## Running Backend Separately

Once the model is trained and `face_model.h5` + `class_names.json` are saved, you can run just the backend server without retraining:

**Option 1: Quick start (minimal dependencies)**
```bash
# Install only essential packages
pip install tensorflow websockets nest_asyncio opencv-python numpy

# Run the server
python -c "
import jupyter_client
from ipykernel.kernelapp import IPKernelApp
# Run Cell 31 from the notebook
"
```

**Option 2: Use the notebook debug cell**
1. Run Cell labeled "DEBUG CELL" to verify model loads correctly
2. Then run Cell 31 to start the server

**Option 3: Extract as standalone script**
Create a file `server.py`:
```python
import asyncio, base64, json, cv2, numpy as np
import websockets, nest_asyncio
from tensorflow.keras.models import load_model

# (Copy the code from Cell 31 here)

asyncio.run(_main())
```

Then run:
```bash
python server.py
```

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

- The backend is **modular** → Can be reused for other applications once the model is trained
- The training notebook is self-contained → All preprocessing, training, and evaluation happens in one file
- Preprocessing includes **automatic data augmentation** → 7 variants per image in training set for better generalization
