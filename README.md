# Face Detection & Recognition App
**COMP 6721 – Applied Artificial Intelligence | Winter 2026 | Concordia University**

---

## Project Description

This project is a real-time face detection and recognition application developed as part of the Applied AI course at Concordia University.

The application uses a webcam to detect faces in live video, recognize specific individuals it has been trained on, and label anyone it cannot confidently identify as **Unknown**.

The project covers the full AI pipeline: collecting a custom image dataset of at least 5 people (minimum 30 images each), preprocessing the images through cropping, resizing, normalization, and data augmentation, and training a Convolutional Neural Network (CNN) using either Keras or PyTorch. A face detector (such as OpenCV or MTCNN) handles real-time face localization, feeding cropped faces into the CNN which outputs a predicted identity and a confidence score. If the score falls below a chosen threshold, the person is marked as Unknown.

The final application displays a live webcam feed with bounding boxes around detected faces, the predicted name or "Unknown", and the associated confidence score.
