# Real-Time Facial Emotion Recognition

A Python-based computer vision project that recognizes facial emotions through a webcam. The application uses a CNN model trained with TensorFlow/Keras and OpenCV for detecting faces and predicting emotions from live video.

The system recognizes five emotions: **Angry, Fear, Happy, Neutral, and Sad.**

## Project Highlights

- Custom CNN model built using TensorFlow and Keras.
- Trained on a dataset of 12,023 facial images.
- Supports classification across five emotion categories.
- Achieved 93.26% accuracy on the test dataset.
- Integrated with OpenCV for webcam-based predictions.

## Dataset Information

The dataset consists of facial images divided into training and testing sets.

| Category | Images |
|---|---:|
| Training Set | 9,619 |
| Testing Set | 2,404 |
| Total | 12,023 |

**Image specifications:**
- Input dimensions: 128 × 128 × 3
- Number of classes: 5
- Pixel values normalized to 0–1.

### Data Preparation

To prepare the images for training, the following preprocessing and augmentation techniques were applied:

- Image resizing to a uniform dimension.
- Pixel normalization.
- Random horizontal flipping.
- Shear transformation.
- Zoom augmentation.

## Model Development

A CNN architecture was implemented to learn facial patterns and distinguish between different expressions.

The network includes:
- Three convolutional layers with 32, 64, and 128 filters.
- Max pooling after each convolutional layer.
- Flatten layer for feature conversion.
- Fully connected layer with 128 neurons.
- Softmax output layer for five-class classification.

### Training Details

| Parameter | Value |
|---|---|
| Framework | TensorFlow/Keras |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Activation | ReLU, Softmax |
| Early Stopping | Patience = 7 |

## Model Evaluation

The trained model was evaluated using the separate test dataset.

<badge color="success">93.26% Test Accuracy</badge>

| Metric | Score |
|---|---:|
| Accuracy | 93.26% |
| Loss | 0.20 |

## How Real-Time Detection Works

The webcam application combines face detection and deep learning prediction.

**Processing flow:**

1. The webcam captures video frames.
2. Haar Cascade identifies faces within the frames.
3. Each detected face is extracted and resized.
4. The image is converted from BGR to RGB.
5. Pixel normalization is applied.
6. The CNN predicts the emotion.
7. The predicted label is displayed on the video frame.

The application continuously processes incoming frames until the user presses `q` to exit.

## Repository Structure

```text
Facial-Emotion-Recognition/
│
├── Emotion_detection.ipynb
├── real_time_emotion_prediction.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam
- pip package manager

### 1. Clone the repository

```bash
git clone https://github.com/username/repository-name.git
```

### 2. Open the project directory

```bash
cd repository-name
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the trained model

The trained model is stored separately because of its file size.

Download the model from the link below and place it in the project directory.

[Download Trained CNN Model](YOUR_MODEL_LINK_HERE)

### 5. Launch the application

```bash
python real_time_emotion_prediction.py
```

Allow webcam access when prompted.

## Libraries & Frameworks

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Possible Future Enhancements

- Experiment with transfer learning and pretrained CNN architectures.
- Improve face detection in challenging lighting conditions.
- Evaluate the model using a confusion matrix and class-wise performance metrics.
- Improve prediction speed for real-time applications.

---

*This project was developed as a practical exploration of deep learning and computer vision, covering the complete process from image preprocessing and model training to real-time deployment.*
