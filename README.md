# Face Recognition System Using OpenCV & FaceNet

## Overview
This project implements a robust face recognition system that leverages OpenCV, MTCNN, and FaceNet (InceptionResnetV1) to detect, align, and recognize faces from images. The system extracts unique facial embeddings for known individuals and compares them to unknown faces for identification. It also includes performance evaluation metrics such as detection accuracy, recognition accuracy, processing speed (FPS), and average inference time.

## Features
- **Face Detection:** Uses MTCNN to accurately detect and align faces.
- **Feature Extraction:** Extracts 512-dimensional embeddings using FaceNet (InceptionResnetV1).
- **Face Recognition:** Compares embeddings using Euclidean distance for identity assignment.
- **Performance Evaluation:** Measures detection accuracy, recognition accuracy, FPS, and processing time.
- **Known Faces Database:** Organizes known faces in a structured directory for recognition.
- **Graphical User Interface (GUI):** Provides a user-friendly interface for image recognition, webcam-based processing, batch processing, and adding new persons.
- **HEIC Format Support:** Loads HEIC images using pillow-heif, alongside standard formats (JPEG, PNG).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Evaluation & Performance](#evaluation--performance)
- [Future Improvements](#future-improvements)
- [Requirements File](#requirements-file)
- [Contributors](#contributors)
- [License](#license)

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Git
- (Optional) Docker for containerization

### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/abhishiktmahajan/face_recognition_project.git
cd face_recognition_project
```

### Virtual Environment Setup
Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

Note: The requirements.txt includes packages such as opencv-python, numpy, torch, facenet-pytorch, Pillow, and pillow-heif for HEIC support.

## Usage

### Running the GUI
Launch the graphical user interface:

```bash
python gui.py
```

The GUI provides options to:

- **Open Image:** Load images (supports JPG, JPEG, PNG, HEIC) and process face recognition.
- **Start Webcam:** Use a live webcam feed for real-time face recognition.
- **Batch Process:** Process all images in a selected folder.
- **Add Person:** Enroll a new person by selecting one or more images.
- **Settings:** Adjust detection confidence and recognition threshold.
- **View Report:** Display performance evaluation metrics.

### Running the Command-Line Version (Optional)
If provided (e.g., in app.py), run:

```bash
python app.py --mode test --image test_images/test1.jpg
```

For more options, use:

```bash
python app.py --help
```

## Project Structure
```
face_recognition_project/
├── config.py                  # Configuration settings and default values
├── database.py                # FaceDatabase: manages known faces
├── evaluation_metrics.py      # FaceRecognitionEvaluator: collects metrics and reports
├── gui.py                     # Tkinter-based graphical user interface
├── app.py                     # (Optional) Command-line interface version
├── requirements.txt           # List of Python package dependencies
├── known_faces/               # Directory for known faces (organized by person)
├── test_images/               # Sample test images for recognition
└── README.md                  # Project documentation
```

## Methodology

### Face Detection & Alignment:
MTCNN detects faces and aligns them for consistent embedding extraction.

### Feature Extraction:
FaceNet computes a 512-dimensional embedding for each aligned face.

### Recognition:
The system compares input embeddings to those in the known faces database using Euclidean distance. If the distance is below a configurable threshold, the face is identified as a known person.

### Evaluation:
Performance metrics such as detection rate, recognition accuracy, average processing time, and FPS are recorded and reported.

## Evaluation & Performance
- **Detection Accuracy:** Percentage of faces detected versus expected.
- **Recognition Accuracy:** Percentage of correctly recognized faces.
- **Processing Speed:** Average time per image and overall frames per second.
- **Reporting:** Detailed reports are available via the GUI's "View Report" button.

## Future Improvements
- **Threshold Optimization:** Fine-tune recognition thresholds for improved accuracy.
- **Real-time Enhancements:** Optimize processing for higher FPS and lower latency.
- **Web Integration:** Convert the GUI to a web application using Flask or Streamlit.
- **Extended Format Support:** Add support for additional image formats and improve batch processing.

## Requirements File
The requirements.txt includes:

```
opencv-python
numpy
torch
facenet-pytorch
Pillow
pillow-heif
```

Add any additional packages as necessary.

## Contributors
Abhishikt Mahajan 
Email: abhishiktmahajan@gmail.com
LinkedIn: linkedin.com/in/abhishiktmahajan

## License
This project is licensed under the MIT License. See the LICENSE file for details.
