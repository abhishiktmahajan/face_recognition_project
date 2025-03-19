# Face Recognition System Using OpenCV & FaceNet

## Overview
This project implements a robust face recognition system using OpenCV, MTCNN, and FaceNet (InceptionResnetV1). The system detects faces in images, extracts unique facial embeddings, and compares them to a database of known faces for recognition. It evaluates performance in terms of detection accuracy, recognition accuracy, and inference speed, ensuring reliability in various scenarios.

## Features
- **Face Detection:** Utilizes MTCNN for precise detection and alignment of faces.
- **Feature Extraction:** Leverages FaceNet (InceptionResnetV1) to generate 512-dimensional embeddings.
- **Face Recognition:** Compares embeddings using Euclidean distance for identity assignment.
- **Performance Evaluation:** Measures detection accuracy, processing speed (FPS), and inference time.
- **Known Faces Database:** Maintains embeddings of known individuals for recognition.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Evaluation & Performance](#evaluation--performance)
- [Future Improvements](#future-improvements)
- [Requirements File](#requirements-file)
- [Contributors](#contributors)
- [License](#license)

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Git (for version control)
- (Optional) Docker for containerization

### Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/abhishiktmahajan/face_recognition_project.git
cd face_recognition_project
