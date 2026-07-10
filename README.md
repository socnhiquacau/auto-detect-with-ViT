# Person Detection and Recognition System

Backend API and frontend for detecting people in images and videos, then identifying them using YOLO and Vision Transformer (ViT) models.

## Features

- Upload an image or video for person detection and recognition.
- Process videos from a direct URL.
- Store detections and known-person feature vectors in MongoDB.
- Choose similarity matching or known/unknown classification.
- View detected person crops through the static API route.

## Frontend

### Home dashboard

The dashboard displays system readiness, known-person statistics, and the available image and video workflows.

![Home dashboard](bk2.png)

### Image recognition

The image workflow lets users upload a file, run detection and recognition, and review the matched people with similarity scores.

![Image recognition workflow](bk1.png)

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and configure MongoDB, paths, and model files.
4. Start the API:

   ```bash
   python -m uvicorn main:app --reload
   ```

The API documentation is available at `http://127.0.0.1:8000/docs`.
