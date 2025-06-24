# Visionary AI: Hacking Blindness Before It Begins

This project is an AI-based system to automatically detect up to 39 different fundus diseases and conditions from retinal photographs using deep learning, built for the "Visionary AI" hackathon.

## Features

- **Multi-Class Classification**: Detects 39 types of eye conditions.
- **Explainable AI (XAI)**: Implements Grad-CAM to visualize which parts of the image were important for the model's prediction.
- **Web Interface**: A simple, user-friendly UI to upload images and see results in real-time.
- **FastAPI Backend**: A high-performance Python web framework for serving the model.
- **Deployment Ready**: Includes a `Dockerfile` for easy containerization.

## Project Structure

```
visionary_ai/
├── backend/            # FastAPI source code
├── frontend/           # HTML/CSS for the UI
├── models/             # Contains the trained Keras model and class mappings
├── Dockerfile          # For containerization
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How to Run

### 1. Prerequisites

-  Python 3.10
- Docker (optional)

### 2. Local Setup and Execution

1.  **Clone the Repository and Navigate to the Directory:**
    ```bash
    git clone <your-repo-url>
    cd visionary_ai
    ```

2.  **Place Model Files:**
    Make sure your trained `eye_disease_model.h5` and `class_indices.json` are placed inside the `models/` folder.

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Web Application:**
    Navigate to the `backend` directory and start the server.
    ```bash
    cd backend
    uvicorn app:app --reload
    ```

5.  **Access the App:**
    Open your web browser and go to `http://127.0.0.1:8000`.

### 3. Running with Docker

1.  **Build the Docker Image:**
    From the root `visionary_ai` directory:
    ```bash
    docker build -t visionary-ai .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -p 8000:8000 visionary-ai
    ```

3.  **Access the App:**
    Open your web browser and go to `http://localhost:8000`.