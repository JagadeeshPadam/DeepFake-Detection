# Deployment Instructions

This guide explains how to run the Deepfake Detection Web UI locally and deploy it to Render.

## 🚀 Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r web_app/requirements.txt
   ```

2. **Run the Application**:
   ```bash
   cd web_app
   python app.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. **Access the UI**:
   Open your browser and navigate to `http://localhost:8000`.

## ☁️ Deploying to Render

1. **Prepare Repository**:
   Ensure your project structure looks like this:
   ```
   /
   ├── app.py
   ├── requirements.txt
   ├── models/
   │   ├── best_temporal_vit.pth
   │   └── model_metadata.json
   ├── static/
   │   ├── css/style.css
   │   └── js/main.js
   └── templates/
       └── index.html
   ```

2. **Create a Web Service on Render**:
   - Link your GitHub repository.
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**:
   Render provides a default Python environment. Ensure you have enough disk space for the model weights (~100MB).

> [!IMPORTANT]
> Since this application uses `torch` and `timm`, the free tier of Render might be slow or hit memory limits. For production use or project reviews, consider using a 'Starter' plan or a GPU-enabled instance.

## 🛠️ Performance Tips
- The model is loaded into memory only once at startup.
- Haar Cascade face detection is used for speed, falling back to center-crop if no face is found.
- 16 frames are sampled from the video for inference, balancing speed and temporal accuracy.
