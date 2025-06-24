import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
import base64
import os

from helpers import predict, preprocess_image, get_model
from grad_cam import make_gradcam_heatmap, get_gradcam_overlay

# Initialize FastAPI app
app = FastAPI(title="Visionary AI - Eye Disease Detection")

# --- Mount static files (for the frontend) ---
# This allows the server to serve the index.html file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "../frontend")), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page for the frontend."""
    frontend_path = os.path.join(BASE_DIR, "../frontend/index.html")
    try:
        with open(frontend_path) as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.post("/predict_explain")
async def handle_predict_explain(file: UploadFile = File(...)):
    """
    Handles prediction and provides a Grad-CAM explanation.
    This is the main endpoint called by the frontend.
    """
    image_bytes = await file.read()

    try:
        model = get_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded on the server.")

        # Get prediction result
        prediction_result = predict(image_bytes)

        # Generate Grad-CAM explanation
        processed_image = preprocess_image(image_bytes)
        heatmap = make_gradcam_heatmap(processed_image, model)
        overlay_bytes = get_gradcam_overlay(image_bytes, heatmap)

        # Encode the overlay image to base64 to send in JSON
        overlay_base64 = base64.b64encode(overlay_bytes).decode('utf-8')

        # Combine results into a single JSON response
        return JSONResponse(content={
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "explainability_map": f"data:image/png;base64,{overlay_base64}"
        })

    except Exception as e:
        # Return a structured error message
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # This allows you to run the app with `python app.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)