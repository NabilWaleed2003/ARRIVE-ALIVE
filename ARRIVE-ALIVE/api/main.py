"""FastAPI inference server for drowsiness detection."""

import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging
from src.data.transforms import get_eval_transform
from src.utils import MODEL_SAVE_PATH, CLASS_NAMES, DEVICE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Driver Drowsiness Detection API",
    description="API for detecting drowsiness in driver images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = torch.jit.load(MODEL_SAVE_PATH)
    model = model.to(DEVICE)
    model.eval()
    logger.info(f"Model loaded successfully from {MODEL_SAVE_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Get transforms
transform = get_eval_transform()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Driver Drowsiness Detection API",
        "status": "active",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_available": model is not None
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict drowsiness from an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with prediction and confidence
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction_idx = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction_idx].item()
        
        prediction_label = CLASS_NAMES.get(prediction_idx, "Unknown")
        
        return JSONResponse(
            content={
                "prediction": prediction_label,
                "confidence": float(confidence),
                "class_index": prediction_idx,
                "probabilities": {
                    CLASS_NAMES[0]: float(probabilities[0][0].item()),
                    CLASS_NAMES[1]: float(probabilities[0][1].item()),
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict_batch/")
async def predict_batch(files: list = File(...)):
    """Predict drowsiness from multiple images.
    
    Args:
        files: List of image files
        
    Returns:
        JSON response with predictions for all images
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded."
        )
    
    predictions = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(DEVICE)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                prediction_idx = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][prediction_idx].item()
            
            predictions.append({
                "filename": file.filename,
                "prediction": CLASS_NAMES.get(prediction_idx, "Unknown"),
                "confidence": float(confidence),
            })
        
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            predictions.append({
                "filename": file.filename,
                "error": str(e),
            })
    
    return JSONResponse(content={"predictions": predictions})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
