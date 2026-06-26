import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager
from src.utils import configLogger
from src.prediction import Predictor

logger = configLogger("app","app.log")

@asynccontextmanager
async def lifespan(app: FastAPI):

    # [STARTUP LOGIC]: Instantiates and caches the model component
    logger.info("Initializing application lifecycle.")
    pipeline_instance = Predictor()
    
    # Expose the pipeline resource to the app endpoints via state dictionary context
    yield {"pipeline": pipeline_instance}
    
    # [SHUTDOWN LOGIC]: Executes when the server process terminates
    logger.info("Tearing down application infrastructure context.")

app = FastAPI(
    title="Kidney Tumor Detection API",
    description="server for screening ResNet50V2 predictions over CT scan images.",
    version="0.0.1",
    lifespan=lifespan
)

@app.get("/")
async def home() -> dict:
    logger.info("hello from uvicorn")
    return {'message' : 'hellow from uvicorn'}

@app.post("/predict", tags=["Inference Engine"])
async def predict_ct_scan(request: Request, file: UploadFile = File(...)):

    # Guard check: Verify correct image extension signatures
    if not str(file.filename).lower().endswith(('.jpg', '.jpeg', '.png')):
        logger.warning("Rejected file upload with invalid extension: %s", file.filename)
        raise HTTPException(status_code=400, detail="Invalid file schema. Please upload a JPEG or PNG image.")
    
    try:
        pipeline = request.state.pipeline
        
        # Read incoming network stream bytes into system memory
        image_bytes = await file.read()

        result = pipeline.predict(image_bytes)
        
        if result["status"] == "error":
            logger.exception("Failed to predict : %s", result['message'])
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result
        
    except Exception as e:
        logger.exception("Exception : %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Failure: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

