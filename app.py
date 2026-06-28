import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from src.utils import configLogger
from src.prediction import (ImageProcessor,
                            BinaryPredictor,
                            MultiPredictor)

logger = configLogger("app","app.log")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # [STARTUP LOGIC]: Instantiates and caches the model component
        logger.info("Initializing application lifecycle.")
        img_processor = ImageProcessor()
        binary_pred = BinaryPredictor()
        multi_pred = MultiPredictor()
        
        yield {"img_processor": img_processor,
               "binary_pred" : binary_pred,
               "multi_pred" : multi_pred}
        
        # [SHUTDOWN LOGIC]: Executes when the server process terminates
        logger.info("Tearing down application infrastructure context.")
    
    except Exception as e:
        logger.exception("Error while creating lifespan : %s", e)
        raise

app = FastAPI(
    title="Kidney Tumor Detection API",
    description="server for screening ResNet50V2 predictions over CT scan images.",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
async def home():
    logger.info("Serving user-interface dashboard templates index.")
    try:
        template_path = Path("templates/index.html")
        return HTMLResponse(content=template_path.read_text(), status_code=200)
    except Exception as e:
        logger.error("Failed to render frontend template: %s", e)
        raise HTTPException(status_code=500, detail="User interface template missing on server.")


@app.post("/predict/binary", tags=["Inference Engine"])
async def predict_binary(request: Request, file: UploadFile = File(...)):

    # Guard check: Verify correct image extension signatures
    if not str(file.filename).lower().endswith(('.jpg', '.jpeg', '.png')):
        logger.warning("Rejected file upload with invalid extension: %s", file.filename)
        raise HTTPException(status_code=400, detail="Invalid file schema. Please upload a JPEG or PNG image.")
    
    try:
        processor = request.state.processor
        binary_model = request.state.binary_model
        
        image_bytes = await file.read()
        processed_tensor = processor.process(image_bytes)
        
        binary_result = binary_model.predict(processed_tensor)
        
        if binary_result["status"] == "error":
            raise HTTPException(status_code=500, detail=binary_result["message"])

        # Returns just the binary result. The frontend will use this to ask the user what to do next.
        return JSONResponse(content={"binary_screening": binary_result})
        
    except HTTPException as e:
        logger.exception("HTTP exception : %s", e)
        raise

    except Exception as e:
        logger.exception("Internal application error: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Failure: {str(e)}")


@app.post("/predict/detailed", tags=["Inference Engine"])
async def predict_detailed(request: Request, file: UploadFile = File(...)):
    if not str(file.filename).lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")
    
    try:
        processor = request.state.processor
        multi_model = request.state.multi_model
        
        image_bytes = await file.read()
        processed_tensor = processor.process(image_bytes)
        
        multi_result = multi_model.predict(processed_tensor)
        
        if multi_result["status"] == "error":
            raise HTTPException(status_code=500, detail=multi_result["message"])
                
        return JSONResponse(content={"detailed_diagnosis": multi_result})
        
    except Exception as e:
        logger.exception("Internal application error: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Failure: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

