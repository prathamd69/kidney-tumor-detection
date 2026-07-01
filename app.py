import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import json
from src.utils import configLogger, loadYaml
from src.prediction import (ImageProcessor,
                            BinaryPredictor,
                            MultiPredictor)

logger = configLogger("app","app.log")

def read_descirpts() -> dict:

    try:
        config = loadYaml(Path("config/config.yaml"))
        reports_path = Path(config.descripts_path.class_descriptions_path)
        with open(reports_path, "r") as f:   
            class_descriptions = json.load(f)

        return class_descriptions
    
    except Exception as e:
        logger.exception("Failed to load class descriptions : %s", e)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # [STARTUP LOGIC]: Instantiates and caches the model component
        logger.info("Initializing application lifecycle.")
        img_processor = ImageProcessor()
        binary_pred = BinaryPredictor()
        multi_pred = MultiPredictor()

        class_descripts = read_descirpts()
        
        
        yield {"img_processor": img_processor,
               "binary_pred" : binary_pred,
               "multi_pred" : multi_pred,
               "class_descripts" : class_descripts}
        
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
        binary_pred = request.state.binary_pred
        
        image_bytes = await file.read()

        binary_result = binary_pred.predict(image_bytes)
        
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
        multi_pred = request.state.multi_pred
        
        image_bytes = await file.read()
        
        multi_result = multi_pred.predict(image_bytes)
        pred_class = multi_result["prediction"]

        descriptions = request.state.class_descripts
        report_text = descriptions.get(pred_class, "Description not available for this class.")
        multi_result["medical_description"] = report_text

        if multi_result["status"] == "error":
            raise HTTPException(status_code=500, detail=multi_result["message"])
                
        return JSONResponse(content={"detailed_diagnosis": multi_result})
        
    except Exception as e:
        logger.exception("Internal application error: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Failure: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

