# ocr_service.py
from fastapi import FastAPI, HTTPException
from paddleocr import PaddleOCR
import uvicorn
import base64
import numpy as np
import cv2
from pydantic import BaseModel
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="PaddleOCR Service")

# Load models once at startup
PADDLEOCR_HOME = "C:/Users/User/.paddlex"
os.environ['PADDLEOCR_HOME'] = PADDLEOCR_HOME
os.environ['HUB_HOME'] = PADDLEOCR_HOME

script_dir = os.path.dirname(os.path.abspath(__file__))
ocr_en = PaddleOCR(paddlex_config=os.path.join(script_dir, "ocr_en_config.yaml"))
ocr_ar = PaddleOCR(paddlex_config=os.path.join(script_dir, "ocr_ar_config.yaml"))

# Single-threaded executor for serial OCR processing
ocr_executor = ThreadPoolExecutor(max_workers=1)

class OCRRequest(BaseModel):
    image_base64: str
    lang: str = "en"

def _run_ocr(image, lang):
    """Synchronous OCR execution"""
    model = ocr_ar if lang == "ar" else ocr_en
    return model.predict(image)

@app.post("/ocr")
async def perform_ocr(request: OCRRequest):
    try:
        # Decode image
        img_data = base64.b64decode(request.image_base64)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Run OCR in single-threaded executor (enforces serial execution)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(ocr_executor, _run_ocr, image, request.lang)
        
        # Convert to JSON-serializable format
        serializable_result = []
        if result and len(result) > 0:
            ocr_result_obj = result[0]
            
            if isinstance(ocr_result_obj, dict) and 'rec_texts' in ocr_result_obj:
                rec_texts = ocr_result_obj['rec_texts']
                rec_scores = ocr_result_obj['rec_scores']
                rec_polys = ocr_result_obj['rec_polys']

                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                    text_str = str(text) if not isinstance(text, str) else text
                    serializable_result.append([
                        poly.tolist() if hasattr(poly, 'tolist') else poly,
                        (text_str, float(score))
                    ])
                
            else:
                serializable_result = result
        
        return {"success": True, "result": [serializable_result]}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "models": {"en": True, "ar": True}}

if __name__ == "__main__":
    # workers=1 is critical
    uvicorn.run(app, host="0.0.0.0", port=8081, workers=1)