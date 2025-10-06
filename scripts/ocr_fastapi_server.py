"""
FastAPI OCR Server - High Performance Alternative to Flask
Designed for better async handling and ML model serving
"""

import os
import base64
import json
import logging
from typing import Dict, Optional, List, Tuple
import cv2
import numpy as np
from pathlib import Path
import re
import io
from PIL import Image
import time
import traceback
import requests
import asyncio
from functools import lru_cache
from collections import deque
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

_ocr_en_lock = None
_ocr_ar_lock = None

def get_ocr_locks():
    """Get or create OCR locks"""
    global _ocr_en_lock, _ocr_ar_lock
    if _ocr_en_lock is None:
        _ocr_en_lock = asyncio.Lock()
    if _ocr_ar_lock is None:
        _ocr_ar_lock = asyncio.Lock()
    return _ocr_en_lock, _ocr_ar_lock

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UAE License Plate OCR API",
    description="High-performance OCR service for UAE license plates",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PlateAnalysisRequest(BaseModel):
    base64_data: Optional[str] = Field(None, description="Base64 encoded image data")
    plate_cropped_image: Optional[str] = Field(None, description="Alternative field name for base64 data")
    plateimage: Optional[str] = Field(None, description="Alternative field name for base64 data")
    image: Optional[str] = Field(None, description="Alternative field name for base64 data")

class PlateAnalysisResponse(BaseModel):
    category: Optional[str] = Field(None, description="License plate category (letter or number)")
    state: Optional[str] = Field(None, description="UAE emirate name")
    number: Optional[str] = Field(None, description="License plate number")
    full_text: Optional[str] = Field(None, description="Complete plate text")
    confidence: Optional[float] = Field(None, description="Overall confidence score")
    plate_color: Optional[str] = Field(None, description="Detected plate background color")
    color_confidence: Optional[float] = Field(None, description="Color detection confidence")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type based on plate color")
    method: Optional[str] = Field(None, description="Processing method used")
    emirate_detected: Optional[str] = Field(None, description="Emirate detected during preprocessing")
    emirate_confidence: Optional[float] = Field(None, description="Emirate detection confidence")
    error: Optional[str] = Field(None, description="Error message if processing failed")

class HealthResponse(BaseModel):
    status: str
    ocr_models_loaded: Dict[str, bool]
    ollama_available: bool
    ollama_model: str
    timestamp: float

class FastAPIUAEPlateAnalyzer:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initialize with async-friendly PaddleOCR models and Ollama for Qwen2.5-VL"""

        PADDLEOCR_HOME = "C:/Users/User/.paddlex"
        os.environ['PADDLEOCR_HOME'] = PADDLEOCR_HOME
        os.environ['HUB_HOME'] = PADDLEOCR_HOME

        # self.model_path = model_path
        self.ollama_url = ollama_url
        self.ollama_model = "qwen2.5vl:7b"
        # self.ocr_model_en = None
        # self.ocr_model_ar = None

        # self.ocr_en_lock = asyncio.Lock()
        # self.ocr_ar_lock = asyncio.Lock()

        # from concurrent.futures import ThreadPoolExecutor
        # self.ocr_en_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr_en")
        # self.ocr_ar_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr_ar")

        # self.ocr_en_lock, self.ocr_ar_lock = get_ocr_locks()

        # OCR service URL
        self.ocr_service_url = os.getenv("OCR_SERVICE_URL", "http://localhost:8081")

        # Semaphore for Ollama rate limiting (max 4 concurrent)
        self.ollama_semaphore = asyncio.Semaphore(4)

        # Request queue (max 10 waiting)
        self.request_queue = asyncio.Queue(maxsize=10)

        # Queue statistics
        self.queue_stats = {
            'total_processed': 0,
            'total_rejected': 0,
            'wait_times': [],
            'current_processing': 0
        }
        self.stats_lock = asyncio.Lock()
        
        # Initialize models in a thread-safe way
        # self._load_ocr_models()
        self._check_ollama_connection()
        
        # Emirates mapping for normalization
        self.emirate_mappings = {
            'abu dhabi': 'Abu Dhabi',
            'a.d': 'Abu Dhabi', 
            'ad': 'Abu Dhabi',
            'dubai': 'Dubai',
            'ajman': 'Ajman',
            'sharjah': 'Sharjah',
            'rak': 'Ras Al Khaimah',
            'ras al khaimah': 'Ras Al Khaimah',
            'fujairah': 'Fujairah',
            'umm al quwain': 'Umm Al Quwain'
        }

        self.emirate_to_code ={
            'Abu Dhabi': 'AUH',
            'Dubai': 'DXB',
            'Sharjah': 'SHJ',
            'Ajman': 'AJM',
            'Fujairah': 'FUJ',
            'Ras Al Khaimah': 'RAK',
            'Umm Al Quwain': 'UAQ'
        }
        
        # Valid categories for each emirate
        self.valid_categories = {
            'Abu Dhabi': ['1', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '50'],
            'Sharjah': ['1', '2', '3', '4'],
            'Dubai': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'Ajman': 'ABCDEH',
            'Ras Al Khaimah': 'ABCDIKMNSVX',
            'Fujairah': 'ABCDEFGKMPRSTX',
            'Umm Al Quwain': 'ABCDEFGHIX'
        }
        
        # Plate color to vehicle type mapping
        self.vehicle_type_mapping = {
            'white': 'Private Vehicles',
            'yellow': 'Taxis',
            'red': 'Commercial Vehicles',
            'blue': 'Police Vehicles',
            'green': 'Rental Vehicles',
            'black': 'Temporary Plates',
            'orange': 'Sharjah Special Plates',
            'gold': 'VIP / Personalized',
            'custom': 'VIP / Personalized'
        }

        # System prompt for Ollama Qwen2.5-VL
#         self.qwen_system_prompt = """
# You are an expert in UAE license plate recognition. Your task is to analyze license plate images and extract accurate information.

# === UAE EMIRATE FORMATS ===

# Letter-based categories (A-Z):
# - Dubai: Letter + "DUBAI" + Number
# - Ajman: Letter + "AJMAN" + Number  
# - Ras Al Khaimah: Letter + "UAE.RAK" or "RAS AL KHAIMAH" + Number
# - Fujairah: Letter + "UAE" + الفجيرة (Arabic text) + Number
# - Umm Al Quwain: Letter + أم القيوين (Arabic text only) + Number

# Number-based categories (1-50):
# - Abu Dhabi: Number (1,4-21,50) + "Abu Dhabi" or "A.D" + Number
# - Sharjah: Number (1-4) + "SHARJAH" + Number

# === CRITICAL READING RULES ===

# 1. READ ALL TEXT - Don't stop at the first word you see
# 2. "UAE" ALONE IS INVALID - If you see "UAE", there must be additional text:
#    - "UAE" + الفجيرة → Fujairah
#    - "UAE.RAK" or "UAE" + "RAK" → Ras Al Khaimah
# 3. NEVER return "UAE" as the state name
# 4. Arabic text indicates the emirate - convert to English in your response:
#    - الفجيرة → Fujairah
#    - أم القيوين → Umm Al Quwain
#    - الشارقة → Sharjah
#    - دبي → Dubai
#    - عجمان → Ajman
#    - رأس الخيمة → Ras Al Khaimah
#    - أبو ظبي → Abu Dhabi

# === PLATE COLORS ===

# Identify background color: white, yellow, red, blue, green, black, orange, gold

# === JSON OUTPUT SCHEMA ===

# Return ONLY valid JSON with these exact field names:
# - "category": string - The category letter (A-Z) or number (1-50) you observe
# - "state": string - The specific emirate name in English (Fujairah, Dubai, Abu Dhabi, Sharjah, Ajman, Ras Al Khaimah, Umm Al Quwain)
# - "number": string - The plate number digits you observe
# - "confidence": number - Your confidence score between 0.0 and 1.0
# - "plate_color": string - The background color you observe

# CRITICAL: Extract actual values from the image, not examples or placeholders. No Arabic text in the final JSON - always convert to English emirate names. No "UAE" as state value.
# """
        self.qwen_system_prompt = """
UAE license plate OCR expert. Extract: category, state, number, confidence, plate_color as JSON.

PLATE FORMATS:
Dubai/Ajman/RAK/Fujairah/UAQ: Letter category + Emirate + Number
Abu Dhabi: Number category (1,4-21,50) + "Abu Dhabi"/"A.D" + Number
Sharjah: Number category (1-4) + "SHARJAH" + Number

PLATE FORMAT DISAMBIGUATION: For Abu Dhabi and Sharjah plates, you will see two numbers. If the number nearer to the left/top of the plate is a valid number category specified above, then it is the category, and the number on the right/bottom of the plate is the plate number.

CRITICAL RULE - "UAE" text handling:
When you see "UAE" on the plate, it's NOT the complete emirate name. You MUST look for additional text:
- If "UAE" appears with الفجيرة (Arabic) nearby → state is "Fujairah"
- If "UAE.RAK" or "UAE" with "RAK" → state is "Ras Al Khaimah"
- "UAE" alone is NEVER a valid state value
The specific emirate is always indicated by additional text beyond just "UAE".

ARABIC TO ENGLISH:
الفجيرة=Fujairah, أم القيوين=Umm Al Quwain, الشارقة=Sharjah, دبي=Dubai, عجمان=Ajman, رأس الخيمة=Ras Al Khaimah, أبو ظبي=Abu Dhabi

OUTPUT: JSON containing:
- category (a letter or a number),
- state (specific emirate name, NEVER "UAE"),
- number (1 to 5 digits),
- confidence (0.0-1.0),
- plate_color (white/yellow/red/blue/green/black/orange/gold).
"""
        # self._warm_up_models()

    def _warm_up_models(self):
        """Pre-warm OCR models to avoid first-request delays"""
        try:
            if self.ocr_model_en is not None:
                logger.info("Warming up English OCR model...")
                # Create small dummy image
                dummy_image = np.zeros((50, 200, 3), dtype=np.uint8)
                dummy_image.fill(255)  # White background
                self.ocr_model_en.predict(dummy_image)
                logger.info("English OCR model warmed up")
                
            if self.ocr_model_ar is not None:
                logger.info("Warming up Arabic OCR model...")
                dummy_image = np.zeros((50, 200, 3), dtype=np.uint8) 
                dummy_image.fill(255)
                self.ocr_model_ar.predict(dummy_image)
                logger.info("Arabic OCR model warmed up")
                
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _load_ocr_models(self):
        """Load PaddleOCR models using config files for offline operation"""
        try:
            from paddleocr import PaddleOCR
            import os
            
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Paths to config files (should be in same directory as this script)
            en_config_path = os.path.join(script_dir, "ocr_en_config.yaml")
            ar_config_path = os.path.join(script_dir, "ocr_ar_config.yaml")
            
            # Check if config files exist
            if not os.path.exists(en_config_path):
                raise FileNotFoundError(f"English config not found: {en_config_path}")
            if not os.path.exists(ar_config_path):
                raise FileNotFoundError(f"Arabic config not found: {ar_config_path}")
            
            logger.info(f"Loading English OCR model from config: {en_config_path}")
            
            # Load English model from config file
            self.ocr_model_en = PaddleOCR(paddlex_config=en_config_path)
            logger.info("PaddleOCR English model loaded successfully")
            
            logger.info(f"Loading Arabic OCR model from config: {ar_config_path}")
            
            # Load Arabic model from config file
            self.ocr_model_ar = PaddleOCR(paddlex_config=ar_config_path)
            logger.info("PaddleOCR Arabic model loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Config file missing: {e}")
            logger.error("Make sure ocr_en_config.yaml and ocr_ar_config.yaml are in the same directory")
            self.ocr_model_en = None
            self.ocr_model_ar = None
        except Exception as e:
            logger.error(f"Failed to load OCR models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.ocr_model_en = None
            self.ocr_model_ar = None

    def _check_ollama_connection(self):
        """Check if Ollama is running and has the required model"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"Ollama is running. Available models: {available_models}")
                
                if self.ollama_model in available_models:
                    logger.info(f"Required model {self.ollama_model} is available")
                    self.ollama_available = True
                else:
                    logger.warning(f"Required model {self.ollama_model} not found. Available: {available_models}")
                    self.ollama_available = False
            else:
                logger.warning(f"Ollama responded with status {response.status_code}")
                self.ollama_available = False
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama: {e}")
            self.ollama_available = False

    async def call_ollama_qwen_async(self, image_base64: str, max_retries: int = 3) -> Dict:
        """Async call to Ollama with Qwen2.5-VL for license plate analysis"""
        if not self.ollama_available:
            return {"error": "Ollama service not available"}
        
        async with self.ollama_semaphore:
            # user_prompt = f"""
# TASK: Extract license plate information from this image completely and accurately, and return it as JSON.

# === STEPS ===

# STEP 1 - Identify all visible text:
# - Look for English text (letters, words, numbers)
# - Look for Arabic text (script characters)
# - Note the spatial arrangement (left to right)Step 2: Determine the category (leftmost letter or number)

# STEP 2 - Determine the emirate:
# - If you see "UAE" alone: This is incomplete - look for text next to it
#   - "UAE" with الفجيرة nearby → Emirate is Fujairah
#   - "UAE.RAK" or "UAE" with "RAK" → Emirate is Ras Al Khaimah
# - If you see "DUBAI", "AJMAN", etc. → Use that emirate name
# - If you see Arabic only (أم القيوين) → Convert to English (Umm Al Quwain)

# STEP 3 - Extract components:
# - Category: Leftmost letter or number
# - Emirate: Use rules from Step 2 (never "UAE")
# - Number: Rightmost digit sequence
# - Color: Background color of plate

# STEP 4 - Output JSON following the exact JSON OUTPUT SCHEMA:
# - "category": string - The category letter (A-Z) or number (1-50) you observe
# - "state": string - The specific emirate name in English (Fujairah, Dubai, Abu Dhabi, Sharjah, Ajman, Ras Al Khaimah, Umm Al Quwain)
# - "number": string - The plate number digits you observe
# - "confidence": number - Your confidence score between 0.0 and 1.0
# - "plate_color": string - The background color you observe

# === EMIRATE IDENTIFICATION ===

# CRITICAL: If you see "UAE" in English:
# - This is NOT the complete emirate name
# - Look for additional text next to it:
#   - Arabic text الفجيرة after "UAE" = Fujairah
#   - ".RAK" or "RAK" after "UAE" = Ras Al Khaimah

# Arabic text recognition:
# - الفجيرة = Fujairah
# - أم القيوين = Umm Al Quwain
# - الشارقة = Sharjah
# - دبي = Dubai
# - عجمان = Ajman
# - رأس الخيمة = Ras Al Khaimah
# - أبو ظبي = Abu Dhabi

# ==============================

# ALWAYS remember: 
# - "UAE" is not a valid emirate value. Always identify the specific emirate.
# - Extract the ACTUAL values you see - no defaults, no examples, no "UAE" as state.

# Timestamp: {time.time()}
#         """
            user_prompt = f"""
Extract plate info as JSON. Read ALL text - if you see "UAE" or "U.A.E", look for text next to it (الفجيرة=Fujairah, RAK=Ras Al Khaimah). Convert all Arabic to English. 
Return: category, state (the specific emirate, never "UAE"), number, confidence, plate_color as stated earlier.
"""
                    
            for attempt in range(max_retries):
                try:
                    logger.info(f"┌─── Ollama Qwen2.5-VL Analysis (Attempt {attempt + 1}/{max_retries}) ───┐")
                    
                    payload = {
                        "model": self.ollama_model,
                        "prompt": self.qwen_system_prompt + "\n\n" + user_prompt,
                        "images": [image_base64],
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0,
                            "top_p": 0.9
                        }
                    }
                    
                    logger.info(f"│ Sending request to Ollama...")
                    logger.debug(f"│ Image base64 length: {len(image_base64)} characters")
                    
                    # Use asyncio to make HTTP request non-blocking                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.ollama_url}/api/generate",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=300)
                        ) as response:
                            
                            logger.info(f"│ Ollama HTTP Status: {response.status}")
                            
                            if response.status == 200:
                                result = await response.json()
                                generated_text = result.get("response", "")
                                
                                logger.info(f"│ ✓ Ollama Response Received")
                                logger.info(f"│ Raw Response Length: {len(generated_text)} chars")
                                logger.info(f"│")
                                logger.info(f"│ ─── FULL QWEN RESPONSE ───")
                                logger.info(f"│ {generated_text}")
                                logger.info(f"│ ─── END QWEN RESPONSE ───")
                                logger.info(f"│")
                                
                                # Parse JSON from response
                                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', generated_text, re.DOTALL)
                                if json_match:
                                    try:
                                        json_str = json_match.group()
                                        logger.info(f"│ Extracted JSON: {json_str}")
                                        
                                        parsed_result = json.loads(json_str)
                                        
                                        logger.info(f"│")
                                        logger.info(f"│ ─── QWEN EXTRACTED DATA ───")
                                        logger.info(f"│ Category:     '{parsed_result.get('category', 'N/A')}'")
                                        logger.info(f"│ State:        '{parsed_result.get('state', 'N/A')}'")
                                        logger.info(f"│ Number:       '{parsed_result.get('number', 'N/A')}'")
                                        logger.info(f"│ Confidence:   {parsed_result.get('confidence', 'N/A')}")
                                        logger.info(f"│ Plate Color:  '{parsed_result.get('plate_color', 'N/A')}'")
                                        logger.info(f"│ ───────────────────────────")
                                        logger.info(f"│")

                                        emirate_full_name = parsed_result.get("state", "")
                                        state_code = self.map_emirate_to_code(emirate_full_name)
                                        
                                        logger.info(f"│ Emirate Mapping:")
                                        logger.info(f"│   Input:  '{emirate_full_name}'")
                                        logger.info(f"│   Output: '{state_code}'")
                                        
                                        normalized_result = {
                                            "category": parsed_result.get("category", ""),
                                            "state": state_code,
                                            "number": parsed_result.get("number", ""),
                                            "confidence": parsed_result.get("confidence", 0.95),
                                            "plate_color": parsed_result.get("plate_color", "white").lower(),
                                            "method": "ollama_qwen2.5vl_async",
                                            "vehicle_type": self.get_vehicle_type(parsed_result.get("plate_color", "white").lower())
                                        }
                                        
                                        # Generate full_text if we have components
                                        if normalized_result["category"] and normalized_result["state"] and normalized_result["number"]:
                                            normalized_result["full_text"] = self.generate_full_text(
                                                normalized_result["category"], normalized_result["state"], normalized_result["number"]
                                            )
                                        
                                        logger.info(f"│")
                                        logger.info(f"│ ─── FINAL NORMALIZED RESULT ───")
                                        logger.info(f"│ Category:     '{normalized_result['category']}'")
                                        logger.info(f"│ State:        '{normalized_result['state']}'")
                                        logger.info(f"│ Number:       '{normalized_result['number']}'")
                                        logger.info(f"│ Full Text:    '{normalized_result.get('full_text', 'N/A')}'")
                                        logger.info(f"│ Confidence:   {normalized_result['confidence']}")
                                        logger.info(f"│ Plate Color:  '{normalized_result['plate_color']}'")
                                        logger.info(f"│ Vehicle Type: '{normalized_result['vehicle_type']}'")
                                        logger.info(f"│ Method:       '{normalized_result['method']}'")
                                        logger.info(f"└───────────────────────────────────────────────────┘")
                                        
                                        return normalized_result
                                        
                                    except json.JSONDecodeError as e:
                                        logger.error(f"│ ✗ JSON Decode Error: {e}")
                                        logger.error(f"│ Failed to parse: {json_str[:200]}")
                                        logger.info(f"└───────────────────────────────────────────────────┘")
                                        return {"error": f"Invalid JSON in Ollama response: {e}", "raw_response": generated_text}
                                else:
                                    logger.error(f"│ ✗ No JSON found in Ollama response")
                                    logger.error(f"│ Response was: {generated_text[:500]}")
                                    logger.info(f"└───────────────────────────────────────────────────┘")
                                    return {"error": "No JSON found in Ollama response", "raw_response": generated_text}
                            
                            else:
                                error_text = await response.text()
                                logger.error(f"│ ✗ Ollama HTTP Error: {response.status}")
                                logger.error(f"│ Error details: {error_text[:500]}")
                                logger.info(f"└───────────────────────────────────────────────────┘")
                                return {"error": f"Ollama request failed: {response.status} - {error_text[:500]}"}
                        
                except asyncio.TimeoutError:
                    logger.warning(f"│ ⏱ Timeout on attempt {attempt + 1}/{max_retries}")
                    logger.info(f"└───────────────────────────────────────────────────┘")
                    if attempt == max_retries - 1:
                        return {"error": "Ollama timeout after multiple attempts"}
                    await asyncio.sleep(5)
                    continue
                except Exception as e:
                    logger.error(f"│ ✗ Exception: {e}")
                    logger.error(f"│ Traceback: {traceback.format_exc()}")
                    logger.info(f"└───────────────────────────────────────────────────┘")
                    return {"error": str(e)}
            
            logger.info(f"└───────────────────────────────────────────────────┘")
            return {"error": "Max retries exceeded"}

    def detect_plate_color(self, image: np.ndarray) -> Tuple[str, float]:
        """Detect plate background color using computer vision with confidence scoring"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define expanded color ranges for UAE plate colors (HSV format)
            color_ranges = {
                'white': {
                    'range': ([0, 0, 160], [180, 40, 255]),
                    'priority': 1
                },
                'red': {
                    'range': ([0, 100, 100], [15, 255, 255]),
                    'priority': 3
                },
                'red_alt': {
                    'range': ([165, 100, 100], [180, 255, 255]),
                    'priority': 3,
                    'maps_to': 'red'
                },
                'blue': {
                    'range': ([90, 80, 80], [140, 255, 255]),
                    'priority': 3
                },
                'yellow': {
                    'range': ([15, 80, 120], [35, 255, 255]),
                    'priority': 2
                },
                'black': {
                    'range': ([0, 0, 0], [180, 255, 60]),
                    'priority': 2
                },
                'green': {
                    'range': ([35, 80, 80], [85, 255, 255]),
                    'priority': 3
                },
                'orange': {
                    'range': ([8, 100, 120], [25, 255, 255]),
                    'priority': 3
                },
                'gold': {
                    'range': ([20, 120, 150], [35, 255, 255]),
                    'priority': 3
                }
            }
            
            total_pixels = image.shape[0] * image.shape[1]
            color_scores = {}
            
            # Analyze each color range
            for color_name, color_info in color_ranges.items():
                lower_bound = np.array(color_info['range'][0])
                upper_bound = np.array(color_info['range'][1])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                color_pixels = np.sum(mask > 0)
                percentage = (color_pixels / total_pixels) * 100
                
                # Map alternative names to primary colors
                final_color = color_info.get('maps_to', color_name)
                
                # Weight by priority and percentage
                weighted_score = percentage * color_info['priority']
                
                if final_color in color_scores:
                    color_scores[final_color] = max(color_scores[final_color], weighted_score)
                else:
                    color_scores[final_color] = weighted_score
            
            if not color_scores:
                return 'white', 0.0
            
            # Find best color match
            best_color = max(color_scores, key=color_scores.get)
            best_score = color_scores[best_color]
            
            # Calculate confidence based on dominance
            confidence = min(best_score / 100.0, 1.0)
            
            # Minimum confidence threshold
            if confidence < 0.15:
                return 'white', 0.5
            
            # Special case: if multiple colors have similar scores, reduce confidence
            sorted_scores = sorted(color_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[1] > sorted_scores[0] * 0.7:
                confidence *= 0.7
            
            return best_color, confidence
            
        except Exception as e:
            logger.error(f"Color detection failed: {e}")
            return 'white', 0.0

    def generate_full_text(self, category: str, state: str, number: str) -> str:
        """Generate full text programmatically in consistent format"""
        # State is now a 3-letter code like 'AUH', 'DXB', etc.
        return f"{category} {state} {number}"

    def get_vehicle_type(self, plate_color: str) -> str:
        """Get vehicle type based on plate color"""
        return self.vehicle_type_mapping.get(plate_color, 'Unknown Vehicle Type')

    async def detect_emirate_async(self, image: np.ndarray) -> Tuple[str, float]:
        # """Async emirate detection using PaddleOCR models"""
        # try:
        #     logger.info(f"Starting async emirate detection")
            
        #     # Try English model only
        #     emirate_en, conf_en = "unknown", 0.0
        #     if self.ocr_model_en is not None:
        #         try:
        #             logger.info(f"Running English OCR predict async...")
                    
        #             # Run OCR in thread pool to avoid blocking the event loop
        #             loop = asyncio.get_event_loop()
        #             ocr_results_en = await loop.run_in_executor(
        #                 None, self.ocr_model_en.predict, image
        #             )
                    
        #             logger.info(f"English OCR predict completed")
        #             emirate_en, conf_en = self.parse_emirate_results(ocr_results_en, "en")
        #             logger.info(f"English OCR detected: {emirate_en} (conf: {conf_en:.2f})")
        #         except Exception as e:
        #             logger.error(f"English OCR failed: {e}")
            
        #     # Return English result if found, otherwise unknown
        #     if emirate_en != "unknown" and conf_en > 0.6:
        #         logger.info(f"Using English OCR result: {emirate_en}")
        #         return emirate_en, conf_en
        #     else:
        #         logger.info("English OCR failed to detect emirate")
        #         return "unknown", 0.0
                
        # except Exception as e:
        #     logger.error(f"Async emirate detection failed: {e}")
        #     return "unknown", 0.0
        """Async emirate detection using external OCR service"""
        try:
            logger.info("Starting emirate detection via OCR service")
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode()
            
            # Call external OCR service (English only for emirate detection)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ocr_service_url}/ocr",
                    json={"image_base64": img_base64, "lang": "en"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"OCR service error: {resp.status}")
                        return "unknown", 0.0
                    
                    ocr_data = await resp.json()
                    ocr_results = ocr_data['result']
            
            # Parse emirate from results
            emirate, conf = self.parse_emirate_results(ocr_results, "en")
            logger.info(f"Emirate detected: {emirate} (conf: {conf:.2f})")
            
            return emirate, conf
            
        except Exception as e:
            logger.error(f"Emirate detection failed: {e}")
            return "unknown", 0.0

    def parse_emirate_results(self, ocr_results, lang: str) -> Tuple[str, float]:
        """Parse OCR results to detect emirate"""
        try:
            if not ocr_results or not ocr_results[0]:
                return "unknown", 0.0
            
            # OCR service returns format: [[bbox, (text, score)], ...]
            all_text = []
            for line in ocr_results[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, confidence = text_info[0], text_info[1]
                        all_text.append((str(text).lower().strip(), float(confidence)))
                    elif isinstance(text_info, str):
                        all_text.append((text_info.lower().strip(), 0.8))
            
            # Check for emirate indicators
            emirate_scores = {}
            
            for text, conf in all_text:
                if lang == 'en':
                    if any(indicator in text for indicator in ['abu dhabi', 'a.d', 'abu', 'dhabi']):
                        emirate_scores['Abu Dhabi'] = max(emirate_scores.get('Abu Dhabi', 0), conf)
                    elif 'dubai' in text:
                        emirate_scores['Dubai'] = max(emirate_scores.get('Dubai', 0), conf)
                    elif 'sharjah' in text:
                        emirate_scores['Sharjah'] = max(emirate_scores.get('Sharjah', 0), conf)
                    elif 'ajman' in text:
                        emirate_scores['Ajman'] = max(emirate_scores.get('Ajman', 0), conf)
                    elif any(indicator in text for indicator in ['rak', 'ras al khaimah', 'uae.rak']):
                        emirate_scores['Ras Al Khaimah'] = max(emirate_scores.get('Ras Al Khaimah', 0), conf)
            
            if emirate_scores:
                best_emirate = max(emirate_scores, key=emirate_scores.get)
                confidence = emirate_scores[best_emirate]
                return best_emirate, confidence
            
            return "unknown", 0.0
        
        except Exception as e:
            logger.error(f"Error parsing {lang} emirate results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "unknown", 0.0

    async def analyze_plate_from_base64_async(self, base64_data: str) -> Dict:
        """Main analysis with queue management"""
        queue_id = f"req_{int(time.time() * 1000)}"
        enqueue_time = time.time()
        
        try:
            # Try to add to queue (timeout 5 seconds)
            try:
                await asyncio.wait_for(
                    self.request_queue.put((queue_id, base64_data, enqueue_time)),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                async with self.stats_lock:
                    self.queue_stats['total_rejected'] += 1
                return {"error": "Server is busy. Request queue is full."}
            
            # Process from queue
            queued_id, queued_data, queued_time = await self.request_queue.get()
            
            try:
                async with self.stats_lock:
                    self.queue_stats['current_processing'] += 1
                    wait_time = time.time() - queued_time
                    self.queue_stats['wait_times'].append(wait_time)
                    if len(self.queue_stats['wait_times']) > 100:
                        self.queue_stats['wait_times'].pop(0)
                
                # Call your existing processing logic
                result = await self._process_plate(queued_data)
                
                async with self.stats_lock:
                    self.queue_stats['total_processed'] += 1
                    self.queue_stats['current_processing'] -= 1
                
                return result
            finally:
                self.request_queue.task_done()
                
        except Exception as e:
            logger.error(f"Queue error: {e}")
            return {"error": str(e)}
        
    async def _process_plate(self, base64_data: str):
        """Async analyze plate from base64 string using hybrid approach"""
        try:
            # Decode base64 to image
            image_bytes = base64.b64decode(base64_data)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            logger.debug(f"Image decoded from base64: {cv_image.shape}")
            
            # Step 1: Async emirate detection
            emirate, emirate_confidence = await self.detect_emirate_async(cv_image)
            
            logger.info(f"Detected emirate: {emirate} (confidence: {emirate_confidence:.2f})")
            
            # Use lower confidence threshold since we fixed async handling
            confidence_threshold = 0.7
            
            # Step 2: Route based on emirate complexity and confidence
            if not self.ollama_available:
                # No Ollama available - use traditional OCR for everything
                logger.info("Ollama not available, using traditional OCR fallback")
                result = await self.extract_with_traditional_ocr_async(cv_image, emirate if emirate != "unknown" else None)
                
            elif emirate in ["Abu Dhabi", "Sharjah"] and emirate_confidence > confidence_threshold:
                # Use Ollama for disambiguation - high confidence cases
                logger.info(f"Using Ollama Qwen2.5-VL for {emirate} (disambiguation required, high confidence)")
                result = await self.call_ollama_qwen_async(base64_data)
                
            elif emirate in ["Dubai", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"] and emirate_confidence > confidence_threshold:
                # Use traditional OCR for deterministic formats
                result = await self.extract_with_traditional_ocr_async(cv_image, emirate)
                
                logger.debug(f"*** DEBUG: Traditional OCR returned: {result}")
                
                if result.get("error"):
                    logger.debug(f"*** DEBUG: Traditional OCR FAILED with error: {result.get('error')}")
                    logger.info("*** DEBUG: Falling back to Ollama")
                    llm_result = await self.call_ollama_qwen_async(base64_data)
                    if not llm_result.get("error"):
                        logger.info("*** DEBUG: Using Ollama result instead")
                        result = llm_result
                    else:
                        logger.debug("*** DEBUG: Ollama also failed, keeping traditional OCR error")
                else:
                    logger.info("*** DEBUG: Traditional OCR succeeded, using its result")
                
            else:
                # Low confidence or unknown emirate - let Ollama do fresh analysis
                logger.info(f"Using Ollama Qwen2.5-VL for fresh analysis (emirate: {emirate}, confidence: {emirate_confidence:.2f})")
                result = await self.call_ollama_qwen_async(base64_data)
                
            # Add detection metadata
            if not result.get("error"):
                result["emirate_detected"] = emirate
                result["emirate_confidence"] = emirate_confidence
                
                # Add vehicle type if not already present
                if "vehicle_type" not in result and "plate_color" in result:
                    result["vehicle_type"] = self.get_vehicle_type(result["plate_color"])
                
                # Add color confidence if not present
                if "color_confidence" not in result:
                    result["color_confidence"] = 0.8
            
            return result
            
        except Exception as e:
            logger.error(f"Async base64 analysis failed: {e}")
            return {"error": str(e), "traceback": traceback.format_exc()}

    async def extract_with_traditional_ocr_async(self, image: np.ndarray, emirate: str = None) -> Dict:
        """Call external OCR service"""
        try:
            lang = "ar" if emirate in ['Fujairah', 'Umm Al Quwain'] else "en"
            
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode()
            
            # Call external OCR service
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8081/ocr",
                    json={"image_base64": img_base64, "lang": lang},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        return {"error": f"OCR service error: {resp.status}"}
                    
                    ocr_data = await resp.json()
                    ocr_results = ocr_data['result'][0] if ocr_data['result'] else []  # Unwrap the outer list
            
           
            if ocr_results and len(ocr_results) > 0:
                # OCR service returns: [[bbox, (text, score)], ...]
                # Already in the correct format, just log it
                logger.debug(f"Total async raw detections: {len(ocr_results)}")
            else:
                return {"error": "No text detected by async OCR"}
            
            logger.debug(f"Total async raw detections: {len(ocr_results)}")
            
            # Filter detections
            valid_detections = self.filter_arabic_and_noise(ocr_results)
            logger.debug(f"Valid detections after filtering: {len(valid_detections)}")
            
            if not valid_detections:
                return {
                    "error": "No valid text found after filtering",
                    "raw_detections_count": len(ocr_results),
                    "emirate": emirate or "Unknown"
                }
            
            # Process each detection and split combined text if needed
            processed_detections = []
            for text, conf, bbox in valid_detections:
                # Only split if we know the emirate format
                if emirate:
                    split_texts = self.split_combined_text(text, emirate)
                    for split_text in split_texts:
                        processed_detections.append({
                            'text': split_text,
                            'confidence': conf,
                            'bbox': bbox
                        })
                        logger.debug(f"Processed detection: '{split_text}' (conf: {conf:.2f})")
                else:
                    # Don't split if emirate is unknown
                    processed_detections.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': bbox
                    })
                    logger.debug(f"Processed detection (no split): '{text}' (conf: {conf:.2f})")
            
            # Sort by position (left to right)
            processed_detections.sort(key=lambda x: x['bbox'][0][0])
            
            # Extract components
            category = None
            state = emirate or "Unknown"
            number = None
            
            logger.debug(f"Extracting components for emirate: {emirate or 'Unknown'}")
            
            for detection in processed_detections:
                text = detection['text']
                logger.debug(f"Processing text: '{text}'")
                
                # Skip emirate name text
                emirate_keywords = ['dubai', 'ajman', 'sharjah', 'abu dhabi', 'rak', 'uae.rak', 'الفجيرة', 'أم القيوين', 'ras al khaimah', 'fujairah', 'umm al quwain']
                if any(keyword in text.lower() for keyword in emirate_keywords):
                    logger.debug(f"Skipped '{text}' - emirate name")
                    continue
                
                # Extract category based on known emirate format or try both
                if not category:
                    if emirate in ['Dubai', 'Ajman', 'Ras Al Khaimah', 'Fujairah', 'Umm Al Quwain']:
                        # Letter-based emirates
                        if len(text) == 1 and text.isalpha():
                            if emirate in self.valid_categories and text.upper() in self.valid_categories[emirate]:
                                category = text.upper()
                                logger.debug(f"Found valid category: '{category}' for {emirate}")
                            else:
                                logger.debug(f"Invalid category '{text.upper()}' for {emirate}")
                    elif emirate in ['Abu Dhabi', 'Sharjah']:
                        # Number-based emirates
                        if text.isdigit() and len(text) <= 2:
                            if emirate in self.valid_categories and text in self.valid_categories[emirate]:
                                category = text
                                logger.debug(f"Found valid category: '{category}' for {emirate}")
                            else:
                                logger.debug(f"Invalid category '{text}' for {emirate}")
                    else:
                        # Unknown emirate - try to extract any reasonable category
                        if len(text) == 1 and text.isalpha():
                            category = text.upper()
                            logger.debug(f"Found letter category: '{category}' (emirate unknown)")
                        elif text.isdigit() and len(text) <= 2:
                            category = text
                            logger.debug(f"Found number category: '{category}' (emirate unknown)")
                
                # Extract plate number (1-5 digits)
                if text.isdigit() and 1 <= len(text) <= 5 and number is None:
                    # Skip if this digit was already used as category
                    if text != category:
                        number = text
                        logger.debug(f"Found plate number: '{number}'")
                    else:
                        logger.debug(f"Skipped '{text}' - already used as category")
                elif text.isdigit() and len(text) > 5:
                    logger.debug(f"Number '{text}' too long ({len(text)} digits) - might be combined")
            
            logger.debug(f"Final async extraction - Category: {category}, State: {state}, Number: {number}")
            
            # Validate extraction
            if category and number:
                plate_color, color_confidence = self.detect_plate_color(image)
                vehicle_type = self.get_vehicle_type(plate_color)
                # full_text = self.generate_full_text(category, state, number)
                confidence = sum(d['confidence'] for d in processed_detections) / len(processed_detections)

                state_code = self.map_emirate_to_code(state)
                full_text = self.generate_full_text(category, state_code, number)

                return {
                    "category": category,
                    "state": state_code,
                    "number": number,
                    "full_text": full_text,
                    "confidence": confidence,
                    "method": "traditional_ocr_async",
                    "plate_color": plate_color,
                    "color_confidence": color_confidence,
                    "vehicle_type": vehicle_type
                }
            else:
                # Detailed error reporting
                all_texts = [d['text'] for d in processed_detections]
                return {
                    "error": f"Incomplete async extraction for {emirate or 'Unknown emirate'}",
                    "details": {
                        "category_found": category,
                        "number_found": number,
                        "all_detected_texts": all_texts,
                        "valid_categories_for_emirate": self.valid_categories.get(emirate, "N/A") if emirate else "Unknown emirate",
                        "emirate": emirate or "Unknown"
                    }
                }
        except Exception as e:
            logger.error(f"Async traditional OCR extraction failed: {e}")
            return {"error": str(e), "traceback": traceback.format_exc(), "emirate": emirate or "Unknown"}
        
    def filter_arabic_and_noise(self, ocr_results: List) -> List:
        """Filter out Arabic text, low-confidence results, and noise"""
        ARABIC_WORDS_TO_FILTER = {
            'abu', 'dhabi', 'أبوظبي', 'أبوظبي', 'budhabi', 'أبو', 'ظبي', 'aba'
        }
        VALID_PLATE_CHARS = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        
        valid_detections = []
        
        for line in ocr_results:
            if len(line) < 2:
                continue
                
            bbox = line[0]
            text_info = line[1]
            
            # Extract text and confidence properly
            if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                text = str(text_info[0])  # Get just the text part
                confidence = float(text_info[1])  # Get just the confidence part
            else:
                text = str(text_info)
                confidence = 0.8

            text_clean = text.strip().lower()
            
            # Skip if confidence too low
            if confidence < 0.4:
                continue
                
            # Skip filtered Arabic words
            if any(arabic_word in text_clean for arabic_word in ARABIC_WORDS_TO_FILTER):
                continue
                
            # Skip if mostly non-alphanumeric
            alphanumeric_chars = sum(1 for c in text if c.isalnum())
            if len(text) > 0 and alphanumeric_chars / len(text) < 0.5:
                continue
                
            # Skip very short detections
            if len(text.strip()) < 1:
                continue
                
            # Append only the text string, not the whole tuple
            valid_detections.append((text.strip(), confidence, bbox))
        
        return valid_detections

    def split_combined_text(self, text: str, emirate: str) -> List[str]:
        """Split combined text like 'B12345' into ['B', '12345']"""
        if emirate in ['Dubai', 'Ajman', 'Ras Al Khaimah', 'Fujairah', 'Umm Al Quwain']:
            # For letter-based emirates, look for pattern: Letter + Numbers
            match = re.match(r'^([A-Z])(\d+)$', text.upper())
            if match:
                letter, numbers = match.groups()
                logger.debug(f"Split '{text}' into category '{letter}' and number '{numbers}'")
                return [letter, numbers]
        return [text]
    
    def map_emirate_to_code(self, emirate_name: str) -> str:
        """
        Convert full emirate name to 3-letter code.
        Returns 'Unknown' if emirate is not recognized.
        
        Args:
            emirate_name: Full name of emirate (e.g., 'Abu Dhabi', 'Dubai')
        
        Returns:
            3-letter code (e.g., 'AUH', 'DXB') or 'Unknown'
        """
        if not emirate_name or emirate_name == "Unknown" or emirate_name == "unknown":
            return "Unknown"
        
        # Get the code from mapping, default to 'Unknown' if not found
        return self.emirate_to_code.get(emirate_name, "Unknown")

# Initialize the FastAPI analyzer
analyzer = FastAPIUAEPlateAnalyzer(
    # model_path=os.getenv("MODEL_PATH", "/models"),
    # model_path="C:/Users/User/.paddlex",
    ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
)

@app.post("/analyze", response_model=PlateAnalysisResponse)
async def analyze_plate(request: PlateAnalysisRequest):
    """Main OCR analysis endpoint with async processing"""
    try:
        # Extract base64 image data from multiple possible fields
        base64_data = None
        possible_fields = ['base64_data', 'plate_cropped_image', 'plateimage', 'image']
        
        for field in possible_fields:
            value = getattr(request, field, None)
            if value:
                base64_data = value
                logger.debug(f"Found base64 data in field: {field}")
                break
        
        if not base64_data:
            raise HTTPException(
                status_code=400, 
                detail="No base64 image data found in request"
            )
        
        logger.info(f"Processing async request")
        
        # Analyze the plate with async implementation
        result = await analyzer.analyze_plate_from_base64_async(base64_data)
        
        # Return result - handle errors properly
        if result.get("error"):
            # Return error as JSON, not HTTPException with dict
            raise HTTPException(
                status_code=500, 
                detail=result.get("error")  # ← Changed: Just the string, not the dict
            )
        else:
            return PlateAnalysisResponse(**result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint failed: {e}")
        logger.error(traceback.format_exc())  # ← Log the traceback
        raise HTTPException(
            status_code=500, 
            detail=str(e)  # ← Changed: Just the string
        )

@app.get("/models/info")
async def model_info():
    """Get information about loaded models"""
    # Check OCR service availability
    ocr_service_healthy = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{analyzer.ocr_service_url}/health", timeout=5) as resp:
                ocr_service_healthy = resp.status == 200
    except:
        pass
    
    return {
        "ocr_service": {
            "url": analyzer.ocr_service_url,
            "available": ocr_service_healthy
        },
        "ollama": {
            "available": analyzer.ollama_available,
            "url": analyzer.ollama_url,
            "model": analyzer.ollama_model
        },
        "supported_emirates": list(analyzer.valid_categories.keys()),
        "supported_colors": list(analyzer.vehicle_type_mapping.keys())
    }

@app.get("/ollama/status")
async def ollama_status():
    """Check Ollama status and available models"""
    try:
        if analyzer.ollama_available:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{analyzer.ollama_url}/api/tags", 
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        models = await response.json()
                        return {
                            "status": "available",
                            "models": models.get('models', []),
                            "required_model": analyzer.ollama_model,
                            "model_available": any(model['name'] == analyzer.ollama_model for model in models.get('models', []))
                        }
                    else:
                        return {"status": "unreachable", "error": f"HTTP {response.status}"}
        else:
            return {"status": "unavailable", "error": "Ollama connection failed during startup"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     return HealthResponse(
#         status="healthy",
#         ocr_models_loaded={
#             "english": analyzer.ocr_model_en is not None,
#             "arabic": analyzer.ocr_model_ar is not None
#         },
#         ollama_available=analyzer.ollama_available,
#         ollama_model=analyzer.ollama_model,
#         timestamp=time.time()
#     )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check OCR service health
    ocr_service_healthy = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{analyzer.ocr_service_url}/health", timeout=5) as resp:
                ocr_service_healthy = resp.status == 200
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        ocr_models_loaded={
            "english": ocr_service_healthy,
            "arabic": ocr_service_healthy
        },
        ollama_available=analyzer.ollama_available,
        ollama_model=analyzer.ollama_model,
        timestamp=time.time()
    )

@app.get("/queue/stats")
async def queue_statistics():
    """Get queue statistics"""
    async with analyzer.stats_lock:
        avg_wait = (sum(analyzer.queue_stats['wait_times']) / 
                   len(analyzer.queue_stats['wait_times']) 
                   if analyzer.queue_stats['wait_times'] else 0.0)
        
        return {
            'queue_size': analyzer.request_queue.qsize(),
            'max_queue_size': analyzer.request_queue.maxsize,
            'total_processed': analyzer.queue_stats['total_processed'],
            'total_rejected': analyzer.queue_stats['total_rejected'],
            'average_wait_time': round(avg_wait, 2),
            'current_processing': analyzer.queue_stats['current_processing'],
            'ollama_semaphore_available': analyzer.ollama_semaphore._value
        }

if __name__ == '__main__':
    # Configuration for FastAPI with uvicorn
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    workers = int(os.getenv('WORKERS', 1))
    
    logger.info(f"Starting FastAPI OCR Server on {host}:{port}")
    # logger.info(f"PaddleOCR model path: {analyzer.model_path}")
    logger.info(f"Ollama URL: {analyzer.ollama_url}")
    logger.info(f"Ollama model: {analyzer.ollama_model}")
    
    # Run with uvicorn for production
    uvicorn.run(
        "ocr_fastapi_server:app",
        host=host,
        port=port,
        workers=1,  # CRITICAL: MUST BE 1
        timeout_keep_alive=300,
        limit_concurrency=10,  # Limit total concurrent connections
        access_log=True
    )