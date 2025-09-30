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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

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
    def __init__(self, model_path: str = "/models", ollama_url: str = "http://localhost:11434"):
        """Initialize with async-friendly PaddleOCR models and Ollama for Qwen2.5-VL"""
        self.model_path = model_path
        self.ollama_url = ollama_url
        self.ollama_model = "qwen2.5vl:7b"
        self.ocr_model_en = None
        self.ocr_model_ar = None
        
        # Initialize models in a thread-safe way
        self._load_ocr_models()
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
        self.qwen_system_prompt = """
You are an expert in UAE license plate recognition for ALL seven emirates. Analyze this license plate image and extract information WITHOUT any pre-assumptions about which emirate it might be.

UAE Emirates and their formats:
- Abu Dhabi: Number categories (1, 4-21, 50) + "Abu Dhabi"/"A.D" + plate number
- Sharjah: Number categories (1-4) + "SHARJAH" + plate number  
- Dubai: Letter categories (A-Z) + "DUBAI" + plate number
- Ajman: Letter categories + "AJMAN" + plate number
- Ras Al Khaimah: Letter categories + "UAE.RAK"/"RAS AL KHAIMAH" + plate number
- Fujairah: Letter categories + Arabic text + plate number
- Umm Al Quwain: Letter categories + Arabic text + plate number

For Abu Dhabi and Sharjah: Apply disambiguation if you see two numbers - check if left number is in valid categories. If yes, then left number is the category.
For others: Letter category + state name + number (no ambiguity).

For Fujairah and Umm Al Quwain: The Emirate is stated in Arabic. Look for Arabic text and use that to determine the Emirate. Do NOT attempt to extract any English letters as the Emirate.

IMPORTANT: Always normalize Emirate names: Arabic text → English text, "A.D" → "Abu Dhabi", "RAK" → "Ras Al Khaimah".

Identify plate background color: white, yellow, red, blue, green, black, orange, gold.

IMPORTANT: Return a valid JSON format using exactly these field names:
- "category": The actual category you see (letter A-Z or number 1-50)
- "state": The actual emirate name you identify  
- "number": The actual plate number you see
- "confidence": Your confidence level (0.0-1.0)
- "plate_color": The actual background color you observe

You MUST not return any Arabic text in the final JSON. Always convert it to English.

CRITICAL: Extract real values from the image, not placeholder examples.
"""
        self._warm_up_models()

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
        """Load PaddleOCR models with optimized settings for FastAPI"""
        try:
            from paddleocr import PaddleOCR
            
            # Load English model with CPU optimization
            self.ocr_model_en = PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                enable_mkldnn=True,  # Safe to use with FastAPI's event loop
                cpu_threads=4  # Optimize for FastAPI async handling
            )
            logger.info("PaddleOCR English model loaded successfully")
            
            # Load Arabic model
            self.ocr_model_ar = PaddleOCR(
                use_textline_orientation=True,
                lang='ar',
                enable_mkldnn=True,
                cpu_threads=4
            )
            logger.info("PaddleOCR Arabic model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load OCR models: {e}")
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
            
        user_prompt = f"""
Analyze this image of a cropped license plate from scratch. Do NOT make any assumptions about which emirate it belongs to.

Pay special attention to:
1. All text visible on the plate (both English and Arabic script)
2. Category indicators (single letters A-Z or numbers 1-50)
3. State/emirate names in both Arabic and English
4. Plate numbers (typically 1-5 digits)
5. Background plate color

Arabic Text Emirate Recognition Guide:
- الفجيرة (al-Fujairah) = Fujairah emirate
- أم القيوين (Umm al-Quwain) = Umm Al Quwain emirate  
- الشارقة (al-Sharjah) = Sharjah emirate
- دبي (Dubai) = Dubai emirate
- عجمان (Ajman) = Ajman emirate
- رأس الخيمة (Ras al-Khaimah) = Ras Al Khaimah emirate
- أبو ظبي (Abu Dhabi) = Abu Dhabi emirate

Extract exactly what you see in this specific image - do not default to "Dubai" or use any placeholder values.

Provide your final JSON in the exact format that we have specified earlier.

Timestamp: {time.time()}
"""
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Ollama Qwen2.5-VL async attempt {attempt + 1}/{max_retries}")
                
                payload = {
                    "model": self.ollama_model,
                    "prompt": self.qwen_system_prompt + "\n\n" + user_prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "top_p": 0.9
                    }
                }
                
                # Use asyncio to make HTTP request non-blocking
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.ollama_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response:
                        
                        logger.info(f"Ollama response status: {response.status}")
                        
                        if response.status == 200:
                            result = await response.json()
                            generated_text = result.get("response", "")
                            
                            logger.info(f"Ollama generated text: {generated_text[:200]}...")
                            
                            # Parse JSON from response
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', generated_text, re.DOTALL)
                            if json_match:
                                try:
                                    parsed_result = json.loads(json_match.group())
                                    
                                    # Normalize response to expected format
                                    normalized_result = {
                                        "category": parsed_result.get("category", ""),
                                        "state": parsed_result.get("state", ""),
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
                                    
                                    return normalized_result
                                    
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON decode error: {e}")
                                    return {"error": f"Invalid JSON in Ollama response: {e}", "raw_response": generated_text}
                            else:
                                logger.error("No JSON found in Ollama response")
                                return {"error": "No JSON found in Ollama response", "raw_response": generated_text}
                        
                        else:
                            error_text = await response.text()
                            logger.error(f"Ollama request failed: {response.status} - {error_text[:500]}")
                            return {"error": f"Ollama request failed: {response.status} - {error_text[:500]}"}
                    
            except asyncio.TimeoutError:
                logger.warning(f"Ollama timeout on attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    return {"error": "Ollama timeout after multiple attempts"}
                await asyncio.sleep(5)  # Non-blocking sleep
                continue
            except Exception as e:
                logger.error(f"Ollama call exception: {e}")
                return {"error": str(e)}
        
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
        return f"{category} {state} {number}"

    def get_vehicle_type(self, plate_color: str) -> str:
        """Get vehicle type based on plate color"""
        return self.vehicle_type_mapping.get(plate_color, 'Unknown Vehicle Type')

    async def detect_emirate_async(self, image: np.ndarray) -> Tuple[str, float]:
        """Async emirate detection using PaddleOCR models"""
        try:
            logger.info(f"Starting async emirate detection")
            
            # Try English model only
            emirate_en, conf_en = "unknown", 0.0
            if self.ocr_model_en is not None:
                try:
                    logger.info(f"Running English OCR predict async...")
                    
                    # Run OCR in thread pool to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    ocr_results_en = await loop.run_in_executor(
                        None, self.ocr_model_en.predict, image
                    )
                    
                    logger.info(f"English OCR predict completed")
                    emirate_en, conf_en = self.parse_emirate_results(ocr_results_en, "en")
                    logger.info(f"English OCR detected: {emirate_en} (conf: {conf_en:.2f})")
                except Exception as e:
                    logger.error(f"English OCR failed: {e}")
            
            # Return English result if found, otherwise unknown
            if emirate_en != "unknown" and conf_en > 0.6:
                logger.info(f"Using English OCR result: {emirate_en}")
                return emirate_en, conf_en
            else:
                logger.info("English OCR failed to detect emirate")
                return "unknown", 0.0
                
        except Exception as e:
            logger.error(f"Async emirate detection failed: {e}")
            return "unknown", 0.0

    def parse_emirate_results(self, ocr_results, lang: str) -> Tuple[str, float]:
        """Parse OCR results to detect emirate"""
        try:
            if not ocr_results or not ocr_results[0]:
                return "unknown", 0.0
                
            # Handle predict() result format
            if isinstance(ocr_results[0], dict) and 'rec_texts' in ocr_results[0]:
                texts = ocr_results[0]['rec_texts']
                scores = ocr_results[0]['rec_scores']
                all_text = [(text.lower().strip(), score) for text, score in zip(texts, scores)]
            else:
                # Handle standard format
                all_text = []
                for line in ocr_results[0]:
                    if len(line) >= 2:
                        text_info = line[1]
                        if isinstance(text_info, tuple):
                            text, confidence = text_info
                        else:
                            text = text_info
                            confidence = 0.8
                        all_text.append((text.lower().strip(), confidence))
            
            # Check for emirate indicators based on language
            emirate_scores = {}
            
            for text, conf in all_text:
                if lang == 'en':
                    # English emirate detection
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
            return "unknown", 0.0

    async def analyze_plate_from_base64_async(self, base64_data: str) -> Dict:
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
        """Async traditional OCR extraction using pre-loaded models"""
        try:
            def run_with_preloaded_models(img, emirate_name):
                """Use pre-loaded OCR models instead of creating fresh instances"""
                # Choose model based on emirate
                if emirate_name in ['Fujairah', 'Umm Al Quwain'] and self.ocr_model_ar:
                    model = self.ocr_model_ar
                    logger.debug(f"Using pre-loaded Arabic OCR model for {emirate_name}")
                elif self.ocr_model_en:
                    model = self.ocr_model_en
                    logger.debug(f"Using pre-loaded English OCR model for {emirate_name or 'Unknown emirate'}")
                else:
                    raise Exception("No OCR model available")
                
                # Run prediction with pre-loaded model
                return model.predict(img)
            
            # Run OCR with pre-loaded models in thread pool
            loop = asyncio.get_event_loop()
            ocr_results = await loop.run_in_executor(None, run_with_preloaded_models, image, emirate)

            # Handle OCR result format - CONVERT NEW FORMAT TO OLD FORMAT
            if ocr_results and len(ocr_results) > 0:
                # Check if we got the new OCRResult object
                ocr_result_obj = ocr_results[0]
                
                # Try to access data - OCRResult objects behave like dictionaries
                try:
                    rec_texts = ocr_result_obj['rec_texts']
                    rec_scores = ocr_result_obj['rec_scores']
                    rec_polys = ocr_result_obj['rec_polys']
                    
                    logger.debug(f"Converting new OCR format - found {len(rec_texts)} text detections")
                    
                    formatted_results = []
                    for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                        if text.strip():  # Skip empty strings
                            formatted_results.append([poly.tolist(), (text, score)])
                    
                    ocr_results = formatted_results
                    
                except (KeyError, TypeError):
                    # Not the new format, check if it's old format
                    if isinstance(ocr_results[0], list):
                        # Handle old format - extract from the list
                        ocr_results = ocr_results[0]
                        logger.debug(f"Using old format - extracted list with {len(ocr_results)} detections")
                    else:
                        logger.error(f"Could not access OCR data from {type(ocr_results[0])}")
                        return {"error": f"Could not access OCR data from {type(ocr_results[0])}"}
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
                full_text = self.generate_full_text(category, state, number)
                confidence = sum(d['confidence'] for d in processed_detections) / len(processed_detections)
                
                return {
                    "category": category,
                    "state": state,
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
            
            if isinstance(text_info, tuple):
                text, confidence = text_info
            else:
                text = text_info
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

# Initialize the FastAPI analyzer
analyzer = FastAPIUAEPlateAnalyzer(
    model_path=os.getenv("MODEL_PATH", "/models"),
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
        
        # Return result
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result)
        else:
            return PlateAnalysisResponse(**result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/models/info")
async def model_info():
    """Get information about loaded models"""
    return {
        "paddleocr_english": {
            "loaded": analyzer.ocr_model_en is not None,
            "model_path": analyzer.model_path
        },
        "paddleocr_arabic": {
            "loaded": analyzer.ocr_model_ar is not None,
            "model_path": analyzer.model_path
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        ocr_models_loaded={
            "english": analyzer.ocr_model_en is not None,
            "arabic": analyzer.ocr_model_ar is not None
        },
        ollama_available=analyzer.ollama_available,
        ollama_model=analyzer.ollama_model,
        timestamp=time.time()
    )

if __name__ == '__main__':
    # Configuration for FastAPI with uvicorn
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    workers = int(os.getenv('WORKERS', 1))
    
    logger.info(f"Starting FastAPI OCR Server on {host}:{port}")
    logger.info(f"PaddleOCR model path: {analyzer.model_path}")
    logger.info(f"Ollama URL: {analyzer.ollama_url}")
    logger.info(f"Ollama model: {analyzer.ollama_model}")
    
    # Run with uvicorn for production
    uvicorn.run(
        "ocr_fastapi_server:app",
        host=host,
        port=port,
        workers=workers,
        # loop="uvloop",  # High-performance event loop
        access_log=True
    )