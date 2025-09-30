"""
PaddleOCR Model Download Script
Downloads separate English and Arabic PaddleOCR models to local cache for offline use
"""

import os
from paddleocr import PaddleOCR

def download_paddleocr_models():
    """Download PaddleOCR models for English and Arabic languages separately"""
    
    print("PaddleOCR Model Downloader")
    print("=" * 40)
    print("Downloading models to: C:\\Users\\User\\.paddlex\\official_models\\")
    print("Note: English and Arabic models will be downloaded separately")
    print()
    
    models_to_download = [
        {"lang": "en", "name": "English"},
        {"lang": "ar", "name": "Arabic"}
    ]
    
    for model_config in models_to_download:
        lang = model_config["lang"]
        name = model_config["name"]
        
        print(f"Downloading {name} models...")
        
        try:
            # Initialize PaddleOCR - this triggers model download
            ocr = PaddleOCR(
                use_textline_orientation=True,
                lang=lang,
            )
            print(f"✓ {name} models downloaded")
            
        except Exception as e:
            print(f"✗ Failed to download {name} models: {e}")
    
    print()
    print("Download complete!")
    print("Both English and Arabic models are cached locally")
    print("The main OCR script will use both models strategically:")
    print("- English model for most emirate detection")
    print("- Arabic model for Fujairah and Umm Al Quwain")

if __name__ == "__main__":
    print("Script started...")
    download_paddleocr_models()
    print("Script finished.")