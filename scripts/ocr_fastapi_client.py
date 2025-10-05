"""
FastAPI OCR Client for NiFi ExecuteStreamCommand
Updated client for FastAPI-based OCR service
"""

import sys
import json
import requests
import logging
import time
import os
from typing import Dict, Optional
import asyncio
import aiohttp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastAPIOCRClient:
    def __init__(self, api_base_url: str = None, timeout: int = 300, max_retries: int = 3):
        """Initialize FastAPI OCR client"""
        self.api_base_url = api_base_url or os.getenv('OCR_API_URL', 'http://localhost:8080')
        self.timeout = timeout
        self.max_retries = max_retries
        self.analyze_endpoint = f"{self.api_base_url}/analyze"
        self.health_endpoint = f"{self.api_base_url}/health"
        
        # Headers for FastAPI
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            # 'Authorization': f'Bearer {os.getenv("OCR_API_TOKEN", "")}',  # Uncomment when auth is needed
        }
        
        logger.info(f"FastAPI OCR Client initialized - Base URL: {self.api_base_url}")

    def check_health(self) -> bool:
        """Check if FastAPI OCR service is healthy"""
        try:
            response = requests.get(self.health_endpoint, headers=self.headers, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"FastAPI OCR service is healthy: {health_data}")
                return True
            else:
                logger.warning(f"FastAPI OCR service health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def analyze_plate(self, data: Dict) -> Dict:
        """Send plate data to FastAPI OCR API for analysis"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending FastAPI OCR request (attempt {attempt + 1}/{self.max_retries})")
                
                # Add timestamp for tracking
                data['request_timestamp'] = time.time()
                
                # FastAPI expects data in the request body with proper field names
                request_data = {}
                
                # Map various field names to FastAPI expected fields
                if 'base64_data' in data:
                    request_data['base64_data'] = data['base64_data']
                elif 'plate_cropped_image' in data:
                    request_data['plate_cropped_image'] = data['plate_cropped_image']
                elif 'plateimage' in data:
                    request_data['plateimage'] = data['plateimage']
                elif 'image' in data:
                    request_data['image'] = data['image']
                else:
                    # Try to find any field with base64 data
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 100 and '/' in value:  # Likely base64
                            request_data['base64_data'] = value
                            break
                
                if not request_data:
                    return {"error": f"No base64 image data found in input. Available fields: {list(data.keys())}"}
                
                response = requests.post(
                    self.analyze_endpoint,
                    json=request_data,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                logger.info(f"FastAPI OCR response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("FastAPI OCR analysis successful")
                    return result
                elif response.status_code == 422:
                    # Validation error - don't retry
                    error_data = response.json() if response.content else {"error": "Validation error"}
                    logger.error(f"FastAPI validation error: {error_data}")
                    return {"error": f"FastAPI validation error", "details": error_data}
                elif response.status_code == 500:
                    # Server error - might be worth retrying
                    error_data = response.json() if response.content else {"error": "Server error"}
                    logger.error(f"FastAPI server error: {error_data}")
                    if attempt == self.max_retries - 1:
                        return {"error": f"FastAPI server error after {self.max_retries} attempts", "details": error_data}
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    # Other client error - don't retry
                    error_msg = response.text if response.content else "Unknown error"
                    logger.error(f"FastAPI client error {response.status_code}: {error_msg}")
                    return {"error": f"FastAPI error {response.status_code}: {error_msg}"}
                    
            except requests.exceptions.Timeout:
                logger.warning(f"FastAPI timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    return {"error": f"FastAPI timeout after {self.max_retries} attempts"}
                time.sleep(5)  # Wait before retry
                continue
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error to FastAPI on attempt {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    return {"error": f"Cannot connect to FastAPI after {self.max_retries} attempts"}
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Unexpected error in FastAPI call: {e}")
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}

    async def analyze_plate_async(self, data: Dict) -> Dict:
        """Async version of analyze_plate for better performance"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending async FastAPI OCR request (attempt {attempt + 1}/{self.max_retries})")
                
                # Add timestamp for tracking
                data['request_timestamp'] = time.time()
                
                # Prepare request data
                request_data = {}
                
                if 'base64_data' in data:
                    request_data['base64_data'] = data['base64_data']
                elif 'plate_cropped_image' in data:
                    request_data['plate_cropped_image'] = data['plate_cropped_image']
                elif 'plateimage' in data:
                    request_data['plateimage'] = data['plateimage']
                elif 'image' in data:
                    request_data['image'] = data['image']
                else:
                    # Try to find any field with base64 data
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 100 and '/' in value:  # Likely base64
                            request_data['base64_data'] = value
                            break
                
                if not request_data:
                    return {"error": f"No base64 image data found in input. Available fields: {list(data.keys())}"}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.analyze_endpoint,
                        json=request_data,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        logger.info(f"Async FastAPI OCR response status: {response.status}")
                        
                        if response.status == 200:
                            result = await response.json()
                            logger.info("Async FastAPI OCR analysis successful")
                            return result
                        elif response.status == 422:
                            # Validation error - don't retry
                            error_data = await response.json() if response.content_length else {"error": "Validation error"}
                            logger.error(f"FastAPI validation error: {error_data}")
                            return {"error": f"FastAPI validation error", "details": error_data}
                        elif response.status == 500:
                            # Server error - might be worth retrying
                            error_data = await response.json() if response.content_length else {"error": "Server error"}
                            logger.error(f"FastAPI server error: {error_data}")
                            if attempt == self.max_retries - 1:
                                return {"error": f"FastAPI server error after {self.max_retries} attempts", "details": error_data}
                            await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            # Other error
                            error_msg = await response.text() if response.content_length else "Unknown error"
                            logger.error(f"FastAPI client error {response.status}: {error_msg}")
                            return {"error": f"FastAPI error {response.status}: {error_msg}"}
                    
            except asyncio.TimeoutError:
                logger.warning(f"Async FastAPI timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    return {"error": f"Async FastAPI timeout after {self.max_retries} attempts"}
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Unexpected error in async FastAPI call: {e}")
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}

def process_stdin():
    """Process data from stdin for NiFi integration"""
    try:
        # Read input data from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            result = {"error": "No input data provided"}
        else:
            # Initialize FastAPI client
            client = FastAPIOCRClient()
            
            # Check if service is available (optional - remove if causing delays)
            if not client.check_health():
                logger.warning("FastAPI OCR service health check failed, proceeding anyway...")
            
            # Parse input data
            try:
                json_input = json.loads(input_data)
                logger.info(f"Parsed JSON input, keys: {list(json_input.keys())}")
                
                # Send to FastAPI OCR API
                result = client.analyze_plate(json_input)
                        
            except json.JSONDecodeError:
                logger.info("Input is not JSON, treating as raw base64")
                # Treat as raw base64 data
                result = client.analyze_plate({"base64_data": input_data})
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {"error": str(e)}
        print(json.dumps(error_result))
        logger.error(f"Client script failed: {e}")

async def process_stdin_async():
    """Async version of process_stdin for better performance"""
    try:
        # Read input data from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            result = {"error": "No input data provided"}
        else:
            # Initialize FastAPI client
            client = FastAPIOCRClient()
            
            # Parse input data
            try:
                json_input = json.loads(input_data)
                logger.info(f"Parsed JSON input, keys: {list(json_input.keys())}")
                
                # Send to FastAPI OCR API asynchronously
                result = await client.analyze_plate_async(json_input)
                        
            except json.JSONDecodeError:
                logger.info("Input is not JSON, treating as raw base64")
                # Treat as raw base64 data
                result = await client.analyze_plate_async({"base64_data": input_data})
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {"error": str(e)}
        print(json.dumps(error_result))
        logger.error(f"Async client script failed: {e}")

def main():
    """Main function for testing"""
    import argparse
    import csv
    
    parser = argparse.ArgumentParser(description='FastAPI OCR Client')
    parser.add_argument('--stdin', action='store_true', help='Process data from stdin (for NiFi)')
    parser.add_argument('--async-stdin', action='store_true', help='Process data from stdin asynchronously')
    parser.add_argument('--csv', help='Process base64 data from CSV file')
    parser.add_argument('--row', type=int, default=0, help='CSV row to process (default: 0)')
    parser.add_argument('--url', help='FastAPI OCR base URL')
    parser.add_argument('--test', action='store_true', help='Run health check test')
    parser.add_argument('--use-async', action='store_true', help='Use async client for CSV processing')
    
    args = parser.parse_args()
    
    if args.stdin:
        process_stdin()
    elif args.async_stdin:
        asyncio.run(process_stdin_async())
    elif args.csv:
        # Process CSV file input
        try:
            with open(args.csv, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                rows = list(csv_reader)
                
                if args.row >= len(rows):
                    print(f"Error: Row {args.row} not found. CSV has {len(rows)} rows (0-{len(rows)-1})", file=sys.stderr)
                    sys.exit(1)
                
                row = rows[args.row]
                print(f"Processing row {args.row} from {args.csv}", file=sys.stderr)
                print(f"Camera: {row.get('cameraName', 'N/A')}", file=sys.stderr)
                print(f"Timestamp: {row.get('timestamp', 'N/A')}", file=sys.stderr)
                print(f"Original plate read: {row.get('plateRead', 'N/A')}", file=sys.stderr)
                print("-" * 50, file=sys.stderr)
                
                # Extract base64 image data
                base64_data = row.get('plate_cropped_image')
                if not base64_data:
                    print("Error: No 'plate_cropped_image' field found in CSV row", file=sys.stderr)
                    print(f"Available fields: {list(row.keys())}", file=sys.stderr)
                    sys.exit(1)
            
                client = FastAPIOCRClient(api_base_url=args.url)
                
                async def process_csv_async():
                    # Check if service is available
                    if not client.check_health():
                        logger.warning("FastAPI OCR service health check failed, proceeding anyway...")
                    
                    # Send to FastAPI OCR API
                    if args.use_async:
                        result = await client.analyze_plate_async({"base64_data": base64_data})
                    else:
                        result = client.analyze_plate({"base64_data": base64_data})
                    
                    print("FastAPI OCR Results:", file=sys.stderr)
                    print(json.dumps(result, indent=2), file=sys.stderr)
                    
                    # Compare with original if available
                    if not result.get('error'):
                        print("\nComparison with original CSV data:", file=sys.stderr)
                        print(f"Original plate read: {row.get('plateRead', 'N/A')}", file=sys.stderr)
                        print(f"OCR detected: {result.get('number', 'N/A')}", file=sys.stderr)
                        print(f"Original region: {row.get('plateRegion', 'N/A')}", file=sys.stderr)
                        print(f"OCR detected: {result.get('state', 'N/A')}", file=sys.stderr)
                
                if args.use_async:
                    asyncio.run(process_csv_async())
                else:
                    asyncio.run(process_csv_async())  # Still need asyncio for health check compatibility
                
        except FileNotFoundError:
            print(f"Error: File '{args.csv}' not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error processing CSV: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.test:
        # Test mode - check service health
        client = FastAPIOCRClient(api_base_url=args.url)
        if client.check_health():
            print("FastAPI OCR service is healthy", file=sys.stderr)
            sys.exit(0)
        else:
            print("FastAPI OCR service is not healthy", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage:", file=sys.stderr)
        print("  python fastapi_ocr_client.py --stdin                         (for NiFi)", file=sys.stderr)
        print("  python fastapi_ocr_client.py --async-stdin                   (async for NiFi)", file=sys.stderr)
        print("  python fastapi_ocr_client.py --csv filename [--row N] [--async]  (process CSV)", file=sys.stderr)
        print("  python fastapi_ocr_client.py --test                          (health check)", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()