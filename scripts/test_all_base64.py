#!/usr/bin/env python3
"""
Test all base64 files in base64_test_files folder
Sends each to the OCR API and reports results
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from datetime import datetime

def test_file(filepath: Path, api_url: str = "http://localhost:8080/analyze"):
    """Test a single base64 file"""
    try:
        # Read base64 data
        with open(filepath, 'r') as f:
            base64_data = f.read().strip()
        
        # Send request
        start_time = time.time()
        response = requests.post(
            api_url,
            json={'base64_data': base64_data},
            timeout=300
        )
        elapsed = time.time() - start_time
        
        # Parse result
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'file': filepath.name,
                'elapsed': elapsed,
                'category': result.get('category'),
                'state': result.get('state'),
                'number': result.get('number'),
                'full_text': result.get('full_text'),
                'method': result.get('method'),
                'error': result.get('error')
            }
        else:
            return {
                'success': False,
                'file': filepath.name,
                'elapsed': elapsed,
                'error': f"HTTP {response.status_code}: {response.text[:100]}"
            }
            
    except Exception as e:
        return {
            'success': False,
            'file': filepath.name,
            'error': str(e)
        }

def main():
    # Configuration
    base64_folder = Path("../base64_test_files")
    api_url = "http://localhost:8080/analyze"
    
    # Find all .txt files
    files = sorted(base64_folder.glob("*.txt"))
    
    if not files:
        print(f"No .txt files found in {base64_folder}")
        return
    
    print(f"Found {len(files)} files to test")
    print("=" * 80)
    
    # Test each file
    results = []
    successful = 0
    failed = 0
    
    for i, filepath in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Testing: {filepath.name}")
        
        result = test_file(filepath, api_url)
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"  ✓ SUCCESS ({result['elapsed']:.2f}s)")
            print(f"    Plate: {result.get('full_text', 'N/A')}")
            print(f"    Method: {result.get('method', 'N/A')}")
            if result.get('error'):
                print(f"    Warning: {result['error']}")
        else:
            failed += 1
            print(f"  ✗ FAILED")
            print(f"    Error: {result['error']}")
        
        # Small delay between requests
        if i < len(files):
            time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(files)*100):.1f}%")
    
    # Detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['file']:40} | {result.get('full_text', result.get('error', 'ERROR'))}")

if __name__ == "__main__":
    main()