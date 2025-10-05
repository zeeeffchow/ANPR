#!/usr/bin/env python3
"""
Start Ollama with Model Warmup
This script starts Ollama server and pre-loads the model to eliminate cold-start latency
"""

import os
import sys
import time
import subprocess
import requests
import argparse
import threading
from typing import Optional
from datetime import datetime

def log_timestamp(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def monitor_ollama_logs(process: subprocess.Popen):
    """Monitor Ollama output in a separate thread"""
    def read_output(pipe, prefix):
        try:
            for line in iter(pipe.readline, ''):
                if line.strip():
                    # Filter for interesting log lines
                    if any(keyword in line for keyword in ['POST', 'GET', 'generate', 'chat', 'embeddings', 'error', 'ERROR', 'WARN']):
                        log_timestamp(f"{prefix} {line.strip()}")
        except:
            pass
    
    # Only start monitoring if we have pipes
    if hasattr(process, 'stdout') and process.stdout:
        threading.Thread(target=read_output, args=(process.stdout, "📤"), daemon=True).start()
    if hasattr(process, 'stderr') and process.stderr:
        threading.Thread(target=read_output, args=(process.stderr, "⚠️ "), daemon=True).start()

def start_ollama_server() -> subprocess.Popen:
    """Start Ollama server in the background"""
    log_timestamp("🚀 Starting Ollama server with optimized settings...")
    
    # Set environment variable to keep models loaded indefinitely
    os.environ['OLLAMA_KEEP_ALIVE'] = '-1'
    
    # Start Ollama serve with pipes to capture output
    process = subprocess.Popen(
        ['ollama', 'serve'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    log_timestamp(f"✓ Ollama server started (PID: {process.pid})")
    
    # Start monitoring logs
    monitor_ollama_logs(process)
    
    return process

def wait_for_server(max_attempts: int = 30) -> bool:
    """Wait for Ollama server to be ready"""
    log_timestamp("⏳ Waiting for server to be ready...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://127.0.0.1:11434/api/tags', timeout=1)
            if response.status_code == 200:
                log_timestamp("✓ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            print('.', end='', flush=True)
            time.sleep(1)
    
    print("\n")
    log_timestamp("✗ Failed to start Ollama server")
    return False

def warmup_model(model_name: str, keep_alive: str = '-1') -> bool:
    """Pre-load the model with a warmup request"""
    log_timestamp(f"🔥 Warming up model: {model_name}")
    
    # Convert keep_alive to proper format for API
    # Use 0 for indefinite (relies on OLLAMA_KEEP_ALIVE env var)
    api_keep_alive = 0 if keep_alive == '-1' else keep_alive
    
    try:
        response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json={
                'model': model_name,
                'prompt': 'Hello',
                'stream': False,
                'keep_alive': api_keep_alive
            },
            timeout=120  # Give it time to load
        )
        
        if response.status_code == 200:
            log_timestamp("✓ Model loaded and ready!")
            return True
        else:
            log_timestamp(f"✗ Failed to warm up model: HTTP {response.status_code}")
            try:
                error_detail = response.json()
                log_timestamp(f"   Error details: {error_detail}")
            except:
                log_timestamp(f"   Response text: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        log_timestamp(f"✗ Failed to warm up model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Start Ollama server with model pre-loading'
    )
    parser.add_argument(
        '--model',
        default='qwen2.5vl:7b',
        help='Model name to pre-load (default: qwen2.5vl:7b)'
    )
    parser.add_argument(
        '--keep-alive',
        default='-1',
        help='Keep-alive duration: -1 (indefinite), 24h, 12h, etc. (default: -1)'
    )
    
    args = parser.parse_args()
    
    # Start Ollama server
    ollama_process = start_ollama_server()
    
    try:
        # Wait for server to be ready
        if not wait_for_server():
            ollama_process.terminate()
            sys.exit(1)
        
        # Warm up the model
        if not warmup_model(args.model, args.keep_alive):
            ollama_process.terminate()
            sys.exit(1)
        
        log_timestamp(f"✨ Ollama is running with '{args.model}' kept warm in memory")
        log_timestamp(f"📌 Keep-alive setting: {args.keep_alive}")
        log_timestamp("👀 Monitoring requests (you'll see activity when FastAPI calls Ollama)...")
        print("\n⌨️  Press Ctrl+C to stop Ollama server\n")
        
        # Keep the process running and wait for it
        while True:
            if ollama_process.poll() is not None:
                log_timestamp("⚠️  Ollama server has stopped unexpectedly")
                break
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n")
        log_timestamp("🛑 Stopping Ollama server...")
        ollama_process.terminate()
        ollama_process.wait()
        log_timestamp("✓ Ollama server stopped")
        sys.exit(0)
    except Exception as e:
        log_timestamp(f"✗ Error: {e}")
        ollama_process.terminate()
        sys.exit(1)

if __name__ == '__main__':
    main()