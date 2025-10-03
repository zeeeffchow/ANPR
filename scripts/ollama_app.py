#!/usr/bin/env python3
"""
Ollama CML Application Wrapper
Keeps Ollama service running persistently in Cloudera AI
"""

import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self):
        self.ollama_process = None
        self.running = True
        
        # Configuration from environment variables
        self.ollama_host = os.getenv('OLLAMA_HOST', '0.0.0.0:11434')
        self.ollama_models_dir = os.getenv('OLLAMA_MODELS', '/home/cdsw/.ollama/models')
        self.model_name = os.getenv('OLLAMA_MODEL', 'qwen2.5-vl:7b')
        
        # Ensure models directory exists
        Path(self.ollama_models_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ollama Configuration:")
        logger.info(f"  Host: {self.ollama_host}")
        logger.info(f"  Models Directory: {self.ollama_models_dir}")
        logger.info(f"  Model: {self.model_name}")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.ollama_process:
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Ollama didn't stop gracefully, killing...")
                self.ollama_process.kill()
        sys.exit(0)
    
    def start_ollama_server(self):
        """Start Ollama server as subprocess"""
        try:
            logger.info("Starting Ollama server...")
            
            # Set environment for Ollama
            env = os.environ.copy()
            env['OLLAMA_HOST'] = self.ollama_host
            env['OLLAMA_MODELS'] = self.ollama_models_dir
            
            # Start Ollama serve
            self.ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"Ollama server started with PID: {self.ollama_process.pid}")
            
            # Give it a moment to start
            time.sleep(5)
            
            # Check if it's still running
            if self.ollama_process.poll() is not None:
                logger.error("Ollama server failed to start!")
                return False
            
            logger.info("Ollama server is running successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    def verify_model_loaded(self):
        """Verify that the required model is available"""
        try:
            logger.info(f"Checking if model '{self.model_name}' is available...")
            
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if self.model_name in result.stdout:
                logger.info(f"✅ Model '{self.model_name}' is available")
                return True
            else:
                logger.warning(f"⚠️ Model '{self.model_name}' not found in Ollama")
                logger.info("Available models:")
                logger.info(result.stdout)
                logger.info("\nTo load the model, you need to have pre-loaded it into:")
                logger.info(f"  {self.ollama_models_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify model: {e}")
            return False
    
    def monitor_ollama(self):
        """Monitor Ollama process and output logs"""
        logger.info("Monitoring Ollama server output...")
        
        try:
            # Stream output from Ollama
            while self.running and self.ollama_process:
                # Check if process is still running
                if self.ollama_process.poll() is not None:
                    logger.error("Ollama process has stopped unexpectedly!")
                    logger.error(f"Exit code: {self.ollama_process.returncode}")
                    return False
                
                # Read and log output
                line = self.ollama_process.stdout.readline()
                if line:
                    logger.info(f"[Ollama] {line.strip()}")
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error monitoring Ollama: {e}")
            return False
        
        return True
    
    def run(self):
        """Main run method"""
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        logger.info("=" * 60)
        logger.info("Ollama CML Application Starting")
        logger.info("=" * 60)
        
        # Start Ollama server
        if not self.start_ollama_server():
            logger.error("Failed to start Ollama server. Exiting.")
            sys.exit(1)
        
        # Verify model is loaded
        time.sleep(2)  # Give Ollama a moment to fully initialize
        self.verify_model_loaded()
        
        logger.info("=" * 60)
        logger.info("Ollama CML Application Ready")
        logger.info(f"Access via: http://<cml-app-url>:11434")
        logger.info("=" * 60)
        
        # Monitor Ollama and keep application alive
        try:
            self.monitor_ollama()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.signal_handler(signal.SIGINT, None)

if __name__ == '__main__':
    service = OllamaService()
    service.run()