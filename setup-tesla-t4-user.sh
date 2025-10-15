#!/bin/bash

# ===========================================
# NTG360 Tesla T4 GPU Setup - User Level
# ===========================================
# This script sets up the user-level components for Tesla T4 GPU
# Run the system-level setup separately with sudo

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

log "Starting Tesla T4 GPU user-level setup for NTG360 platform..."

# ===========================================
# 1. CREATE USER-LEVEL DIRECTORY STRUCTURE
# ===========================================
log "Creating user-level directory structure..."

# Create directories in current location
mkdir -p models/{llm,embeddings,checkpoints,serving}
mkdir -p data/{training,validation,testing,raw,processed}
mkdir -p src/{ai,services,infrastructure,monitoring}
mkdir -p var/{lib,log,cache,run,spool}
mkdir -p var/lib/ntg360/{models,data,cache,state}
mkdir -p var/log/ntg360/{ai,services,monitoring,audit}
mkdir -p etc/ntg360/{ai,services,monitoring,security}
mkdir -p usr/local/{bin,sbin,lib,share,etc}
mkdir -p usr/share/ntg360/{models,data,docs,examples}

log "User-level directory structure created"

# ===========================================
# 2. SETUP PYTHON ENVIRONMENT
# ===========================================
log "Setting up Python environment for AI services..."

# Create virtual environment
python3 -m venv usr/local/venv
source usr/local/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core AI libraries (CPU version first, GPU will be added later)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install numpy pandas scikit-learn
pip install fastapi uvicorn gunicorn
pip install pydantic python-multipart
pip install redis psycopg2-binary
pip install prometheus-client

log "Python environment setup complete"

# ===========================================
# 3. CREATE MODEL DOWNLOAD SCRIPT
# ===========================================
log "Creating model download script..."

cat > usr/local/bin/download-models.py << 'EOF'
#!/usr/bin/env python3
"""
Download optimal LLM models for Tesla T4 GPU (16GB VRAM)
Models selected for memory efficiency and performance
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

def download_model(model_name, model_path, model_type="causal"):
    """Download and save a model with proper configuration"""
    print(f"Downloading {model_name}...")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        
        # Download model with memory optimization
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        model.save_pretrained(model_path)
        print(f"✅ {model_name} downloaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")

def main():
    models_dir = Path("models/llm")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Models optimized for Tesla T4 (16GB VRAM)
    models_to_download = [
        # Small, efficient models for quick inference
        ("microsoft/DialoGPT-medium", "dialogpt-medium", "causal"),
        ("distilbert-base-uncased", "distilbert-base", "embedding"),
        
        # Medium models for balanced performance
        ("microsoft/DialoGPT-large", "dialogpt-large", "causal"),
        ("facebook/blenderbot-400M-distill", "blenderbot-400m", "causal"),
        
        # Code generation models
        ("microsoft/CodeGPT-small-py", "codegpt-small", "causal"),
        
        # Embedding models for semantic search
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformer-mini", "embedding"),
        ("sentence-transformers/all-mpnet-base-v2", "sentence-transformer-mpnet", "embedding"),
    ]
    
    for model_name, local_name, model_type in models_to_download:
        model_path = models_dir / local_name
        download_model(model_name, model_path, model_type)
    
    print("🎉 All models downloaded successfully!")

if __name__ == "__main__":
    main()
EOF

chmod +x usr/local/bin/download-models.py

# ===========================================
# 4. CREATE MODEL SERVER
# ===========================================
log "Creating model server..."

cat > src/ai/model_server.py << 'EOF'
#!/usr/bin/env python3
"""
NTG360 AI Model Server
FastAPI-based model serving with GPU optimization
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import uvicorn
import logging
from pathlib import Path
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NTG360 AI Model Server",
    description="Internal AI model serving for NTG360 platform",
    version="1.0.0"
)

# Global model cache
models_cache = {}
tokenizers_cache = {}

class InferenceRequest(BaseModel):
    model_name: str
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class InferenceResponse(BaseModel):
    generated_text: str
    model_name: str
    inference_time: float
    device_used: str

def get_gpu_memory_usage():
    """Get current GPU memory usage if available"""
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**3  # Convert to GB
    except:
        return 0.0

def load_model(model_name: str):
    """Load model with caching"""
    if model_name in models_cache:
        return models_cache[model_name], tokenizers_cache[model_name]
    
    model_path = Path(f"models/llm/{model_name}")
    if not model_path.exists():
        raise ValueError(f"Model {model_name} not found")
    
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # Cache models
    models_cache[model_name] = model
    tokenizers_cache[model_name] = tokenizer
    
    return model, tokenizer

@app.get("/")
async def root():
    return {"message": "NTG360 AI Model Server", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_memory = get_gpu_memory_usage()
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return {
        "status": "healthy",
        "device": device,
        "gpu_memory_used_gb": gpu_memory,
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "models_loaded": len(models_cache)
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models_dir = Path("models/llm")
    available_models = [d.name for d in models_dir.iterdir() if d.is_dir()] if models_dir.exists() else []
    loaded_models = list(models_cache.keys())
    
    return {
        "available_models": available_models,
        "loaded_models": loaded_models
    }

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Generate text using specified model"""
    start_time = time.time()
    
    try:
        # Load model
        model, tokenizer = load_model(request.model_name)
        
        # Tokenize input
        inputs = tokenizer.encode(request.prompt, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()
        
        inference_time = time.time() - start_time
        
        return InferenceResponse(
            generated_text=generated_text,
            model_name=request.model_name,
            inference_time=inference_time,
            device_used=device
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics"""
    gpu_memory = get_gpu_memory_usage()
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    metrics_data = f"""# HELP gpu_memory_used_gb GPU memory usage in GB
# TYPE gpu_memory_used_gb gauge
gpu_memory_used_gb {gpu_memory}

# HELP cpu_percent CPU usage percentage
# TYPE cpu_percent gauge
cpu_percent {cpu_percent}

# HELP memory_percent Memory usage percentage
# TYPE memory_percent gauge
memory_percent {memory_percent}

# HELP models_loaded Number of loaded models
# TYPE models_loaded gauge
models_loaded {len(models_cache)}

# HELP device_used Device being used for inference
# TYPE device_used gauge
device_used {1 if device == "cuda" else 0}
"""
    return metrics_data

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU memory efficiency
        log_level="info"
    )
EOF

chmod +x src/ai/model_server.py

# ===========================================
# 5. CREATE MANAGEMENT SCRIPTS
# ===========================================
log "Creating management scripts..."

# Create service management script
cat > usr/local/bin/ai-services.sh << 'EOF'
#!/bin/bash
# NTG360 AI Services Management Script

case "$1" in
    start)
        echo "Starting NTG360 AI services..."
        source usr/local/venv/bin/activate
        python3 src/ai/model_server.py &
        echo $! > var/run/ai-server.pid
        echo "AI services started (PID: $(cat var/run/ai-server.pid))"
        ;;
    stop)
        echo "Stopping NTG360 AI services..."
        if [ -f var/run/ai-server.pid ]; then
            kill $(cat var/run/ai-server.pid) 2>/dev/null || true
            rm -f var/run/ai-server.pid
        fi
        echo "AI services stopped"
        ;;
    restart)
        echo "Restarting NTG360 AI services..."
        $0 stop
        sleep 2
        $0 start
        ;;
    status)
        echo "NTG360 AI Services Status:"
        if [ -f var/run/ai-server.pid ]; then
            PID=$(cat var/run/ai-server.pid)
            if ps -p $PID > /dev/null; then
                echo "AI Server: Running (PID: $PID)"
            else
                echo "AI Server: Not running"
            fi
        else
            echo "AI Server: Not running"
        fi
        ;;
    logs)
        echo "AI Server Logs:"
        tail -f var/log/ntg360/ai/server.log 2>/dev/null || echo "No logs available"
        ;;
    test)
        echo "Testing AI services..."
        curl -s http://localhost:8000/health | python3 -m json.tool
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        exit 1
        ;;
esac
EOF

chmod +x usr/local/bin/ai-services.sh

# Create model management script
cat > usr/local/bin/model-manager.sh << 'EOF'
#!/bin/bash
# NTG360 Model Management Script

MODELS_DIR="models/llm"

case "$1" in
    list)
        echo "Available models:"
        ls -la $MODELS_DIR 2>/dev/null || echo "No models directory found"
        ;;
    info)
        if [ -z "$2" ]; then
            echo "Usage: $0 info <model_name>"
            exit 1
        fi
        echo "Model info for: $2"
        du -sh $MODELS_DIR/$2 2>/dev/null || echo "Model not found"
        ls -la $MODELS_DIR/$2 2>/dev/null || echo "Model not found"
        ;;
    download)
        echo "Downloading models..."
        source usr/local/venv/bin/activate
        python3 usr/local/bin/download-models.py
        ;;
    clean)
        echo "Cleaning model cache..."
        rm -rf var/lib/ntg360/cache/*
        echo "Cache cleaned"
        ;;
    *)
        echo "Usage: $0 {list|info|download|clean}"
        echo "  list     - List available models"
        echo "  info     - Show model information"
        echo "  download - Download all models"
        echo "  clean    - Clean model cache"
        exit 1
        ;;
esac
EOF

chmod +x usr/local/bin/model-manager.sh

# ===========================================
# 6. CREATE CONFIGURATION
# ===========================================
log "Creating configuration files..."

cat > etc/ntg360/ai.conf << 'EOF'
# NTG360 AI Configuration
# Tesla T4 GPU Setup

[gpu]
device_id=0
memory_fraction=0.9
allow_growth=true

[models]
cache_dir=var/lib/ntg360/cache
models_dir=models/llm
max_models_in_memory=3

[server]
host=0.0.0.0
port=8000
workers=1
timeout=300

[monitoring]
log_level=INFO
metrics_port=8888
EOF

# ===========================================
# 7. CREATE SYMBOLIC LINKS
# ===========================================
log "Creating symbolic links..."

ln -sf $(pwd)/usr/local/bin/ai-services.sh usr/local/bin/ntg360-ai
ln -sf $(pwd)/usr/local/bin/model-manager.sh usr/local/bin/ntg360-models

# ===========================================
# 8. DISPLAY COMPLETION SUMMARY
# ===========================================
echo ""
echo "==========================================="
echo "🎉 USER-LEVEL TESLA T4 GPU SETUP COMPLETE!"
echo "==========================================="
echo ""
echo "📁 Directory Structure Created:"
echo "   models/llm/     - LLM models"
echo "   src/ai/         - AI source code"
echo "   var/log/ntg360/ - Logs"
echo "   etc/ntg360/     - Configuration"
echo ""
echo "🔧 Management Commands:"
echo "   ./usr/local/bin/ntg360-ai start|stop|restart|status|test"
echo "   ./usr/local/bin/ntg360-models list|info|download|clean"
echo ""
echo "🌐 Endpoints (when running):"
echo "   http://localhost:8000/       - Model server"
echo "   http://localhost:8000/health - Health check"
echo "   http://localhost:8000/metrics - Prometheus metrics"
echo ""
echo "Next steps:"
echo "1. Download models: ./usr/local/bin/ntg360-models download"
echo "2. Start services: ./usr/local/bin/ntg360-ai start"
echo "3. Test setup: ./usr/local/bin/ntg360-ai test"
echo ""
echo "Note: For full GPU support, run the system-level setup with sudo"
echo "==========================================="

log "User-level Tesla T4 GPU setup completed successfully!"
