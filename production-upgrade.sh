#!/bin/bash

# ===========================================
# NTG360 Production Upgrade Script
# ===========================================
# This script upgrades the basic setup to full production
# with NVIDIA drivers, CUDA, and proper server directory structure

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

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root or with sudo"
fi

log "Starting NTG360 Production Upgrade..."

# ===========================================
# 1. INSTALL NVIDIA DRIVERS AND CUDA
# ===========================================
log "Installing NVIDIA drivers and CUDA toolkit..."

# Update package list
apt update

# Install prerequisites
apt install -y wget curl gnupg2 software-properties-common

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt update

# Install NVIDIA drivers (latest stable)
apt install -y nvidia-driver-535

# Install CUDA toolkit (version 12.2 for Tesla T4 compatibility)
apt install -y cuda-toolkit-12-2

# Install cuDNN for deep learning acceleration
apt install -y libcudnn8 libcudnn8-dev

# Install Python development packages
apt install -y python3.12-venv python3.12-dev python3-pip

log "NVIDIA drivers and CUDA toolkit installed"

# ===========================================
# 2. CREATE PROPER SERVER DIRECTORY STRUCTURE
# ===========================================
log "Creating proper server directory structure..."

# Create server-level directories following FHS
mkdir -p /opt/ntg360/{models,data,src,var,etc,usr,bin,sbin,lib,logs,tmp,run}
mkdir -p /opt/ntg360/models/{llm,embeddings,checkpoints,serving}
mkdir -p /opt/ntg360/data/{training,validation,testing,raw,processed}
mkdir -p /opt/ntg360/src/{ai,services,infrastructure,monitoring}
mkdir -p /opt/ntg360/var/{lib,log,cache,run,spool}
mkdir -p /opt/ntg360/var/lib/ntg360/{models,data,cache,state}
mkdir -p /opt/ntg360/var/log/ntg360/{ai,services,monitoring,audit}
mkdir -p /opt/ntg360/etc/ntg360/{ai,services,monitoring,security}
mkdir -p /opt/ntg360/usr/{bin,sbin,lib,share,local}
mkdir -p /opt/ntg360/usr/local/{bin,sbin,lib,share,etc}
mkdir -p /opt/ntg360/usr/share/ntg360/{models,data,docs,examples}

# Set proper permissions
chown -R ntg360:ntg360 /opt/ntg360/ 2>/dev/null || chown -R $(whoami):$(whoami) /opt/ntg360/
chmod -R 755 /opt/ntg360/
chmod -R 700 /opt/ntg360/var/lib/ntg360/
chmod -R 750 /opt/ntg360/var/log/ntg360/

log "Server directory structure created"

# ===========================================
# 3. SETUP PYTHON ENVIRONMENT
# ===========================================
log "Setting up Python environment for AI services..."

# Create virtual environment in proper server location
python3.12 -m venv /opt/ntg360/usr/local/venv
source /opt/ntg360/usr/local/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core AI libraries with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install numpy pandas scikit-learn
pip install fastapi uvicorn gunicorn
pip install pydantic python-multipart
pip install redis psycopg2-binary
pip install prometheus-client
pip install nvidia-ml-py3

log "Python environment setup complete"

# ===========================================
# 4. CREATE MODEL DOWNLOAD SCRIPT
# ===========================================
log "Creating model download script for server directory..."

cat > /opt/ntg360/usr/local/bin/download-models.py << 'EOF'
#!/usr/bin/env python3
"""
Download optimal LLM models for Tesla T4 GPU (16GB VRAM)
Models saved to proper server directory structure
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
    models_dir = Path("/opt/ntg360/models/llm")
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
    
    print("🎉 All models downloaded successfully to /opt/ntg360/models/llm/")

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/ntg360/usr/local/bin/download-models.py

# ===========================================
# 5. CREATE PRODUCTION AI SERVER
# ===========================================
log "Creating production AI server..."

cat > /opt/ntg360/src/ai/model_server.py << 'EOF'
#!/usr/bin/env python3
"""
NTG360 Production AI Model Server
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
import nvidia_ml_py3 as nvml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/ntg360/var/log/ntg360/ai/server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NTG360 Production AI Model Server",
    description="Production AI model serving for NTG360 platform",
    version="2.0.0"
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
    gpu_memory_used: float
    device_used: str

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    try:
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
    
    model_path = Path(f"/opt/ntg360/models/llm/{model_name}")
    if not model_path.exists():
        raise ValueError(f"Model {model_name} not found at {model_path}")
    
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
    return {"message": "NTG360 Production AI Model Server", "status": "running", "version": "2.0.0"}

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
        "models_loaded": len(models_cache),
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models_dir = Path("/opt/ntg360/models/llm")
    available_models = [d.name for d in models_dir.iterdir() if d.is_dir()] if models_dir.exists() else []
    loaded_models = list(models_cache.keys())
    
    return {
        "available_models": available_models,
        "loaded_models": loaded_models,
        "models_directory": str(models_dir)
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
        gpu_memory = get_gpu_memory_usage()
        
        return InferenceResponse(
            generated_text=generated_text,
            model_name=request.model_name,
            inference_time=inference_time,
            gpu_memory_used=gpu_memory,
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

# HELP cuda_available CUDA availability
# TYPE cuda_available gauge
cuda_available {1 if device == "cuda" else 0}
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

chmod +x /opt/ntg360/src/ai/model_server.py

# ===========================================
# 6. CREATE SYSTEMD SERVICES
# ===========================================
log "Creating systemd services..."

# Create ntg360 user if it doesn't exist
if ! id "ntg360" &>/dev/null; then
    useradd -r -s /bin/bash -d /opt/ntg360 -c "NTG360 Platform User" ntg360
fi

# Create systemd service for model server
cat > /etc/systemd/system/ntg360-ai-server.service << 'EOF'
[Unit]
Description=NTG360 Production AI Model Server
After=network.target

[Service]
Type=simple
User=ntg360
Group=ntg360
WorkingDirectory=/opt/ntg360
Environment=PATH=/opt/ntg360/usr/local/venv/bin
ExecStart=/opt/ntg360/usr/local/venv/bin/python /opt/ntg360/src/ai/model_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/ntg360/var

[Install]
WantedBy=multi-user.target
EOF

# Set ownership
chown -R ntg360:ntg360 /opt/ntg360/

log "Systemd services created"

# ===========================================
# 7. CREATE MANAGEMENT SCRIPTS
# ===========================================
log "Creating management scripts..."

# Create service management script
cat > /opt/ntg360/usr/local/bin/ai-services.sh << 'EOF'
#!/bin/bash
# NTG360 Production AI Services Management Script

case "$1" in
    start)
        echo "Starting NTG360 Production AI services..."
        systemctl start ntg360-ai-server
        systemctl enable ntg360-ai-server
        echo "AI services started"
        ;;
    stop)
        echo "Stopping NTG360 Production AI services..."
        systemctl stop ntg360-ai-server
        echo "AI services stopped"
        ;;
    restart)
        echo "Restarting NTG360 Production AI services..."
        systemctl restart ntg360-ai-server
        echo "AI services restarted"
        ;;
    status)
        echo "NTG360 Production AI Services Status:"
        systemctl status ntg360-ai-server --no-pager
        ;;
    logs)
        echo "AI Server Logs:"
        journalctl -u ntg360-ai-server -f
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

chmod +x /opt/ntg360/usr/local/bin/ai-services.sh

# Create model management script
cat > /opt/ntg360/usr/local/bin/model-manager.sh << 'EOF'
#!/bin/bash
# NTG360 Production Model Management Script

MODELS_DIR="/opt/ntg360/models/llm"

case "$1" in
    list)
        echo "Available models:"
        ls -la $MODELS_DIR
        ;;
    info)
        if [ -z "$2" ]; then
            echo "Usage: $0 info <model_name>"
            exit 1
        fi
        echo "Model info for: $2"
        du -sh $MODELS_DIR/$2
        ls -la $MODELS_DIR/$2
        ;;
    download)
        echo "Downloading models to server directory..."
        cd /opt/ntg360
        source /opt/ntg360/usr/local/venv/bin/activate
        python3 /opt/ntg360/usr/local/bin/download-models.py
        ;;
    clean)
        echo "Cleaning model cache..."
        rm -rf /opt/ntg360/var/lib/ntg360/cache/*
        echo "Cache cleaned"
        ;;
    *)
        echo "Usage: $0 {list|info|download|clean}"
        echo "  list     - List available models"
        echo "  info     - Show model information"
        echo "  download - Download all models to server directory"
        echo "  clean    - Clean model cache"
        exit 1
        ;;
esac
EOF

chmod +x /opt/ntg360/usr/local/bin/model-manager.sh

# ===========================================
# 8. CREATE SYMBOLIC LINKS
# ===========================================
log "Creating symbolic links..."

ln -sf /opt/ntg360/usr/local/bin/ai-services.sh /usr/local/bin/ntg360-ai
ln -sf /opt/ntg360/usr/local/bin/model-manager.sh /usr/local/bin/ntg360-models

# ===========================================
# 9. CONFIGURE NETWORK AND FIREWALL
# ===========================================
log "Configuring network settings for VLAN 60..."

# Create network configuration
cat > /opt/ntg360/etc/ntg360/network.conf << 'EOF'
# NTG360 AI Services Network Configuration
# VLAN 60 - AI Services Network

# AI Service Ports (from firewall aliases)
AI_PORTS="11434 11435 8000 8001 8888 8889 6006 5000 5001 8080"

# Model Server Configuration
MODEL_SERVER_PORT=8000
MODEL_SERVER_HOST=0.0.0.0

# Health Check Port
HEALTH_CHECK_PORT=8001

# Metrics Port
METRICS_PORT=8888

# Admin Interface Port
ADMIN_PORT=8889
EOF

# Create ufw rules for AI services (if ufw is enabled)
if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
    log "Configuring UFW rules for AI services..."
    
    # Allow AI service ports
    for port in 8000 8001 8888 8889; do
        ufw allow from 10.60.0.0/16 to any port $port comment "NTG360 AI Services"
    done
    
    # Allow SSH from management
    ufw allow from 10.0.0.0/8 to any port 22 comment "Management SSH"
    
    log "UFW rules configured"
fi

log "Network configuration complete"

# ===========================================
# 10. FINAL SETUP AND VERIFICATION
# ===========================================
log "Performing final setup and verification..."

# Reload systemd
systemctl daemon-reload

# Create configuration file
cat > /opt/ntg360/etc/ntg360/ai.conf << 'EOF'
# NTG360 Production AI Configuration
# Tesla T4 GPU Setup

[gpu]
device_id=0
memory_fraction=0.9
allow_growth=true

[models]
cache_dir=/opt/ntg360/var/lib/ntg360/cache
models_dir=/opt/ntg360/models/llm
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

# Set final permissions
chown -R ntg360:ntg360 /opt/ntg360/
chmod -R 755 /opt/ntg360/usr/local/bin/

log "Final setup complete"

# ===========================================
# 11. DISPLAY COMPLETION SUMMARY
# ===========================================
echo ""
echo "==========================================="
echo "🎉 PRODUCTION UPGRADE COMPLETE!"
echo "==========================================="
echo ""
echo "📁 Server Directory Structure:"
echo "   /opt/ntg360/models/llm/     - LLM models (server location)"
echo "   /opt/ntg360/src/ai/         - AI source code"
echo "   /opt/ntg360/var/log/ntg360/ - Logs"
echo "   /opt/ntg360/etc/ntg360/     - Configuration"
echo ""
echo "🚀 Services:"
echo "   ntg360-ai-server.service    - Production model serving"
echo ""
echo "🔧 Management Commands:"
echo "   ntg360-ai start|stop|restart|status|logs|test"
echo "   ntg360-models list|info|download|clean"
echo ""
echo "🌐 Endpoints:"
echo "   http://localhost:8000/       - Model server"
echo "   http://localhost:8000/health - Health check"
echo "   http://localhost:8000/metrics - Prometheus metrics"
echo ""
echo "🔒 Network Configuration:"
echo "   VLAN 60 AI services configured"
echo "   Ports 8000, 8001, 8888, 8889 open"
echo ""
echo "Next steps:"
echo "1. Download models: ntg360-models download"
echo "2. Start services: ntg360-ai start"
echo "3. Test setup: ntg360-ai test"
echo ""
echo "==========================================="

log "Production upgrade completed successfully!"
