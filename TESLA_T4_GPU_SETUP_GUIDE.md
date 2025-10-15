# NTG360 Tesla T4 GPU Setup Guide

## Overview

This guide provides comprehensive instructions for setting up a Tesla T4 GPU for internal AI service models on the NTG360 platform. The setup follows proper server directory structure, VLAN 60 network requirements, and enterprise-grade best practices.

## Prerequisites

- Ubuntu 22.04 LTS or compatible Debian-based system
- Tesla T4 GPU (16GB VRAM)
- Root or sudo access
- Internet connectivity for package downloads
- Minimum 50GB free disk space for models and dependencies

## Quick Start

1. **Run the setup script:**
   ```bash
   sudo chmod +x setup-tesla-t4-gpu.sh
   sudo ./setup-tesla-t4-gpu.sh
   ```

2. **Start the AI services:**
   ```bash
   ntg360-ai start
   ```

3. **Test the setup:**
   ```bash
   ntg360-ai test
   ```

## Directory Structure

The setup creates a comprehensive directory structure following Linux FHS standards:

```
/opt/ntg360/
├── models/llm/                    # LLM model storage
├── src/ai/                        # AI service source code
├── var/log/ntg360/ai/            # AI service logs
├── var/lib/ntg360/cache/         # Model cache
├── etc/ntg360/                   # Configuration files
├── usr/local/venv/               # Python virtual environment
└── usr/local/bin/                # Management scripts
```

## Optimal LLM Models for Tesla T4

The setup automatically downloads models optimized for Tesla T4's 16GB VRAM:

### Causal Language Models
- **DialoGPT-medium** (345M parameters) - Conversational AI
- **DialoGPT-large** (774M parameters) - Advanced conversations
- **BlenderBot-400M** (400M parameters) - Multi-turn dialogue
- **CodeGPT-small** (124M parameters) - Code generation

### Embedding Models
- **DistilBERT-base** (66M parameters) - Text embeddings
- **Sentence-Transformers MiniLM** (22M parameters) - Fast embeddings
- **Sentence-Transformers MPNet** (109M parameters) - High-quality embeddings

### Model Selection Rationale
- **Memory Efficiency**: All models fit comfortably in 16GB VRAM
- **Performance Balance**: Optimized for inference speed vs. quality
- **Use Case Coverage**: Conversation, code generation, and embeddings
- **Half Precision**: Models use FP16 for 2x memory efficiency

## Network Configuration (VLAN 60)

### Firewall Rules
The setup configures UFW rules for AI services:

```bash
# AI Service Ports
8000 - Model server (FastAPI)
8001 - Health check endpoint
8888 - Prometheus metrics
8889 - Admin interface

# Network Access
- Inbound: 10.60.0.0/16 (VLAN 60 AI subnets)
- Outbound: Trusted ports (80, 443, 53, 123)
- Management: SSH from 10.0.0.0/8
```

### Port Aliases (from firewall config)
- `AI_PORTS_ALL`: 11434, 11435, 8000, 8001, 8888, 8889, 6006, 5000, 5001, 8080
- `WEB_PORTS_ALL`: 80, 443
- `SSH_STANDARD`: 22

## Services and Management

### Systemd Services
- `ntg360-ai-server.service` - Model serving API
- `ntg360-gpu-monitor.service` - GPU monitoring

### Management Commands

#### AI Services Management
```bash
ntg360-ai start      # Start all AI services
ntg360-ai stop       # Stop all AI services
ntg360-ai restart    # Restart all AI services
ntg360-ai status     # Check service status
ntg360-ai logs       # View server logs
ntg360-ai gpu-logs   # View GPU monitoring logs
ntg360-ai test       # Test API endpoints
```

#### Model Management
```bash
ntg360-models list           # List available models
ntg360-models info <name>    # Show model information
ntg360-models download       # Download all models
ntg360-models clean          # Clean model cache
```

## API Endpoints

### Model Server (Port 8000)
- `GET /` - Server status
- `GET /health` - Health check with system metrics
- `GET /models` - List available and loaded models
- `POST /inference` - Generate text using models
- `GET /metrics` - Prometheus metrics

### Example API Usage
```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Generate text
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "dialogpt-medium",
    "prompt": "Hello, how are you?",
    "max_length": 50,
    "temperature": 0.7
  }'
```

## Monitoring and Logging

### GPU Monitoring
- **Real-time monitoring**: GPU utilization, memory usage, temperature
- **Log location**: `/opt/ntg360/var/log/ntg360/ai/gpu.log`
- **Metrics**: Available at `http://localhost:8000/metrics`

### Log Management
- **Automatic rotation**: Daily log rotation with 30-day retention
- **Log locations**:
  - AI server: `journalctl -u ntg360-ai-server`
  - GPU monitor: `/opt/ntg360/var/log/ntg360/ai/gpu.log`

### Key Metrics
- GPU memory usage (GB)
- GPU utilization (%)
- Temperature (°C)
- Power consumption (W)
- Model inference time
- CPU and system memory usage

## Security Configuration

### Service Security
- **User isolation**: Services run as `ntg360` user
- **File permissions**: Restricted access to sensitive directories
- **Systemd security**: NoNewPrivileges, PrivateTmp, ProtectSystem

### Network Security
- **VLAN isolation**: AI services on VLAN 60 only
- **Firewall rules**: Restricted to AI subnets
- **No direct WAN access**: All traffic through reverse proxy

## Performance Optimization

### GPU Optimization
- **Half precision**: FP16 for 2x memory efficiency
- **Model caching**: In-memory model caching
- **Single worker**: Optimized for GPU memory usage
- **Device mapping**: Automatic GPU placement

### Memory Management
- **Model loading**: Lazy loading with caching
- **Memory fraction**: 90% GPU memory allocation
- **Cache management**: Automatic cleanup of unused models

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check GPU presence
lspci | grep -i nvidia

# Check driver status
nvidia-smi

# Reinstall drivers if needed
sudo apt install --reinstall nvidia-driver-535
```

#### Service Won't Start
```bash
# Check service status
systemctl status ntg360-ai-server

# View detailed logs
journalctl -u ntg360-ai-server -f

# Check permissions
ls -la /opt/ntg360/
```

#### Out of Memory Errors
```bash
# Check GPU memory
nvidia-smi

# Reduce model cache
# Edit /opt/ntg360/etc/ntg360/ai.conf
# Set max_models_in_memory=1

# Restart services
ntg360-ai restart
```

### Performance Tuning

#### For Higher Throughput
1. Increase `max_models_in_memory` in config
2. Use smaller models for faster inference
3. Enable model quantization

#### For Lower Latency
1. Pre-load frequently used models
2. Use smaller batch sizes
3. Enable model caching

## Maintenance

### Regular Tasks
- **Log rotation**: Automatic daily rotation
- **Model updates**: Use `ntg360-models download` for updates
- **Cache cleanup**: Use `ntg360-models clean` periodically
- **Health monitoring**: Check `/health` endpoint regularly

### Backup Strategy
- **Model backup**: Backup `/opt/ntg360/models/` directory
- **Configuration backup**: Backup `/opt/ntg360/etc/` directory
- **Log archival**: Logs rotated and compressed automatically

## Integration with NTG360 Platform

### Service Integration
- **Internal API**: Models accessible via internal API calls
- **Load balancing**: Can be load balanced across multiple instances
- **Health checks**: Integrated with platform monitoring
- **Metrics**: Prometheus metrics for platform monitoring

### Development Integration
- **Local development**: Use `ntg360-ai test` for testing
- **Model development**: Add new models to download script
- **API development**: Extend FastAPI endpoints as needed

## Support and Documentation

### Additional Resources
- **NVIDIA Tesla T4 Documentation**: [NVIDIA Developer](https://developer.nvidia.com/tesla-t4)
- **CUDA Toolkit Documentation**: [CUDA Toolkit](https://docs.nvidia.com/cuda/)
- **Transformers Library**: [Hugging Face](https://huggingface.co/docs/transformers/)

### Getting Help
1. Check logs: `ntg360-ai logs` and `ntg360-ai gpu-logs`
2. Verify GPU: `nvidia-smi`
3. Test API: `ntg360-ai test`
4. Check network: Verify VLAN 60 connectivity

---

**Note**: This setup is optimized for internal use within the NTG360 platform. All models and services are configured for maximum security and performance within the VLAN 60 AI network segment.
