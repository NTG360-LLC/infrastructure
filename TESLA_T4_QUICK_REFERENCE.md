# Tesla T4 GPU Setup - Quick Reference

## 🚀 Quick Start Commands

```bash
# Run setup (as root/sudo)
sudo chmod +x setup-tesla-t4-gpu.sh
sudo ./setup-tesla-t4-gpu.sh

# Start services
ntg360-ai start

# Test setup
ntg360-ai test
```

## 📁 Key Directories

| Directory | Purpose |
|-----------|---------|
| `/opt/ntg360/models/llm/` | LLM models storage |
| `/opt/ntg360/src/ai/` | AI service source code |
| `/opt/ntg360/var/log/ntg360/ai/` | AI service logs |
| `/opt/ntg360/etc/ntg360/` | Configuration files |

## 🔧 Management Commands

### AI Services
```bash
ntg360-ai start|stop|restart|status
ntg360-ai logs          # Server logs
ntg360-ai gpu-logs      # GPU monitoring
ntg360-ai test          # Test API
```

### Model Management
```bash
ntg360-models list      # List models
ntg360-models info <name>  # Model details
ntg360-models download  # Download models
ntg360-models clean     # Clean cache
```

## 🌐 API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `http://localhost:8000/` | Server status |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/models` | List models |
| `http://localhost:8000/inference` | Generate text |
| `http://localhost:8000/metrics` | Prometheus metrics |

## 📊 Models Included

| Model | Size | Use Case |
|-------|------|----------|
| DialoGPT-medium | 345M | Conversations |
| DialoGPT-large | 774M | Advanced chat |
| BlenderBot-400M | 400M | Multi-turn dialogue |
| CodeGPT-small | 124M | Code generation |
| DistilBERT-base | 66M | Text embeddings |
| Sentence-Transformers | 22M-109M | Semantic search |

## 🔒 Network Configuration

### Ports (VLAN 60)
- **8000**: Model server API
- **8001**: Health check
- **8888**: Metrics
- **8889**: Admin interface

### Firewall Rules
- Inbound: `10.60.0.0/16` (VLAN 60 AI subnets)
- Outbound: Trusted ports (80, 443, 53, 123)
- Management: SSH from `10.0.0.0/8`

## 📈 Monitoring

### GPU Metrics
```bash
# Real-time GPU status
nvidia-smi

# GPU monitoring logs
ntg360-ai gpu-logs

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Key Metrics
- GPU memory usage (GB)
- GPU utilization (%)
- Temperature (°C)
- Power consumption (W)
- Inference time (ms)

## 🛠️ Troubleshooting

### Common Issues
```bash
# GPU not detected
lspci | grep -i nvidia
nvidia-smi

# Service won't start
systemctl status ntg360-ai-server
journalctl -u ntg360-ai-server -f

# Out of memory
nvidia-smi
# Reduce max_models_in_memory in config
```

### Performance Tuning
- **Higher throughput**: Increase `max_models_in_memory`
- **Lower latency**: Pre-load models, use smaller models
- **Memory optimization**: Enable FP16, reduce cache size

## 📋 Example API Usage

```bash
# Health check
curl http://localhost:8000/health

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

## 🔄 Maintenance

### Regular Tasks
- Check logs: `ntg360-ai logs`
- Monitor GPU: `ntg360-ai gpu-logs`
- Clean cache: `ntg360-models clean`
- Update models: `ntg360-models download`

### Backup
- Models: `/opt/ntg360/models/`
- Config: `/opt/ntg360/etc/`
- Logs: Auto-rotated daily

---

**Need help?** Check the full guide: `TESLA_T4_GPU_SETUP_GUIDE.md`
