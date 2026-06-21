#!/bin/bash
# Kill vLLM server and free GPU memory
pkill -f "vllm serve" 2>/dev/null && echo "vLLM stopped." || echo "No vLLM process found."
sleep 2
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
