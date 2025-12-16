#!/bin/bash

# Setup script for DPO training with QLoRA

echo "================================================"
echo "DPO Training Setup for Llama-3-8B with QLoRA"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "PyTorch not installed yet"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -q -r requirements.txt
echo "Dependencies installed!"
echo ""

# Check installations
echo "Verifying installations..."
python3 << EOF
import sys
try:
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
except ImportError:
    print("✗ transformers not found")
    sys.exit(1)

try:
    import trl
    print(f"✓ trl: {trl.__version__}")
except ImportError:
    print("✗ trl not found")
    sys.exit(1)

try:
    import peft
    print(f"✓ peft: {peft.__version__}")
except ImportError:
    print("✗ peft not found")
    sys.exit(1)

try:
    import bitsandbytes
    print(f"✓ bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    print("✗ bitsandbytes not found")
    sys.exit(1)

try:
    import torch
    print(f"✓ torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ torch not found")
    sys.exit(1)
EOF

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Login to Hugging Face:"
echo "   huggingface-cli login"
echo ""
echo "2. (Optional) Login to Weights & Biases:"
echo "   wandb login"
echo ""
echo "3. Start training:"
echo "   python train_dpo_qlora.py"
echo ""
echo "4. After training, run inference:"
echo "   python inference.py"
echo ""
echo "For more details, see README.md"
echo "================================================"
