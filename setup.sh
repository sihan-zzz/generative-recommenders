# conda remove --name  hstu --all -y

#!/bin/bash
# Create and activate a clean environment for generative/recsys workloads

ENV_NAME=hstu1
PYTHON_VERSION=3.10
CUDA_VERSION=12.8

# --- create base env ---
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
eval "$(conda shell.bash hook)"
conda init
conda activate ${ENV_NAME}

# --- core PyTorch stack ---
# PyTorch 2.8 built for CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# --- fbgemm-gpu (embedding kernels) ---
# Compatible with PyTorch 2.8.x / CUDA 12.8 / Python 3.10
pip install fbgemm-gpu==1.3.0 --index-url https://download.pytorch.org/whl/cu128

# # --- optional utilities ---
# pip install numpy pandas tqdm orjson rich
pip install -r requirements.txt
echo ""
echo "✅ Environment '${ENV_NAME}' setup complete. Testing installations..."
python - <<'PY'
import torch, fbgemm_gpu
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("FBGEMM_GPU:", getattr(fbgemm_gpu, "__version__", "unknown"))
PY
echo "✅ fbgemm-gpu installed."
python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
try:
    import fbgemm_gpu
    print("fbgemm_gpu:", getattr(fbgemm_gpu, "__version__", "unknown"))
    print("has op?", hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"))
    if hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"):
        x = torch.tensor([0,2,5,5,9], device="cpu", dtype=torch.int64)
        print("call ok:", torch.ops.fbgemm.asynchronous_complete_cumsum(x) is not None)
except Exception as e:
    print("import fbgemm_gpu failed:", e)
PY
echo "✅ torch.ops.fbgemm test successful."