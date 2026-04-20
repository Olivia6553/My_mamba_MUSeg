import os
import importlib
import torch

print("=== Python / Torch ===")
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))

print("\n=== Basic packages ===")
mods = [
    "torchvision",
    "numpy",
    "timm",
    "einops",
    "yaml",
]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL -> {e}")

print("\n=== Custom CUDA cores ===")
for m in ["adaptive_selective_scan_cuda_core", "selective_scan_cuda_core"]:
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL -> {e}")

print("\n=== Done ===")