"""
GPU Auto-Profiler — detects hardware at runtime and selects the optimal
TensorRT engine precision + batch size.

Decision tree:
  VRAM < 6 GB               → INT8,  batch 4   (aggressive memory saving)
  Compute ≥ 8.0 (Ampere+)   → FP16,  batch 32  (native tensor cores)
  Compute ≥ 7.0 (Volta/T4)  → FP16,  batch 16
  Older GPU                 → FP32,  batch 8

Uses pynvml — the same library nvidia-smi uses internally.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPUProfile:
    name: str
    vram_total_gb: float
    vram_available_gb: float
    compute_capability: tuple
    recommended_precision: str
    max_batch_size: int
    reasoning: str


class GPUAutoProfiler:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_available = True
            self.device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            self.gpu_available = False
            self.device_count = 0

    def profile(self, device_index: int = 0) -> GPUProfile:
        if not self.gpu_available:
            return GPUProfile(
                name="CPU (no GPU detected)",
                vram_total_gb=0,
                vram_available_gb=0,
                compute_capability=(0, 0),
                recommended_precision="fp32",
                max_batch_size=4,
                reasoning="No GPU found. Install NVIDIA drivers + CUDA for acceleration.",
            )

        handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = self.pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total = round(mem.total / (1024 ** 3), 1)
        vram_avail = round(mem.free / (1024 ** 3), 1)
        major, minor = self.pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        cap = (major, minor)

        if vram_avail < 6:
            precision, batch = "int8", 4
            reason = (
                f"Low VRAM ({vram_avail} GB). INT8 minimises memory footprint. "
                f"Small batch size avoids OOM."
            )
        elif cap >= (8, 0):
            precision, batch = "fp16", 32
            reason = (
                f"Ampere+ GPU (compute {major}.{minor}). Native FP16 tensor cores "
                f"give ~2x speedup with negligible accuracy loss."
            )
        elif cap >= (7, 0):
            precision, batch = "fp16", 16
            reason = (
                f"Volta/Turing GPU (compute {major}.{minor}). FP16 tensor cores "
                f"available with moderate batch size for {vram_avail} GB VRAM."
            )
        else:
            precision, batch = "fp32", 8
            reason = (
                f"Older GPU (compute {major}.{minor}). No significant FP16 benefit; "
                f"using FP32 for maximum compatibility."
            )

        return GPUProfile(
            name=name,
            vram_total_gb=vram_total,
            vram_available_gb=vram_avail,
            compute_capability=cap,
            recommended_precision=precision,
            max_batch_size=batch,
            reasoning=reason,
        )

    def select_engine(self, engine_dir: str, model_name: str) -> tuple:
        """Pick the best available TensorRT engine for the current GPU."""
        profile = self.profile()
        fallback_order = {
            "int8": ["int8", "fp16", "fp32"],
            "fp16": ["fp16", "fp32"],
            "fp32": ["fp32"],
        }
        for precision in fallback_order.get(profile.recommended_precision, ["fp32"]):
            candidate = f"{engine_dir}/{model_name}_{precision}.engine"
            if Path(candidate).exists():
                print(f"GPU: {profile.name}")
                print(f"Selected: {candidate}")
                print(f"Reason: {profile.reasoning}")
                return candidate, profile
        raise FileNotFoundError(f"No TensorRT engine found for {model_name} in {engine_dir}")
