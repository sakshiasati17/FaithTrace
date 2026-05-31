"""
Inference benchmarking harness.

Measures p50, p95, p99 latency and throughput across:
  - Raw PyTorch (FP32 baseline)
  - TensorRT FP32 / FP16 / INT8

WHY PERCENTILE LATENCIES?
Average hides tail behaviour. p99 = 200 ms on a 10 ms average means 1% of
users experience 20x degradation. Production SLAs are defined at p95/p99.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class InferenceBenchmark:
    def __init__(self, warmup_runs: int = 50, benchmark_runs: int = 500):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def benchmark_pytorch(self, model, dummy_inputs, device: str = "cuda") -> dict:
        model.eval().to(device)
        inputs = [inp.to(device) for inp in dummy_inputs]

        for _ in range(self.warmup_runs):
            with torch.no_grad():
                model(*inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(self.benchmark_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model(*inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        return self._stats(latencies, "pytorch_fp32")

    def benchmark_tensorrt(self, engine_path: str, dummy_inputs_np: list) -> dict:
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError:
            raise RuntimeError("TensorRT / pycuda not installed")

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        inputs_gpu = []
        for inp in dummy_inputs_np:
            buf = cuda.mem_alloc(inp.nbytes)
            cuda.memcpy_htod(buf, inp)
            inputs_gpu.append(buf)

        out_shape = context.get_tensor_shape(engine.get_tensor_name(len(dummy_inputs_np)))
        output = np.empty(out_shape, dtype=np.float32)
        output_gpu = cuda.mem_alloc(output.nbytes)
        bindings = [int(g) for g in inputs_gpu] + [int(output_gpu)]

        for _ in range(self.warmup_runs):
            context.execute_v2(bindings=bindings)
        cuda.Context.synchronize()

        latencies = []
        for _ in range(self.benchmark_runs):
            cuda.Context.synchronize()
            start = time.perf_counter()
            context.execute_v2(bindings=bindings)
            cuda.Context.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        precision = Path(engine_path).stem.split("_")[-1]
        return self._stats(latencies, f"tensorrt_{precision}")

    def _stats(self, latencies: list, label: str) -> dict:
        arr = np.array(latencies)
        stats = {
            "label": label,
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
            "std_ms": round(float(np.std(arr)), 3),
            "throughput_qps": round(1000 / float(np.mean(arr)), 1),
            "num_runs": len(latencies),
        }
        print(
            f"\n{'='*50}\n  {label}\n{'='*50}\n"
            f"  p50: {stats['p50_ms']} ms\n"
            f"  p95: {stats['p95_ms']} ms\n"
            f"  p99: {stats['p99_ms']} ms\n"
            f"  Throughput: {stats['throughput_qps']} qps"
        )
        return stats

    def full_report(self, model, engine_paths: dict, dummy_inputs) -> dict:
        results = []
        pytorch_stats = self.benchmark_pytorch(model, dummy_inputs)
        results.append(pytorch_stats)

        dummy_np = [inp.cpu().numpy() for inp in dummy_inputs]
        for precision, path in engine_paths.items():
            trt_stats = self.benchmark_tensorrt(path, dummy_np)
            trt_stats["speedup_vs_pytorch"] = round(
                pytorch_stats["p50_ms"] / trt_stats["p50_ms"], 2
            )
            results.append(trt_stats)

        report = {
            "timestamp": datetime.now().isoformat(),
            "gpu": self._gpu_info(),
            "results": results,
        }
        report_path = "benchmarks/latest_report.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {report_path}")
        return report

    def _gpu_info(self) -> dict:
        if torch.cuda.is_available():
            return {
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        return {"name": "CPU", "vram_gb": 0}
