"""
Optimization endpoints — GPU profiling, ONNX export, TensorRT benchmarks.

GET  /optimization/gpu-profile           — detect GPU and return recommended precision
POST /optimization/export/classifier     — export trained PyTorch classifier to ONNX
POST /optimization/benchmark             — run full precision benchmark (async via Celery)
GET  /optimization/benchmark/latest      — retrieve latest benchmark report
POST /optimization/train/pytorch         — train the PyTorch DistilBERT classifier
"""

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import QueryResult

router = APIRouter()


@router.get("/gpu-profile")
async def gpu_profile():
    """Detect GPU and return recommended TensorRT precision + batch size."""
    from app.optimization.gpu_profiler import GPUAutoProfiler
    from dataclasses import asdict

    profiler = GPUAutoProfiler()
    profile = profiler.profile()
    return {
        "name": profile.name,
        "vram_total_gb": profile.vram_total_gb,
        "vram_available_gb": profile.vram_available_gb,
        "compute_capability": list(profile.compute_capability),
        "recommended_precision": profile.recommended_precision,
        "max_batch_size": profile.max_batch_size,
        "reasoning": profile.reasoning,
    }


@router.post("/export/classifier")
async def export_classifier(
    model_path: str = "checkpoints/failure_classifier/best_model.pt",
    output_path: str = "models/failure_classifier.onnx",
):
    """Export the trained PyTorch classifier to ONNX format."""
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    try:
        import torch
        from app.models.failure_classifier import FailureClassifier
        from app.optimization.onnx_export import export_classifier_to_onnx

        model = FailureClassifier(num_classes=6)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        onnx_path = export_classifier_to_onnx(model, output_path=output_path)
        return {"status": "exported", "onnx_path": onnx_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark")
async def run_benchmark():
    """Queue a full precision benchmark (PyTorch vs TensorRT FP32/FP16/INT8)."""
    from app.workers.tasks import run_inference_benchmark
    task = run_inference_benchmark.delay()
    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Benchmark started. Check GET /optimization/benchmark/latest when done.",
    }


@router.get("/benchmark/latest")
async def get_latest_benchmark():
    """Return the most recent benchmark report."""
    report_path = Path("benchmarks/latest_report.json")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No benchmark report found. Run POST /optimization/benchmark first.")
    with open(report_path) as f:
        return json.load(f)


@router.post("/export/optimize")
async def export_and_optimize(
    model_path: str = "checkpoints/failure_classifier/best_model.pt",
    output_dir: str = "models/tensorrt",
):
    """
    End-to-end pipeline: PyTorch checkpoint → ONNX → TensorRT FP32 + FP16 + INT8.

    Steps:
      1. Load trained PyTorch classifier
      2. Export to ONNX (with dynamic batch axis)
      3. Convert to TensorRT engines at all supported precisions
      4. Return paths to generated engines

    GPU auto-profiler is consulted to skip precisions the hardware can't accelerate.
    """
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {model_path}")

    try:
        import torch
        from app.models.failure_classifier import FailureClassifier
        from app.optimization.onnx_export import export_classifier_to_onnx
        from app.optimization.tensorrt_convert import convert_all_precisions, TRT_AVAILABLE
        from app.optimization.gpu_profiler import GPUAutoProfiler

        # Step 1: load model
        model = FailureClassifier(num_classes=6)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

        # Step 2: export to ONNX
        onnx_path = f"{output_dir}/failure_classifier.onnx"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        export_classifier_to_onnx(model, output_path=onnx_path)

        # Step 3: TensorRT conversion (GPU hardware required)
        engines = {}
        trt_available = TRT_AVAILABLE
        if trt_available:
            profile = GPUAutoProfiler().profile()
            engines = convert_all_precisions(onnx_path, output_dir=output_dir)

        return {
            "status": "complete",
            "onnx_path": onnx_path,
            "tensorrt_engines": engines,
            "trt_available": trt_available,
            "gpu": GPUAutoProfiler().profile().name if trt_available else "N/A",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/pytorch")
async def train_pytorch_classifier(
    experiment_id: str,
    output_dir: str = "checkpoints/failure_classifier",
    num_epochs: int = 30,
    db: AsyncSession = Depends(get_db),
):
    """
    Train the PyTorch DistilBERT classifier on labeled query results
    from a completed experiment.

    Queues a Celery task — returns immediately with task_id.
    """
    from app.workers.tasks import train_pytorch_failure_classifier
    task = train_pytorch_failure_classifier.delay(experiment_id, output_dir, num_epochs)
    return {
        "task_id": task.id,
        "status": "queued",
        "message": (
            "PyTorch classifier training started. "
            "Uses DistilBERT encoder + focal loss + label smoothing. "
            f"Training on experiment {experiment_id} query results."
        ),
    }


@router.get("/triton/health")
async def triton_health(url: str = "localhost:8001"):
    """Check if Triton Inference Server is ready."""
    try:
        from app.optimization.triton_client import TritonInferenceClient
        client = TritonInferenceClient(url=url)
        return client.health_check()
    except Exception as e:
        return {"server_ready": False, "error": str(e)}
