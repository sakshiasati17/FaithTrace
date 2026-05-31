"""
Convert ONNX models to TensorRT engines at FP32, FP16, or INT8 precision.

TensorRT optimisations over raw PyTorch:
1. Layer fusion — combines Conv+BN+ReLU into one kernel launch.
2. Precision calibration — FP16 is ~2x faster, INT8 ~4x faster.
3. Kernel auto-tuning — picks the fastest kernel for YOUR specific GPU.
4. Memory planning — minimises VRAM footprint.
"""

from pathlib import Path

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_AVAILABLE = False
    TRT_LOGGER = None


def build_tensorrt_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    workspace_size_gb: int = 4,
    calibration_data=None,
) -> str:
    """
    Convert an ONNX model to a TensorRT engine.

    precision choices:
      fp32 — baseline, safe default
      fp16 — ~2x faster, <0.1% accuracy loss, always worth it on Ampere+
      int8 — ~4x faster, requires calibration_data, 0.5-2% accuracy loss
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed. Run: pip install tensorrt")

    builder = trt.Builder(TRT_LOGGER)

    # TensorRT 10 removed EXPLICIT_BATCH (all networks are explicit-batch by default).
    # TensorRT 8/9 required the flag. Support both.
    try:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    except AttributeError:
        network = builder.create_network()

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1 << 30))

    # TRT 10 removed BuilderFlag.FP16 and BuilderFlag.INT8 entirely.
    # TRT 8/9 had these flags. Use hasattr to support both API generations.
    if precision == "fp16":
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print(f"NOTE: TRT {trt.__version__} removed BuilderFlag.FP16 — building FP32 (precision set per-layer in TRT 10+)")
    elif precision == "int8":
        if hasattr(trt.BuilderFlag, "INT8"):
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_data:
                config.int8_calibrator = calibration_data
        else:
            print(f"NOTE: TRT {trt.__version__} removed BuilderFlag.INT8 — use quantization API for INT8 in TRT 10+")

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = list(t.shape)
        profile.set_shape(
            t.name,
            [1] + shape[1:],
            [max_batch_size // 2] + shape[1:],
            [max_batch_size] + shape[1:],
        )
    config.add_optimization_profile(profile)

    print(f"Building TensorRT engine ({precision}) — this may take a few minutes…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    print(f"TensorRT engine saved: {output_path}")
    return output_path


def convert_all_precisions(onnx_path: str, output_dir: str = "models/tensorrt") -> dict:
    """Convert a single ONNX model to FP32, FP16, and INT8 engines."""
    model_name = Path(onnx_path).stem
    engines = {}
    for precision in ["fp32", "fp16", "int8"]:
        out = f"{output_dir}/{model_name}_{precision}.engine"
        try:
            build_tensorrt_engine(onnx_path, out, precision=precision)
            engines[precision] = out
        except Exception as e:
            print(f"Failed to build {precision} engine: {e}")
    return engines
