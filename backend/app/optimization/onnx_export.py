"""
Export PyTorch models to ONNX format.

ONNX is the intermediate format that bridges PyTorch → TensorRT.
We trace the model with dummy inputs and freeze the compute graph.
Dynamic axes allow variable batch sizes at runtime.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_classifier_to_onnx(
    model,
    output_path: str = "models/failure_classifier.onnx",
    max_length: int = 512,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    dummy_input_ids = torch.randint(0, 30522, (1, max_length)).to(device)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long).to(device)
    dummy_ragas_scores = torch.rand(1, 5).to(device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_ragas_scores),
        output_path,
        opset_version=17,
        input_names=["input_ids", "attention_mask", "ragas_scores"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "ragas_scores": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and validated: {output_path}")
    return output_path


def export_embedding_model_to_onnx(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_path: str = "models/embedding_model.onnx",
    max_length: int = 256,
) -> str:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    dummy_input = tokenizer(
        "Sample sentence for ONNX tracing",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        opset_version=17,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"Embedding model exported: {output_path}")
    return output_path


def verify_onnx_output(pytorch_model, onnx_path: str, dummy_inputs, tolerance: float = 1e-5):
    """Verify ONNX model produces numerically equivalent outputs to PyTorch."""
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(*dummy_inputs).cpu().numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_inputs = {
        name: inp.cpu().numpy()
        for name, inp in zip([i.name for i in session.get_inputs()], dummy_inputs)
    }
    onnx_output = session.run(None, onnx_inputs)[0]

    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"Max output difference: {max_diff:.8f}")
    assert max_diff < tolerance, f"ONNX output differs by {max_diff} (tolerance: {tolerance})"
    print("ONNX verification passed")
