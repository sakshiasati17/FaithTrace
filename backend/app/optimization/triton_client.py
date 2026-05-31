"""
Triton Inference Server gRPC client.

Replaces direct PyTorch model.forward() calls with network requests to
NVIDIA Triton, enabling:
- Dynamic batching: Triton groups requests into GPU batches automatically.
- Model versioning: serve multiple versions, A/B test without downtime.
- Multi-model serving: embed + classify + rerank from one GPU server.
- Built-in Prometheus metrics.
"""

import numpy as np


class TritonInferenceClient:
    def __init__(self, url: str = "localhost:8001"):
        try:
            import tritonclient.grpc as grpcclient
            self._client = grpcclient.InferenceServerClient(url=url)
            self._grpc = grpcclient
        except ImportError:
            raise RuntimeError("tritonclient not installed: pip install tritonclient[grpc]")
        self.model_name = "failure_classifier"

    def classify(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        ragas_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Send a batch to Triton and return logits.

        Args:
            input_ids:      (B, 512) int32
            attention_mask: (B, 512) int32
            ragas_scores:   (B, 5)   float32
        Returns:
            logits: (B, 6) float32
        """
        grpc = self._grpc
        inputs = [
            grpc.InferInput("input_ids", input_ids.shape, "INT32"),
            grpc.InferInput("attention_mask", attention_mask.shape, "INT32"),
            grpc.InferInput("ragas_scores", ragas_scores.shape, "FP32"),
        ]
        inputs[0].set_data_from_numpy(input_ids.astype(np.int32))
        inputs[1].set_data_from_numpy(attention_mask.astype(np.int32))
        inputs[2].set_data_from_numpy(ragas_scores.astype(np.float32))

        outputs = [grpc.InferRequestedOutput("logits")]
        result = self._client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )
        return result.as_numpy("logits")

    def health_check(self) -> dict:
        return {
            "server_ready": self._client.is_server_ready(),
            "model_ready": self._client.is_model_ready(self.model_name),
            "model_name": self.model_name,
        }
