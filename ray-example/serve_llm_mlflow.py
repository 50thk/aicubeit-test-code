import os

import mlflow
from ray.serve.llm import LLMConfig, build_openai_app

# 요청 body의 "model" 값
MODEL_ID = "gs-llm-won-instruct"


def checkpoint_s3_uri(model_uri):
    # LLaMA Factory(transformers MLflowCallback)는 checkpoint-N을 빈 pyfunc 모델로 남기고,
    # Hugging Face 모델 폴더(config.json 포함)는 그 모델의 artifacts/checkpoint-N/ 에 있다
    model = mlflow.MlflowClient().get_logged_model(model_uri.removeprefix("models:/"))
    if not model.artifact_location.startswith("s3://"):
        raise RuntimeError(f"S3 artifact location required, got {model.artifact_location}")
    return f"{model.artifact_location.rstrip('/')}/artifacts/{model.name}"


app = build_openai_app({
    "llm_configs": [
        LLMConfig(
            model_loading_config={
                "model_id": MODEL_ID,
                # s3:// 경로는 Replica가 뜨는 노드에서 boto3로 내려받는다 (MinIO 주소는 AWS_ENDPOINT_URL)
                "model_source": checkpoint_s3_uri(os.environ["MLFLOW_MODEL_URI"]),
            },
            engine_kwargs={
                # GPU를 다른 작업과 나눠 쓰면 남은 메모리 비율 이하로 낮춘다
                "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
            },
        ),
    ],
})
