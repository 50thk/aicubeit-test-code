import logging
import os

import mlflow
import torch
from fastapi import Body, FastAPI
from ray import serve


IMAGE_SIZE = 32
NUM_IMAGE_CHANNELS = 3
NUM_CLASSES = 10

api = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray-serve-mnist-m-vit")


def _prepare_tensor(instances):
    tensor = torch.tensor(instances, dtype=torch.float32)

    if tensor.ndim == 1 and tensor.numel() == NUM_IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE:
        tensor = tensor.view(1, NUM_IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    elif tensor.ndim == 2 and tensor.shape[1] == NUM_IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE:
        tensor = tensor.view(-1, NUM_IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    elif tensor.ndim == 3 and tensor.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGE_CHANNELS):
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape == (NUM_IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4 and tensor.shape[1:] == (NUM_IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        pass
    elif tensor.ndim == 4 and tensor.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGE_CHANNELS):
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError("Unsupported input shape for MNIST-M ViT")

    tensor = tensor / 255.0
    return (tensor - 0.5) / 0.5


def get_env_config():
    return {
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "MLFLOW_MODEL_URI": os.getenv("MLFLOW_MODEL_URI"),
    }


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class MnistVitServe:
    def __init__(self):
        config = get_env_config()
        tracking_uri = config["MLFLOW_TRACKING_URI"]
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        model_uri = os.getenv("MLFLOW_MODEL_URI")
        if not model_uri:
            raise RuntimeError("MLFLOW_MODEL_URI is required")

        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()
        logger.info("Loaded MNIST-M ViT model from %s", model_uri)

    @api.post("/predict")
    async def predict(self, payload: dict = Body(...)):
        instances = payload.get("instances")
        if instances is None:
            return {"error": "Missing 'instances' in request body"}

        try:
            inputs = _prepare_tensor(instances)
        except ValueError as exc:
            return {"error": str(exc)}

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities_tensor = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1).tolist()
            probabilities = probabilities_tensor.tolist()
        return {"predictions": predictions, "probabilities": probabilities}


app = MnistVitServe.bind()
