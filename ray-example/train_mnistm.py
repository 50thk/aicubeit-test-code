import os
from contextlib import nullcontext

import mlflow
import mlflow.pytorch
import pandas as pd
import ray
import ray.train
import ray.train.torch
import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "gs-custom-mnistm")


def load(split):
    # Head(드라이버)에서만 호출. uint8 그대로 두어 공유 크기를 줄인다 (train+test 약 210MB)
    path = hf_hub_download("Yeongkwon/MNIST-M-flatten", f"{split}.csv", repo_type="dataset")
    df = pd.read_csv(path, dtype="uint8")
    x = df.drop(columns="label").to_numpy().reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2).copy()
    return x, df["label"].to_numpy().astype("int64")


def to_dataset(x, y):
    # Worker에서 호출. object store의 읽기 전용 배열을 복사해 float로 변환.
    # 서빙 코드(serve_mnist_vit.py)와 같은 정규화: /255 후 (x-0.5)/0.5 → -1~1
    return TensorDataset((torch.tensor(x).float() / 255 - 0.5) / 0.5, torch.tensor(y))


def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10),
    )


def log_result(model, cm, example):
    # 행: 정답 라벨, 열: 예측 라벨
    table = pd.DataFrame(cm.view(10, 10).cpu().numpy()).rename_axis(index="true", columns="pred")
    mlflow.log_text(table.to_csv(), "confusion_matrix.csv")
    mlflow.pytorch.log_model(model.cpu(), name="model", input_example=example)


def train_loop(cfg):
    torch.manual_seed(cfg["seed"])
    ctx = ray.train.get_context()
    rank0 = ctx.get_world_rank() == 0
    device = ray.train.torch.get_device()
    data = ray.get(cfg["data"])
    test_ds = to_dataset(*data["test"])
    train_dl = ray.train.torch.prepare_data_loader(
        DataLoader(to_dataset(*data["train"]), batch_size=cfg["batch_size"], shuffle=True))
    test_dl = ray.train.torch.prepare_data_loader(DataLoader(test_ds, batch_size=cfg["batch_size"]))
    model = ray.train.torch.prepare_model(build_model())
    print(f"worker {ctx.get_world_rank()}/{ctx.get_world_size()}: device={device} "
          f"train_samples={len(train_dl.sampler)}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    if rank0:
        mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="mnistm-ddp") if rank0 else nullcontext():
        if rank0:
            mlflow.log_params({**{k: cfg[k] for k in ("epochs", "batch_size", "lr", "seed")},
                               "num_workers": ctx.get_world_size()})

        for epoch in range(cfg["epochs"]):
            if ctx.get_world_size() > 1:
                train_dl.sampler.set_epoch(epoch)
            model.train()
            loss_sum, n = 0.0, 0
            for x, y in train_dl:
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(y)
                n += len(y)

            model.eval()
            cm = torch.zeros(100, dtype=torch.long, device=device)
            with torch.no_grad():
                for x, y in test_dl:
                    cm += torch.bincount(y * 10 + model(x).argmax(1), minlength=100)

            # 프로세스별 집계를 합산해 전체 데이터 기준 지표를 만든다
            train_stats = torch.tensor([loss_sum, n], dtype=torch.float64, device=device)
            dist.all_reduce(train_stats)
            dist.all_reduce(cm)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": (train_stats[0] / train_stats[1]).item(),
                "test_accuracy": cm.view(10, 10).trace().item() / cm.sum().item(),
            }
            if rank0:
                print(metrics, flush=True)
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "epoch"}, step=epoch + 1)
            ray.train.report(metrics)

        if rank0:
            log_result(getattr(model, "module", model), cm, test_ds.tensors[0][:1].numpy())


if __name__ == "__main__":
    env = os.environ
    cfg = {
        "epochs": int(env.get("EPOCHS", "20")),
        "batch_size": int(env.get("BATCH_SIZE", "128")),
        "lr": float(env.get("LR", "0.001")),
        "seed": int(env.get("SEED", "42")),
    }
    ray.init()
    # Worker 노드에 GPU가 할당되어 있으면 학습 프로세스마다 GPU 1개를 쓴다 (Head는 GPU 0)
    use_gpu = ray.cluster_resources().get("GPU", 0) > 0
    scaling = ScalingConfig(
        num_workers=int(env.get("NUM_WORKERS", "2")),
        use_gpu=use_gpu,
        resources_per_worker={"CPU": int(env.get("CPU_PER_WORKER", "6"))},
    )
    print(f"num_workers={scaling.num_workers} use_gpu={use_gpu}", flush=True)
    # 데이터는 Head에서 한 번만 받아 object store로 공유하고, Worker에는 참조만 넘긴다
    cfg["data"] = ray.put({split: load(split) for split in ("train", "test")})
    TorchTrainer(train_loop, train_loop_config=cfg, scaling_config=scaling).fit()
