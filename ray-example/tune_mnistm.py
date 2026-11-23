import os

import mlflow
import mlflow.pytorch
import pandas as pd
import ray
import torch
from huggingface_hub import hf_hub_download
from ray import tune
from ray.tune.search import BasicVariantGenerator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "gs-custom-mnistm")
LR_MIN, LR_MAX = 1e-4, 1e-2


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


def train_trial(cfg, data):
    torch.manual_seed(cfg["seed"])
    # Ray는 GPU를 배정받은 trial에만 CUDA 장치를 노출한다
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"trial start: lr={cfg['lr']:.2e} device={device} threads={torch.get_num_threads()}", flush=True)
    test_ds = to_dataset(*data["test"])
    train_dl = DataLoader(to_dataset(*data["train"]), batch_size=cfg["batch_size"], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"])
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"trial-lr={cfg['lr']:.2e}", parent_run_id=cfg["parent_run_id"]):
        mlflow.log_params({k: cfg[k] for k in ("lr", "batch_size", "epochs", "seed")})

        for epoch in range(cfg["epochs"]):
            model.train()
            loss_sum, n = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
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
                    x, y = x.to(device), y.to(device)
                    cm += torch.bincount(y * 10 + model(x).argmax(1), minlength=100)

            metrics = {"train_loss": loss_sum / n, "test_accuracy": cm.view(10, 10).trace().item() / cm.sum().item()}
            mlflow.log_metrics(metrics, step=epoch + 1)
            tune.report({"epoch": epoch + 1, **metrics})

        log_result(model, cm, test_ds.tensors[0][:1].numpy())


if __name__ == "__main__":
    env = os.environ
    ray.init()
    workers = [n["Resources"] for n in ray.nodes() if n["Alive"] and "node:__internal_head__" not in n["Resources"]]
    # trial 1개가 worker 1대의 CPU·GPU를 모두 차지하므로 동시 실행 trial 수 = worker 수
    resources = {"cpu": min(w["CPU"] for w in workers), "gpu": min(w.get("GPU", 0) for w in workers)}
    print(f"workers={len(workers)} resources_per_trial={resources}", flush=True)
    data = {split: load(split) for split in ("train", "test")}

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="mnistm-tune") as parent:
        num_trials = int(env.get("NUM_TRIALS", "4"))
        space = {
            "lr": tune.loguniform(LR_MIN, LR_MAX),
            "batch_size": int(env.get("BATCH_SIZE", "128")),
            "epochs": int(env.get("EPOCHS", "10")),
            "seed": int(env.get("SEED", "42")),
            "parent_run_id": parent.info.run_id,
        }
        mlflow.log_params({"num_trials": num_trials, "lr_min": LR_MIN, "lr_max": LR_MAX, "num_workers": len(workers),
                           **{k: space[k] for k in ("batch_size", "epochs", "seed")}})
        tuner = tune.Tuner(
            # 데이터는 Head에서 한 번만 받아 object store로 공유한다
            tune.with_resources(tune.with_parameters(train_trial, data=data), resources),
            param_space=space,
            tune_config=tune.TuneConfig(
                metric="test_accuracy",
                mode="max",
                num_samples=num_trials,
                # 같은 SEED면 trial별 학습률도 같게 뽑힌다
                search_alg=BasicVariantGenerator(random_state=space["seed"]),
            ),
        )
        results = tuner.fit()
        best = results.get_best_result()
        mlflow.log_param("best_lr", best.config["lr"])
        mlflow.log_metric("best_test_accuracy", best.metrics["test_accuracy"])
        mlflow.log_text(results.get_dataframe().to_csv(index=False), "tune_results.csv")
        print(f"best: lr={best.config['lr']:.2e} test_accuracy={best.metrics['test_accuracy']:.4f}", flush=True)
