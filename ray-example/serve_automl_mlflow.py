import os

import mlflow
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.utils.proto_json_utils import dataframe_from_parsed_json
from pydantic import BaseModel
from ray import serve

api = FastAPI(title="MLflow AutoML Serve")


class PredictRequest(BaseModel):
    instances: list[dict]  # 행마다 {컬럼: 값}. 컬럼은 MLflow 모델의 입력 예시(input_example)와 같다


@serve.deployment
@serve.ingress(api)
class AutomlServe:
    def __init__(self):
        self.model_uri = os.getenv("MLFLOW_MODEL_URI")
        if not self.model_uri:
            raise RuntimeError("MLFLOW_MODEL_URI is required (e.g. runs:/<run_id>/model)")

        # AutoML 모델은 task와 상관없이 같은 pyfunc 형식이라 URI만 바꿔 배포한다
        self.model = mlflow.pyfunc.load_model(self.model_uri)
        self.schema = self.model.metadata.get_input_schema()

    @api.post("/predict")
    def predict(self, req: PredictRequest):
        try:
            # 입력 스키마가 있으면 MLflow 서빙과 같은 규칙으로 형을 맞춘다 (JSON 41 → double 컬럼이면 41.0)
            frame = dataframe_from_parsed_json(req.instances, "records", self.schema)
            predictions = self.model.predict(frame)
        except (MlflowException, ValueError) as e:  # 컬럼 누락, 값 타입 불일치
            raise HTTPException(status_code=400, detail=str(e)) from e
        # 분류는 클래스별 확률 컬럼과 prediction(라벨), 회귀는 prediction, 시계열은 예측 구간 행마다 mean·분위수 컬럼이다
        return {"predictions": predictions.to_dict(orient="records")}


app = AutomlServe.bind()
