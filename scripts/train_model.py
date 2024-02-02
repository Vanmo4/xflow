# Импорт библиотек
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient


# Установка URI для отслеживания экспериментов в MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

# Чтение данных из CSV-файла в объект DataFrame
df = pd.read_csv('/home/ivan/Git/MLOps_HW_3S/xflow/datasets/data_train.csv', header=None)
df.columns = ['id', 'counts']
model = LinearRegression()

# Работа с MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/ivan/Git/MLOps_HW_3S/xflow/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()

model.fit(df['id'].values.reshape(-1, 1), df['counts'])

# Сохранение модели
with open('/home/ivan/Git/MLOps_HW_3S/xflow/models/data.pickle', 'wb') as f:
    pickle.dump(model, f)
