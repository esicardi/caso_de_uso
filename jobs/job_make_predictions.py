from datetime import datetime
import pandas as pd
import mlflow
from mlflow import MlflowClient
import time

class MakePredictions():
    def __init__(self, spark, config, exec_time, date=None):
        self.spark = spark
        self.exec_time = exec_time
        self.date = datetime.now().strftime('%Y-%m-%d') if date is None else date

        # config
        self.table_daily_predictions = config['table']['daily_predictions']
        self.table_daily_features = config['table']['daily_features']
        self.model_name = config['model_name']
        self.model_version = -1
        self.feature_cols = config['modeling']['feature_cols']

    def run(self):
        print(" ** job MAKE PREDICTIONS ** ")

        # load training data
        df_daily_features = (
            self.spark.read.table(self.table_daily_features)
            .where(f"fecha='{self.date}'")
            .toPandas()
        )
        clientes = df_daily_features.id_cliente.to_list()
        n = len(clientes)

        # load model (ultima versión productiva)
        mlflow.set_tracking_uri("databricks")
        client = MlflowClient()
        self.model_version = int(client.get_registered_model(self.model_name).latest_versions[-1].version)
        model_uri = f"models:/{self.model_name}/{str(self.model_version)}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Cargamos el modelo {self.model_name} con versión {self.model_version} para hacer las predicciones")

        # make predictions
        y_hat = model.predict(df_daily_features[self.feature_cols])

        # save predictions table
        predictions = {
            "id_cliente": clientes,
            "y_hat": y_hat,
            "fecha": [self.date] * n,
            "model_name": [self.model_name] * n,
            "model_version": [self.model_version] * n,
            "exec_time": [self.exec_time] * n
        }
        df_predictions = pd.DataFrame(predictions)
        sdf = self.spark.createDataFrame(df_predictions)
        sdf.write.mode("append").saveAsTable(self.table_daily_predictions)