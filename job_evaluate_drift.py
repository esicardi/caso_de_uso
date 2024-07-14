from datetime import datetime, timedelta
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score


class EvaluateDrift():
    def __init__(self, spark, config, exec_time, date=None):
        self.spark = spark
        self.exec_time = exec_time

        # restamos un día a la fecha actual
        if date is None:
            date = datetime.now()
        else:
            date = datetime.strptime(date, '%Y-%m-%d')
        date = date - timedelta(days=1)
        self.date = date.strftime('%Y-%m-%d')

        # config
        self.table_daily_predictions = config['table']['daily_predictions']
        self.table_daily_events = config['table']['daily_events']
        self.table_metrics = config['table']['metrics']
        self.model_name = config['model_name']
        self.model_version = -1

    def run(self):

        print(" ** job EVALUATE DRIFT ** ")

        # load y, y_hat
        df_y = (
            self.spark.read.table(self.table_daily_events)
            .where(f"fecha='{self.date}'")
            .toPandas()
        )
        df_y_hat = (
            self.spark.read.table(self.table_daily_predictions)
            .where(f"fecha='{self.date}'")
            .toPandas()
        )
        df = pd.merge(df_y, df_y_hat, on="id_cliente", how="inner")

        # compute score
        y = df.y.to_list()
        y_hat = df.y_hat.to_list()
        score = accuracy_score(y_true=y, y_pred=y_hat)
        print(f"Score del modelo para el día {self.date}: {score:.2f}")

        # tomamos la decision de si queremos reentrenar
        if score < 0.8:
            pass

        # save table metrics
        mlflow.set_tracking_uri("databricks")
        client = MlflowClient()
        self.model_version = int(client.get_registered_model(self.model_name).latest_versions[-1].version)

        metrics = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.date,
            "score": score,
            "exec_time": self.exec_time
        }
        df_metrics = pd.DataFrame([metrics])
        sdf = self.spark.createDataFrame(df_metrics)
        sdf.write.mode("append").saveAsTable(self.table_metrics)