from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient

class ModelTraining():
    def __init__(self, spark, config, exec_time):
        self.spark=spark
        self.exec_time=exec_time

        # config
        self.table_daily_events = config['table']['daily_events']
        self.table_train = config['table']['train']
        self.table_test = config['table']['test']
        self.table_metrics = config['table']['metrics']
        self.model_name = config['model_name']
        self.model_version= -1 # El nro de modelo va a ser dinamico, por eso no va en el yaml
        self.feature_cols = config['modeling']['feature_cols']
        self.test_size = config['modeling']['test_size']
        self.seed = config['modeling']['seed']
    def run(self):
        print("** job MODEL TRAINING **")

        # load training data
        df_events = self.spark.read.table(self.table_daily_events).toPandas()

        # train test split
        X = df_events[self.feature_cols]
        y = df_events['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.seed)

        # model training
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment( f'/Users/estrellasicardi@udelar409.onmicrosoft.com/experiment_{self.model_name}')
        mlflow.sklearn.autolog()
        with mlflow.start_run() as run:
            model=LogisticRegression(random_state=self.seed)
            model.fit(X_train, y_train)
            score=model.score(X_test, y_test)
            mlflow.log_metric("score",score)
            print(f"Score del modelo: {score:.2f}")


            # model registry
            y_pred=model.predict(X_test)
            signature=infer_signature(X_test,y_pred)
            mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path = "sklearn-model",
                signature = signature,
                registered_model_name =  self.model_name,
            )
        client =MlflowClient()
        self.model_version = int(client.get_registered_model(self.model_name).latest_versions[-1].version)
        print(f"Creada la version {self.model_version} del modelo {self.model_name}")

        # save metrics table
        metrics={
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run:id": run.info.run_id,
            "score": score,
            "exec_time": self.exec_time
        }
        df_metrics =  pd.DataFrame([metrics])
        sdf = self.spark.createDataFrame(df_metrics)
        sdf.write.mode("append").saveAsTable(self.table_metrics)

        # save train table

        df_train = pd.merge(X_train, y_train, left_index = True, right_index = True, how = 'inner')
        df_train['model_name'] = self.model_name
        df_train['model_version']=self.model_version
        sdf = self.spark.createDataFrame(df_train)
        sdf.write.mode("append").saveAsTable(self.table_train)

        # save test table

        df_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='inner')
        df_test['model_name'] = self.model_name
        df_test['model_version'] = self.model_version
        sdf = self.spark.createDataFrame(df_test)
        sdf.write.mode("append").saveAsTable(self.table_test)
