from databricks.connect import DatabricksSession as SparkSession
from databricks.sdk.core import Config
from utils import read_yaml

if __name__ =='__main__':


    # 1) Incializacion de spark
    config =  Config(profile="DEFAULT", cluster_id="0709-232718-y2zn1btt")
    spark = SparkSession.builder.sdkConfig(config).getOrCreate()

    # 2) yaml con todos los parametros
    config=read_yaml()