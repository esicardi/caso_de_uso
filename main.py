from databricks.connect import DatabricksSession as SparkSession
from databricks.sdk.core import Config
from utils import read_yaml
from datetime import datetime


if __name__ =='__main__':

    # 1) Incializacion de spark
    config =  Config(profile="DEFAULT", cluster_id="0709-232718-y2zn1btt")
    spark = SparkSession.builder.sdkConfig(config).getOrCreate()

    # 2) yaml con todos los parametros
    config=read_yaml()

    # 3) fecha de ejecucion del proceso completo (first execution o daily execution)
    exec_time=datetime.now()
    exec_time=exec_time.strftime('%Y-%m-%dT%H:%M:%S')
    print(f'Fecha de ejecucion: {exec_time}')

    """ 31 de enero >>  First execution """

    #job  model-training

    #job make-predicitions

