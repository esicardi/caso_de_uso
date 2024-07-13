


if __name__ =='__main__':
    from databricks.connect import DatabricksSession as SparkSession
    from databricks.sdk.core import Config

    config =  Config(profile="DEFAULT", cluster_id="0709-232718-y2zn1btt")
    spark = SparkSession.builder.sdkConfig(config).getOrCreate()

    spark.sql("show databases").show()