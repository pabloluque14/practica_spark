import sys
import os.path
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

spark = SparkSession \
    .builder \
    .appName("Practica 4 - Pablo Luque Moreno") \
    .getOrCreate() #uso de spark session para crear dataframes


def set_conf():
    # create Spark context with Spark configuration 
    conf = SparkConf().setAppName("Practica 4 - Pablo Luque Moreno")
    sc = SparkContext.getOrCreate(conf=conf)
    return sc

def check_dataset():

    # Check if the dataset file exist
    if (os.path.exists("./smallTrainSet") and os.path.isfile("./smallTrainSet")):      
        df = spark.read.csv("./smallTrainSet", header=True, sep=",", inferSchema=True)
        return df
    else:
        raise Exception("ERROR: Dataset file doesn't found.")


def scale_dataset(df):
    
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

    # Compute summary statistics and generate MinMaxScalerModel
    scalerModel = scaler.fit(df)

    # rescale each feature to range [min, max].
    scaledData = scalerModel.transform(df)

    print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
    scaledData.select("features", "scaledFeatures").show()

    return scaledData




if __name__ == "__main__":
    pass

    sc = set_conf()

    df = check_dataset()

    #sqlc = SQLContext(sc)
    #df = sqlc.read.csv('./smallTrainSet', header=True, sep=',',inferSchema=True)

    assembler = VectorAssembler(inputCols=['PSSM_central_1_P', 'PSSM_r1_3_H', 'PSSM_r1_2_F', 'PSSM_r2_0_G', 'PSSM_r1_-4_I', 'PSSM_r1_2_H'], outputCol='features')
    dataset = assembler.transform(df)
    dataset = dataset.selectExpr('features as features', 'class as label')
    dataset = dataset.select('features', 'label')


    scaledData = scale_dataset(dataset)  
