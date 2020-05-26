import sys
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression


def set_conf():
    # create Spark context with Spark configuration 
    conf = SparkConf().setAppName("Practica 4 - Pablo Luque Moreno")
    sc = SparkContext(conf=conf)
    return sc

def read_data(sc):

    # read hearders from HDFS to RDD, and collect it as a list
    headers = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header")
    headers = headers.collect()
    print(headers)





    

    








if __name__ == "__main__":

    sc = set_conf()
    read_data(sc)


    """ df = sc.read.csv("/user/ccsaDNI/fichero.csv", header = True, sep = ",", inferSchema = True)
    
    df.show() 
    df.createOrReplaceTempView("sql_dataset")
    sqlDF = sc.sql("SELECT campo1, camp3, ... c6 FROM sql_dataset LIMIT 12") 
    sqlDF.show()
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8) 
    lrModel = lr.fit(sqlDF)
    lrModel.summary() """
    #df.collect() <- NO!