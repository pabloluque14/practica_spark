import sys
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Practica 4 - Pablo Luque Moreno") \
    .getOrCreate() #uso de spark session para crear dataframes


def set_conf():
    # create Spark context with Spark configuration 
    conf = SparkConf().setAppName("Practica 4 - Pablo Luque Moreno")
    sc = SparkContext.getOrCreate(conf=conf)
    return sc

def read_data(sc):

    # read hearders from HDFS to RDD, and collect it as a list
    headers = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header")
    headers = headers.collect()
    
    # colums that I have to select from the dataset
    myColumns = ['PSSM_central_1_P', 'PSSM_r1_3_H', 'PSSM_r1_2_F', 'PSSM_r2_0_G', 'PSSM_r1_-4_I', 'PSSM_r1_2_H']

    # filter only the inputs from the arff headers file
    columns = [head for head in headers if "@inputs" in head]
    # delete @inputs, spaces and split in to a list
    columsList = columns[0].replace('@inputs', '').replace(' ','').split(',')
    # add class column
    columsList.append('class')

    # read csv file with no heather and activating the infering schema mode
    dataset = spark.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.data", header=False, inferSchema=True)

    # replace  datset colums 
    for c in range(0, len(datdataset.columns)):
        data = dataset.withColumnRenamed(data.columns[c], columsList[c])

    # select only my columns
    df = data.select(myColumns)
    df.write.csv('./smallTrainSet', header=True, mode="overwrite")
    
    return df




    

    








if __name__ == "__main__":

    sc = set_conf()
    df = read_data(sc)


    print(df.head)

    sc.stop()

    """ df = sc.read.csv("/user/ccsaDNI/fichero.csv", header = True, sep = ",", inferSchema = True)
    
    df.show() 
    df.createOrReplaceTempView("sql_dataset")
    sqlDF = sc.sql("SELECT campo1, camp3, ... c6 FROM sql_dataset LIMIT 12") 
    sqlDF.show()
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8) 
    lrModel = lr.fit(sqlDF)
    lrModel.summary() """
    #df.collect() <- NO!