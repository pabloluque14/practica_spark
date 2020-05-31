import sys
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

''' spark = SparkSession \
    .builder \
    .appName("Practica 4 - Pablo Luque Moreno") \
    .getOrCreate() # we can use spark session to create dataframes '''


def set_conf():
    # create Spark context with Spark configuration 
    conf = SparkConf().setAppName("Practica 4 - Pablo Luque Moreno")
    sc = SparkContext.getOrCreate(conf = conf)
    return sc

def read_data(sc):

    # read hearders from HDFS to RDD, and collect it as a list
    headers = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header")
    headers = headers.collect()
    
    # colums that I have to select from the dataset
    myColumns = ['PSSM_central_1_P', 'PSSM_r1_3_H', 'PSSM_r1_2_F', 'PSSM_r2_0_G', 'PSSM_r1_-4_I', 'PSSM_r1_2_H', 'class']

    # filter only the inputs from the arff headers file
    columns = [head for head in headers if "@inputs" in head]
    # delete @inputs, spaces and split in to a list
    columsList = columns[0].replace('@inputs', '').replace(' ','').split(',')
    # add class column
    columsList.append('class')

    # read csv file with no heather and activating the infering schema mode
    #dataset = spark.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.data", header=False, inferSchema=True)
    
    sqlc = SQLContext(sc)
    dataset = sqlc.read.csv('/user/datasets/ecbdl14/ECBDL14_IR2.data', header=False, inferSchema=True)

    cols = dataset.columns


    # replace  datset colums 
    for c in range(0, len(dataset.columns)):
        dataset = dataset.withColumnRenamed(cols[c], columsList[c])

    # select only my columns
    df = dataset.select(myColumns)
    df.write.csv('./smallTrainSet', header=True, mode="overwrite")
    
    return df


if __name__ == "__main__":

    sc = set_conf()
    df = read_data(sc)
    #print(df.head)

    #shutdown spark context
    sc.stop()