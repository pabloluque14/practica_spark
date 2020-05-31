import sys
import time
import os.path
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator



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
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
        sc._jsc.hadoopConfiguration())
    
    if fs.exists(sc._jvm.org.apache.hadoop.fs.Path("./smallTrainSet")):
        
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

    #check we normalized the data between 0 - 1
    print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
    scaledData.select("features", "scaledFeatures").show()

    return scaledData



def logistic_regression(df):

    first=time.time()
    # Split the data into training and test sets (30% held out for testing)
    (train, test) = df.randomSplit([0.7, 0.3], seed = 31)

    lr = LogisticRegression(featuresCol="features", labelCol="label", standardization=False)


    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
        .addGrid(lr.maxIter, [5, 10, 15, 20]) \
        .build()


    # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    tvs = TrainValidationSplit(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=BinaryClassificationEvaluator(),
                        # 80% of the data will be used for training, 20% for validation.
                        trainRatio=0.8,
                        seed=31)
    
    # Run TrainValidationSplit, and choose the best set of parameters.
    model = tvs.fit(train)

    # Make predictions on test data. model is the model with combination of parameters
    # that performed best.
    model.transform(test)\
    .select("features", "label", "prediction")\
    .show()

    #bestModel = model.bestModel
    
    predictions = model.transform(test)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator()
    areaUnderROC = evaluator.evaluate(predictions)
    print("Area under ROC, best result:", areaUnderROC)

    print("Hiperparameters")
    for i, item in enumerate(model.getEstimatorParamMaps()):
        gridResult = ["%s: %s" % (parameter.name, str(value)) for parameter, value in item.items()]
        print(gridResult, model.getEvaluator().getMetricName(), model.validationMetrics[i])

    print("TIME: ",(time.time()-first))    




def mlp(df):

    first=time.time()
    # Split the data into training and test sets (30% held out for testing)
    (train, test) = df.randomSplit([0.7, 0.3], seed = 31)

    # specify layers for the neural network:
    # input layer of size 6 (features), two intermediate of size 5 and 4
    # and output of size 2 (classes)
    layers = [6, 5, 4, 2]
    mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=31)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # TrainValidationSplit will try all combinations of values and determine best model using
    # the evaluator.
    paramGrid = ParamGridBuilder()\
    .addGrid(mlp.blockSize, [32, 64, 128])\
    .build()

    # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    tvs = TrainValidationSplit(estimator=mlp,
                        estimatorParamMaps=paramGrid,
                        evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction'),
                        # 80% of the data will be used for training, 20% for validation.
                        trainRatio=0.8,
                        seed=31)

    # Run TrainValidationSplit, and choose the best set of parameters.
    model = tvs.fit(train)

    # Make predictions on test data. model is the model with combination of parameters
    # that performed best.
    model.transform(test)\
    .select("features", "label", "prediction")\
    .show()

    bestModel = model.bestModel
    
    predictions = bestModel.transform(test)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction')
    areaUnderROC = evaluator.evaluate(predictions)
    print("Area under ROC:", areaUnderROC)

    # show best hiperparameters from the best model
    #print("Best hiperparameters")
    #print(bestModel.explainParam('blockSize'))
    #print(bestModel.explainParam())

    print("Hiperparameters")
    for i, item in enumerate(model.getEstimatorParamMaps()):
        gridResult = ["%s: %s" % (parameter.name, str(value)) for parameter, value in item.items()]
        print(gridResult, model.getEvaluator().getMetricName(), model.validationMetrics[i])



    print("TIME: ",(time.time()-first)) 



def random_forest(df):

    first=time.time()

    # Split the data into training and test sets (30% held out for testing)
    (train, test) = df.randomSplit([0.7, 0.3], seed = 31)

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # TrainValidationSplit will try all combinations of values and determine best model using
    # the evaluator.
    paramGrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [10, 15, 20]) \
    .addGrid(rf.impurity, ['entropy', 'gini'])\
    .build()
    

    # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    tvs = TrainValidationSplit(estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=BinaryClassificationEvaluator(),
                        trainRatio=0.8,
                        seed=31)

    # Run TrainValidationSplit, and choose the best set of parameters.
    model = tvs.fit(train)

    # Make predictions on test data. model is the model with combination of parameters
    # that performed best.
    model.transform(test)\
    .select("features", "label", "prediction")\
    .show()

    bestModel = model.bestModel
    
    predictions = model.transform(test)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator()
    areaUnderROC = evaluator.evaluate(predictions)
    print("Area under ROC:", areaUnderROC)

    """ evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
    areaUnderPR = evaluator.evaluate(predictions)
    print("Area under PR:", areaUnderPR) """


    # show best hiperparameters from the best model
    print("Best hiperparameters")
    print("Num of Trees: ", bestModel._java_obj.getNumTrees())
    print("Impurity: ", bestModel._java_obj.getImpurity())

    print("Hiperparameters")
    for i, item in enumerate(model.getEstimatorParamMaps()):
        gridResult = ["%s: %s" % (parameter.name, str(value)) for parameter, value in item.items()]
        print(gridResult, model.getEvaluator().getMetricName(), model.validationMetrics[i])


    #bestModel.save("rf_model")
    print("TIME: ",(time.time()-first)) 

if __name__ == "__main__":
    pass

    sc = set_conf()

    #df = spark.read.csv("./smallTrainSet", header=True, sep=",", inferSchema=True)

    df = check_dataset()

    #sqlc = SQLContext(sc)
    #df = sqlc.read.csv('./smallTrainSet', header=True, sep=',',inferSchema=True)

    # split the datset by class
    positive = df.filter(df["class"] == 1.0)
    negative = df.filter(df["class"] == 0.0)

    # take the ratio
    ratio = float(positive.count()) / float(negative.count())
    # random undersampling
    negative = negative.sample(withReplacement = False, fraction = ratio, seed = 31)
    # merge the two dataframes
    df = negative.union(positive)

    # BEFORE ('Positivos: ', 687729, 'Negativos: ', 1375458)
    # AFTER  ('Positives: ', 687729, 'Negatives: ', 687827)
    print("Positives: ",df.filter(df["class"] == 1.0).count(),"Negatives: ",df.filter(df["class"] == 0.0).count())

    # merge our features columns in to a single column
    assembler = VectorAssembler(inputCols=['PSSM_central_1_P', 'PSSM_r1_3_H', 'PSSM_r1_2_F', 'PSSM_r2_0_G', 'PSSM_r1_-4_I', 'PSSM_r1_2_H'], outputCol='features')
    dataset = assembler.transform(df)
    dataset = dataset.selectExpr('features as features', 'class as label')
    dataset = dataset.select('features', 'label')

    scaledData = scale_dataset(dataset)


    #scaledData.write.csv('./smallTrainSet_transformed', header=True, mode="overwrite")

    # Techniques to apply
    random_forest(scaledData)
    #mlp(scaledData)
    #logistic_regression(scaledData)   



    sc.stop()


   