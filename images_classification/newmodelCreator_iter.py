from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import string

img_dir = "/home/ourui/deeplearning/images_classification-master/personalities"

#fo = open("/home/ourui/deeplearning/images_classification-master/train.log", "w")

for maxIter1 in [50,100,500,1000]:
    for regParam1 in [0.005,0.05,0.5]:
        for elasticNetParam1 in [0.1,0.3,0.6,0.8]:
            fs = open("/home/ourui/deeplearning/images_classification-master/train.log", "a+")
            #Read images and Create training & test DataFrames for transfer learning
            jobs_df = ImageSchema.readImages(img_dir + "/jobs").withColumn("label", lit(1))
            zuckerberg_df = ImageSchema.readImages(img_dir + "/zuckerberg").withColumn("label", lit(0))
            jobs_train, jobs_test = jobs_df.randomSplit([0.6, 0.4])
            zuckerberg_train, zuckerberg_test = zuckerberg_df.randomSplit([0.6, 0.4])
            #dataframe for training a classification model
            train_df = jobs_train.unionAll(zuckerberg_train)
            #dataframe for testing the classification model
            test_df = jobs_test.unionAll(zuckerberg_test)
            featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
            lr = LogisticRegression(maxIter=maxIter1, regParam=regParam1, elasticNetParam=elasticNetParam1, labelCol="label")
            p = Pipeline(stages=[featurizer, lr])
            p_model = p.fit(train_df)
            print("training model")
            predictions = p_model.transform(test_df).select("image", "probability", "label","prediction")
            predictionAndLabels = predictions.select("prediction", "label")
            predictionAndLabels.show(truncate=False)
            evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
            value = str(evaluator.evaluate(predictionAndLabels)) 
            print("Training set accuracy = " + value )
            predictions.show()
            log = str(maxIter1)+":"+str(regParam1)+":"+str(elasticNetParam1)+":"+value+"\n"
	    #print("******************" + str(log))
            fs.write(log)
            fs.close		

