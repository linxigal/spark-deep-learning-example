from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
img_dir = "/home/ourui/deeplearning/images_classification-master/personalities"

#Read images and Create training & test DataFrames for transfer learning
jobs_df = ImageSchema.readImages(img_dir + "/jobs").withColumn("label", lit(1))
zuckerberg_df = ImageSchema.readImages(img_dir + "/zuckerberg").withColumn("label", lit(0))
jobs_train, jobs_test = jobs_df.randomSplit([0.6, 0.4])
zuckerberg_train, zuckerberg_test = zuckerberg_df.randomSplit([0.6, 0.4])
#dataframe for training a classification model
train_df = jobs_train.unionAll(zuckerberg_train)

#dataframe for testing the classification model
test_df = jobs_test.unionAll(zuckerberg_test)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=50, regParam=0.005, elasticNetParam=0.1, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)
print("training model")
predictions = p_model.transform(test_df).select("image", "probability", "label","prediction")
predictionAndLabels = predictions.select("prediction", "label")
predictionAndLabels.show(truncate=False)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
predictions.show()
testzuckerberg_df=ImageSchema.readImages(img_dir + "/test").withColumn("label", lit(0))
predictions1 = p_model.transform(testzuckerberg_df).select("image", "probability", "label","prediction")
predictions1.select("prediction", "label").show(truncate=False)




