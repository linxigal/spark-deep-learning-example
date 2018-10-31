from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
img_dir = "/home/ourui/deeplearning/images_classification/personalities"

#Read images and Create training & test DataFrames for transfer learning
jobs_df = ImageSchema.readImages(img_dir + "/jobs").withColumn("label", lit(2))
zuckerberg_df = ImageSchema.readImages(img_dir + "/zuckerberg").withColumn("label", lit(1))
yiyi_df = ImageSchema.readImages(img_dir + "/yiyi").withColumn("label", lit(0))
zuckerberg_train, zuckerberg_test = zuckerberg_df.randomSplit([0.6, 0.4])
jobs_train, jobs_test = jobs_df.randomSplit([0.6, 0.4])
yiyi_train, yiyi_test = yiyi_df.randomSplit([0.6, 0.4])

#dataframe for training a classification model
train_df = jobs_train.unionAll(zuckerberg_train).unionAll(yiyi_train)

#dataframe for testing the classification model
test_df = yiyi_test.unionAll(zuckerberg_test).unionAll(jobs_test)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=1000, regParam=0.005, elasticNetParam=0.1, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)
print("training model")
predictions = p_model.transform(test_df).select("image", "probability", "label","prediction")
predictionAndLabels = predictions.select("prediction", "label")
predictionAndLabels.show(1000)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
yiyi=ImageSchema.readImages(img_dir + "/test").withColumn("label", lit(0))
predictions1 = p_model.transform(yiyi).select("image", "probability", "label","prediction")
predictions1.select("prediction", "label").show(truncate=False)

