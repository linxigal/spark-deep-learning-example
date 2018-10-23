from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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
lr = LogisticRegression(maxIter=50, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
#p_model = p.fit(train_df)

paramGrid = ParamGridBuilder()\
    .addGrid(lr.maxIter, [50,100,500,1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

bc = MulticlassClassificationEvaluator(metricName="accuracy")
cv = CrossValidator(estimatorParamMaps=paramGrid, evaluator=bc, numFolds=2)
cv.setEstimator(p)
p_model = cv.fit(train_df)

print("training model")
predictions = p_model.transform(test_df).select("image", "probability", "label","prediction")
predictionAndLabels = predictions.select("prediction", "label")
predictionAndLabels.show(truncate=False)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
predictions.show()



