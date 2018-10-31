source activate tensorflow 
export set JAVA_OPTS="-Xmx8G -XX:MaxPermSize=2G -XX:+UseCompressedOops -XX:MaxMetaspaceSize=512m"
spark-submit --master local[4]  \
  --executor-memory 16g \
  --driver-memory 16g \
  --packages databricks:spark-deep-learning:1.2.0-spark2.3-s_2.11  newmodelCreator_mybaby.py 
 # --packages databricks:spark-deep-learning:1.2.0-spark2.3-s_2.11  newmodelCreator_iter.py 
