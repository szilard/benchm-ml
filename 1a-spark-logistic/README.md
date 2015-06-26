
#### Spark MLlib logistic regression accuracy

AUC for R, Python, VW, H2O very close to each other, AUC for Spark is 
[lower](https://github.com/szilard/benchm-ml#linear-models) (initially Spark 1.3, but also 1.4)

Similar results by [others](https://github.com/BIDData/BIDMach/wiki/Benchmarks#reuters-data)

For `n = 1M`: [train](https://s3.amazonaws.com/benchm-ml--spark/spark-train-1m.csv) and 
[test](https://s3.amazonaws.com/benchm-ml--spark/spark-test-1m.csv) data, 
[R script](glmnet.R), 
[Spark code](spark.txt)

