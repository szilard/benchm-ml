
### Spark MLlib Random Forest 

Benchmarking on same [dataset](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD) 
as Databricks used in their random forest [blog post](https://databricks.com/blog/2015/01/21/random-forests-and-boosting-in-mllib.html).

#### Data

The original task is predicting the release year of a song from audio features. I'm making that
a binary classification problem: predicting if release year is >=2004 or not. The script downloading
the data and transforming it to classification is [here](1-data.txt). There are 515K observations and 90 numerical 
predictors, no missing values. I use the same train-test split (463K:51K) as suggested 
on the data repository and also used in the Databricks blog post.

#### Setup

Testing on a r3.8xlarge instance (32 cores, 250GB RAM). Scripts to generate the results: 
[Spark](2-spark.txt) and [H2O](3-h2o.R) for comparison.

Parameters: `numTrees = 10`, `maxDepth = 20`, `maxBins = 50`, `impurity = "entropy"` (same for H2O).

#### Results

             | Spark   | H2O
-------------|---------|-------
Time (sec)   |   160   | 20
max-RAM (GB) |   80    | 10
AUC          |  70.4   | 73.4

10x slower, 10x more memory (that will bite big data), lower AUC (weird)


TODO: 

Study scaling:

1. 1 core vs 10 cores
2. dataset vs 10x dataset
3. 1 node vs 10 nodes (distributed setting)
4. 10 trees vs 100 trees

