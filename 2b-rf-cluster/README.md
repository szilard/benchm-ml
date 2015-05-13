
### Distributed Random Forests 

H2O and Spark can run on multi-node clusters.

#### Setup

Testing on r3.8xlarge instances (32 cores, 250GB RAM). Scripts to generate the results: 
[H2O](1-h2o.R) and [Spark](2-spark.txt). The maximal depth of trees is 20.


#### Results

Running times

 n(M) |  #trees  | #nodes  |  H2O  |  Spark
------|----------|---------|-------|--------
  1   |    10    |    1    |  15   |  30
  1   |    10    |    5    |  30   |  25
  1   |    100   |    1    |  120  |  300
  1   |    100   |    5    |       |  150
  10  |    10    |    1    |  80   |  160
  10  |    10    |    5    |  80   |  70
  10  |    100   |    1    |       |  1900
  10  |    100   |    5    |       |  700


AUC

 n(M) |  #trees  |  H2O  |  Spark
------|----------|-------|--------
  1   |    10    |  73   |   53 
  1   |    100   |  75   |   62
  10  |    10    |  76   |   54
  10  |    100   |       |   62


RAM(GB)

 n(M) |  #trees  |  H2O  |  Spark
------|----------|-------|--------
  1   |    10    |   4   |   70 
  1   |    100   |   5   |   90
  10  |    10    |   9   |   130
  10  |    100   |       |   150

