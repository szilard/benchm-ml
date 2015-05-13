
### Distributed Random Forests 

H2O and Spark can run on multi-node clusters.

#### Setup

Testing on r3.8xlarge instances (32 cores, 250GB RAM). Scripts to generate the results: 
[H2O](1-h2o.R) and [Spark](2-spark.txt). The maximal depth of trees is 20.


#### Results

Running times (sec)

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

[for the numbers missing in the tables above I need to more runs]

The amount of memory that Spark can run on might be smaller, as depending
on settings, garbage collection kicks in or not. Anyway, Spark seems to have a large (10x)
memory footprint. 

However, it seems to scale better to multiple nodes as far as speed is concerned. 
One can therefore use that (many nodes) to alleviate the memory problems. 

However, AUC is low. Finding the reason for the lower AUC would need more investigation
(one reason might be that `predict` for Spark decision trees returns 0/1 and not probability scores therefore
the random forest prediction is based on voting not probability averaging - but that might not be the only 
reason).



