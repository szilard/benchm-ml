
### Simple/limited/incomplete benchmark for scalability/speed and accuracy of machine learning libraries for classification

This project aims at a *minimal* benchmark for scalability, speed and accuracy of commonly used implementations
of a few machine learning algorithms. The target of this study is binary classification with numeric and categorical inputs (of 
limited cardinality i.e. not very sparse) and no missing data. If the input matrix is of *n* x *p*, *n* is 
varied as 10K, 100K, 1M, 10M, while *p* is about 1K (after expanding the categoricals into dummy 
variables/one-hot encoding). This particular type of data type/size (the largest) stems from this author's interest in 
credit card fraud detection at work.

The algorithms studied are 
- linear (logistic regression, linear SVM)
- random forest
- boosting 
- deep neural network

in various commonly used open source implementations like 
- R packages
- Python scikit-learn
- Vowpal Wabbit
- H2O 
- Spark MLlib.

Random forest, boosting and more recently deep neural networks are the algos expected to perform the best on the structure/sizes
described above (e.g. vs alternatives such as *k*-nearest neighbors, naive-Bayes, decision trees, linear models etc). 
Non-linear SVMs are also among the best in accuracy in general, but become slow/cannot scale for the larger *n*
sizes we want to deal with. The linear models are less accurate in general and are used here only 
as a baseline (but they can scale better and some of them can deal with very sparse features). 

By scalability we mean here that the algos are able to complete (in decent time) for the given data sizes with 
the main contraint being RAM (a given algo/implementation will crash if running out of memory). Some 
of the algos/implementations can work in a distributed setting, although the largest dataset in this
study *n* = 10M is less than 1GB, so scaling out to multiple machines should not be necessary and
is not the focus of this current study. (Also, some of the algos perform relatively poorly speedwise in the multi-node setting, where 
communication is over the network rather than via updating shared memory.)
Speed (in the single node setting) is determined by computational
complexity but also if the algo/implementation can use multiple processor cores.
Accuracy is measured by AUC. The interpretability of models is not of concern in this project.

In summary, we are focusing on which algos/implementations can be used to train relatively accurate binary classifiers for data
with millions of observations and thousands of features processed on commodity hardware (mainly one machine with decent RAM and several cores).

### Data

Training datasets of sizes 10K, 100K, 1M, 10M are [generated](0-init/2-gendata.txt) from the well-known airline dataset (using years 2005 and 2006). 
A test set of size 100K is generated from the same (using year 2007). The task is to predict whether a flight will
be delayed by more than 15 minutes. While we study primarily the scalability of algos/implementations, it is also interesting
to see how much more information and consequently accuracy the same model can obtain with more data (more observations).

### Setup 

The tests have been carried out on a Amazon EC2 c3.8xlarge instance (32 cores, 60GB RAM). The tools are freely available and 
their [installation](0-init/1-install.txt) is trivial (the link also has the version information for each tool). For some
of the models that ran out of memory for the larger data sizes a r3.8xlarge instance (32 cores, 250GB RAM) has been used
occasionally.

As a first step, the models have been trained with default parameters. As a next step we should do search in the hyper-parameter
space with cross validation (that will require more work and way more running time).

### Results

For each algo/tool and each size *n* we observe the following: training time, maximum memory usage during training, CPU usage on the cores, 
and AUC as a measure for predictive accuracy. 
Times to read the data, pre-process the data, score the test data are also observed but not
reported (not the bottleneck).

#### Linear Models

The linear models are not the primary focus of this study because of their not so great accuracy vs
the more complex models (on this type of data). 
They are analysed here only to get some sort of baseline.

The R glm package (the basic R tool for logistic regression) is very slow, 500 seconds on *n* = 0.1M (AUC 70.6).
Therefore, for R the glmnet package is used. For Python/scikit-learn LogisticRegression
(based on the LIBLINEAR C++ library) has been used.

Tool    | *n*  |   Time (sec)  | RAM (GB) | AUC
--------|------|---------------|----------|--------
R       | 10K  |      0.1      |   1      | 66.7
        | 100K |      0.5      |   1      | 70.3
        | 1M   |      5        |   1      | 71.1
        | 10M  |      90       |   5      | 71.1
Python  | 10K  |      0.2      |   2      | 67.6
        | 100K |       2       |   3      | 70.6
        | 1M   |       25      |   12     | 71.1
        | 10M  |  crash/360    |          | 71.1
VW      | 10K  |     0.3 (/10) |          | 66.6
        | 100K |      3 (/10)  |          | 70.3
        | 1M   |      10 (/10) |          | 71.0
        | 10M  |     15        |          | 71.0
H2O     | 10K  |      1        |   1      | 69.6
        | 100K |      1        |   1      | 70.3
        | 1M   |      2        |   2      | 70.8
        | 10M  |      5        |   3      | 71.0
Spark   | 10K  |      2        |   10     | 66.2
        | 100K |      4        |   12     | 69.7
        | 1M   |      5        |   20     | 70.3
        | 10M  |      15       |   20     | 70.4

Python crashes on the 60GB machine, but completes
when RAM is increased to 250GB (using a sparse format would help with memory footprint
and likely runtime as well).
The Vowpal Wabbit (VW) running times are reported in the table for 10 passes (online learning) 
over the data for 
the smaller sizes. While VW can be run on multiple cores, it has been run here in 
the simplest possible way (1 core). Also keep in mind that VW reads the data on the fly while for the other tools
the times reported exclude reading the data into memory.

One can play with various parameters (such as regularization) and even do some search in the parameter space with
cross-validation to get better accuracy. However, very quick experimentation shows that at least for the larger
sizes regularization does not increase accuracy significantly (which is expected since *n* >> *p*).

![plot-time](1-linear/x-plot-time.png)
![plot-auc](1-linear/x-plot-auc.png)

The main conclusion here is that is is trivial to train linear models even for *n* = 10M rows virtually in
any of these tools on a single machine in a matter of seconds. 
H2O and VW are the most memory efficient (VW needs only 1 observation in memory
at a time therefore is the ultimately scalable solution). H2O and VW are also the fastest (for VW the time reported
includes the time to read the data as it is read on the fly).
With respect to accuracy, for some reason Spark's algo is significantly less accurate
than the other ones (H2O's outlying accuracy for *n* = 0.01M
is due to adding regularization automatically and should not be taken into
consideration). Again, the differences in memory efficiency and speed will start to really matter only for
larger sizes and beyond the scope of this study. 

Note that the linear models' accuracy increases only a little from 100K to 1M and it is virtually 
the same for 1M and 10M. This is because the simple linear structure can be extracted already from 
a smaller dataset and having more data points will not change the classification boundary significantly.
On the other hand, more complex models such as random forests can further improve with increasing 
data size by adjusting further the classification boundary. 
However, one needs to pay a price in increasing computational time for these more complex
models, for example if using H2O (random forest results from next section):

n     |  Time linear  | Time RF     | AUC linear |  AUC RF
------|---------------|-------------|------------|--------------
1M    |        2      |    600      |   70.8     |   75.5
10M   |        5      |    4000     |   71.0     |   77.8

<a name="rf-vs-linear"></a>Nevertheless, the main subject of this study are these more complex models that can
achieve higher accuracy than the simple linear models:

![plot-auc](1-linear/z-auc-lin-rf.png)

An interesting thing to note is that the AUC for random forest trained on 100K observations is better
than the AUC on a linear model trained on 10M observations (so "more data or better algorithms?" - it depends).


#### Random Forest

*Note:* The random forests results have been published in a more organized and self-contained form
in [this blog post](http://datascience.la/benchmarking-random-forest-implementations/).

Random forests with 500 trees have been trained in each tool choosing the default of square root of *p* as the number of
variables to split on.

Tool    | *n*  |   Time (sec)  | RAM (GB) | AUC
-------------------------|------|---------------|----------|--------
R       | 10K  |      50       |   10     | 68.2
        | 100K |     1200      |   35     | 71.2
        | 1M   |     crash     |          |
Python  | 10K  |      2        |   2      | 68.4
        | 100K |     50        |   5      | 71.4
        | 1M   |     900       |   20     | 73.2
        | 10M  |     crash     |          |
H2O     | 10K  |      15       |   2      | 69.8
        | 100K |      150      |   4      | 72.5
        | 1M   |      600      |    5     | 75.5
        | 10M  |     4000      |   25     | 77.8
Spark   | 10K  |      50       |   10     | 63.9
        | 100K |      300      |   30     | 65.1
        | 1M   |  crash/2000   |          | 65.9
xgboost | 10K  |     4         |    1     | 69.9
        | 100K |    20         |    1     | 73.2
        | 1M   |    170        |    2     | 75.3
        | 10M  |    3000       |    9     | 76.3

![plot-time](2-rf/x-plot-time.png)
![plot-auc](2-rf/x-plot-auc.png)

The [R](2-rf/1.R) implementation (randomForest package) is slow and inefficient in memory use. 
It cannot cope by default with a large number of categories, therefore the data had
to be one-hot encoded. The implementation uses 1 processor core, but with 2 lines of extra code
it is easy to build
the trees in parallel using all the cores and combine them at the end. However, it runs out
of memory already for *n* = 1M. I have to emphasize this has nothing to do with R per se (and I still stand by
arguing R is the best data science platform esp. when it comes to data munging and
visualization), it is just this
particular (C and Fortran) RF implementation used by the randomForest package that is inefficient.

The [Python](2-rf/2.py) (scikit-learn) implementation is faster, more memory efficient and uses all the cores.
Variables needed to be one-hot encoded (which is more involved than for R) 
and for *n* = 10M doing this exhausted all the memory. Even if using a larger machine
with 250GB of memory (and 140GB free for RF after transforming all the data) the Python implementation
runs out of memory and crashes for this larger size. The algo 
[finished successfully](https://github.com/szilard/benchm-ml/issues/1) 
though when run on the larger box with simple integer encoding (which
for some datasets/cases might be actually a good approximation/choice).

The [H2O](2-rf/4-h2o.R) implementation is fast, memory efficient and uses all cores. It deals
with categorical variables automatically. It is also more accurate than R/Python, which may be because
of dealing properly with the categorical variables, i.e. internally in the algo
rather than working from a previously 1-hot encoded dataset (where the link between the dummies 
belonging to the same original variable is lost).

[Spark](2-rf/5b-spark.txt) (MLlib) implementation is somewhat slower, provides the lowest accuracy and 
it [crashes](2-rf/5c-spark-crash.txt) already at *n* = 1M due to inefficient memory handling. 
With 250G of RAM it finishes for *n* = 1M, but runs out of memory for *n* = 10M. However, as Spark
can run on a cluster one can throw in even more RAM by using more nodes (see some results linked below).
Alternatively, on a single machine, it is possible to train random forests with a smaller number of trees 
(but then accuracy decreases).
I also tried to provide the categorical
variables encoded simply as integers and passing the `categoricalFeaturesInfo` parameter, but that made
training much slower.
A convenience issue, reading the data is more than one line of code and Spark does not provide a one-hot encoder
for the categorical data (therefore I used R for that). 
Note again the low prediction accuracy vs the other methods. One can improve a bit by increasing
the maximum depth of trees (but only to Spark's limit of 30), but then training slows down further and AUC is
still lower than with the other methods. Finding the reason for the lower AUC would need more investigation
(the reason might be that `predict` for Spark decision trees returns 0/1 and not probability scores therefore
the random forest prediction is based on voting not probability averaging, or different
stopping criteria, or just an algorithm that uses some approximations that hurts accuracy).
Finally, the results of some quick experiments with Spark on a cluster of 5 nodes can be found 
[here](2b-rf-cluster). 

In addition to the above, several other random forest implementations have been tested 
(Weka, Revo ScaleR, Rborist R package, Mahout), 
but all of them proved slow and/or unable to scale to the larger sizes.

I also tried [xgboost](2-rf/6-xgboost.R), a popular library for boosting
which is capable to build random forests as well.  
It is fast, memory efficient and of high accuracy. Note the different shapes of the
AUC and runtime vs dataset size curves for H2O and xgboost, some discussions 
[here](https://github.com/szilard/benchm-ml/issues/14).

It would be nice to study the dependence of running time and accuracy as a function of
the (hyper)parameter values of the algorithm, but a quick idea can be obtained easily for the
H2O implementation from this table (*n* = 10M on 250GB RAM):

ntree    | depth  |   nbins  | mtries  | Time (hrs)   |  AUC
---------|--------|----------|---------|--------------|--------
500      |  20    |    20    | -1 (2)  |      1.2     |  77.8 
500      |  50    |    200   | -1 (2)  |      4.5     |  78.9
500      |  50    |    200   |   3     |      5.5     |  78.9
5000     |  50    |    200   | -1 (2)  |      45      |  79.0
500      |  100   |   1000   | -1 (2)  |      8.3     |  80.1

other hyperparameters being sample rate (at each tree), min number of observations in nodes, impurity
function.

One can see that the AUC could be improved further and the best AUC from this dataset with random forests
seems to be around 80 (the best AUC from linear models seems to be around 71, and we will compare
with boosting and deep learning later).



#### Boosting (Gradient Boosted Trees/Gradient Boosting Machines)

Compared to random forests, GBMs have a more complex relationship between hyperparameters
and accuracy (and also runtime). The main hyperparameters are learning (shrinkage) rate, number of trees, 
max depth of trees, while some others being number of bins, sample rate (at each tree), min number of 
observations in nodes. To add to complexity, GBMs can overfit in the sense that adding more trees at some point will
result in decreasing accuracy on a test set (while on the training set accuracy keeps increasing).

For example using xgboost for `n = 100K` `learn_rate = 0.01` `max_depth = 16` (and the
`printEveryN = 100` and `eval_metric = 'auc'` options) the AUC on the train and test sets,
respectively after `n` number of iterations (trees) are:

n     |  AUC train  |  AUC test
------|-------------|-----------
1     |   73.1      |   69.2
100   |   82.9      |   72.9
200   |   85.3      |   73.4
500   |   89.4      |   73.9
1000  |   92.9      |   74.0
2000  |   96.7      |   73.8
3000  |   98.5      |   73.5
4000  |   99.3      |   73.3

One can see the AUC on the test set decreases after 1000 iterations (overfitting). 
xgboost has a handy early stopping option (`early_stop_round = k`: training
will stop if performance keeps getting worse consecutively for `k` rounds).

Doing an extensive search for the best model is not the main goal of this project.
Nevertheless, a quick exploratory search in the hyperparameter space has been
conducted using xgboost (with the early stopping option).



#### Deep neural networks

...

### Conclusions

...

