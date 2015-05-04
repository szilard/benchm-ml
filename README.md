
### Simple/limited/incomplete benchmark for scalability/speed and accuracy of machine learning libraries for classification

This project aims at a *minimal* benchmark for scalability, speed and accuracy of commonly used implementations
of a few machine learning algorithms. The target of this study is binary classification with numeric and categorical inputs (of 
limited cardinality i.e. not very sparse) and no missing data. If the input matrix is of *n* x *p*, *n* is 
varied as 10K, 100K, 1M, 10M, while *p* is about 1K (after expanding the categoricals into dummy 
variables/one-hot encoding). This particular type of data type/size stems from this author's interest in 
credit card fraud.

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

By scalability we mean here that the algos are able to complete (in decent time) for the given *n* sizes. 
The main contraint is RAM (a given algo/implementation can crash if running out of memory), but some 
of the algos/implementations can work in a distributed setting (although the largest dataset in this
study *n* = 10M is less than 1GB, so scaling out to multiple machines should not be necessary and
is not the target of this current study). Speed is determined by computational
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
They are analysed here only to get an idea in terms of scalability/speed/accuracy.

The R glm package (the basic R tool for logistic regression) is very slow, 500 seconds on *n* = 0.1M (AUC 70.6).
The R glmnet package is faster, 10 sec for *n* = 0.1M (AUC 70.4 with no regularization) or 
120 seconds for *n* = 1M (AUC 71.1), but that is still slow compared to the next alternatives. 
The R LiblineaR package is faster, 30 seconds for *n* = 1M and it's presented in more details below, along with
Python/scikit-learn's logistic regression based on the same LIBLINEAR C++ library.

Tool    | *n*  |   Time (sec)  | RAM (GB) | AUC
--------|------|---------------|----------|--------
R       | 10K  |      0.3      |   2      | 67.4
        | 100K |       3       |   3      | 70.6
        | 1M   |      30       |   12     | 71.1
        | 10M  |     crash     |          |
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
        | 1M   |      10       |   20     | 70.3
        | 10M  |   crash/70    |          | 70.4

For the largest size of *n* = 10M, the R package crashes, while Python anf Spark crash on the 60GB machine, but complete
when RAM is increased to 250GB. The Vowpal Wabbit (VW) running times are reported in the table for 10 passes (online learning) 
over the data for 
the smaller sizes. While VW can be run on multiple cores, it has been run here in 
the simplest possible way (1 core).

One can play with various parameters (such as regularization) and even do some search in the parameter space with
cross-validation to get better accuracy. However, very quick experimentation shows that at least for the larger
sizes regularization does not increase accuracy significantly (which is expected since *n* >> *p*).

![plot-time](1-linear/x-plot-time.png)
![plot-auc](1-linear/x-plot-auc.png)

The main conclusion here is that is is trivial to train linear models even for *n* = 10M rows virtually in
any of these tools on a single machine. H2O and VW are the most memory efficient (VW needs only 1 observation in memory
at a time therefore is the ultimately scalable solution). H2O and VW are also the fastest.
H2O, VW and the Python implementation seems to be the most accurate (H2O's outlying accuracy for *n* = 0.01M
is due to adding regularization automatically and should not be taken into
consideration). In fact, the differences in memory efficiency and speed will start to really matter only for
larger sizes (beyond the scope of this study). 

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

Nevertheless, the main subject of this study are these more complex models that can
achieve higher accuracy than the simple linear models:

![plot-auc](1-linear/z-auc-lin-rf.png)

An interesting thing to note is that the AUC for random forest trained on 100K observations is better
than the AUC on a linear model trained on 10M observations (so "more data or better algorithms?" - it depends).


#### Random Forest

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
Spark   | 10K  |      150      |   10     | 65.5
        | 100K |      1000     |   30     | 67.9
        | 1M   |     crash     |          |

![plot-time](2-rf/x-plot-time.png)
![plot-auc](2-rf/x-plot-auc.png)

The [R](2-rf/1.R) implementation (randomForest package) is slow and inefficient in memory use. 
It cannot cope by default with a large number of categories, therefore the data had
to be one-hot encoded. The implementation uses 1 processor core, but with 2 lines of extra code
it is easy to build
the trees in parallel using all the cores and combine them at the end. However, it runs out
of memory already for *n* = 1M.

The [Python](2-rf/2.py) (scikit-learn) implementation is faster, more memory efficient and uses all the cores.
Variables needed to be one-hot encoded (which is more involved than for R) 
and for *n* = 10M doing this exhausted all the memory. However, even if using a larger machine
with 250GB of memory (and 140GB free for RF after transforming all the data) the Python implementation
runs out of memory and crashes for this larger size.

The [H2O](2-rf/4-h2o.R) implementation is fast, memory efficient and uses all cores. It deals
with categorical variables automatically. It is also more accurate than R/Python.
I think the reason for the latter is dealing properly with the categorical variables, i.e. internally in the algo
rather than working from a previously 1-hot encoded dataset where the link between the dummies 
belonging to the same original variable is lost. (The R package also deals properly with categorical variables if
the number of categories is small, but not in our case.) 

[Spark](2-rf/5b-spark.txt) (MLlib) implementation is slow, provides the lowest accuracy and 
it [crashes](2-rf/5c-spark-crash.txt) already at *n* = 1M disappointingly
(for a "big data" system).  Even when the machine had 250GB of RAM Spark crashed for *n* = 1M
and 500 trees, while it could finish for a small (and for any practical use pointless) number of trees 
e.g. 10 trees for *n* = 1M or e.g. 1 tree for
*n* = 10M (although in these cases Spark was still very slow).
Also, reading the data is more than one line of code and Spark does not provide a one-hot encoder
for the categorical data (therefore I used R for that). I also tried to provide the categorical
variables encoded simply as integers and passing the `categoricalFeaturesInfo` parameter, but that made
training even slower.
Finally, note again the low prediction accuracy vs the other methods (even with the highest value
allowed for the maximal depth of trees).

In addition to the above, several other random forest implementations have been tested 
(Weka, Revo ScaleR, Rborist R package, Mahout), 
but all of them proved slow and/or unable to scale to the larger sizes.


H2O n=10M 250GB (5000)

ntree    | depth  |   nbins  | mtries  | Time (hrs)   |  AUC
---------|--------|----------|---------|--------------|--------
500      |  20    |    20    | -1 (2)  |      1.2     |  77.8 
500      |  50    |    200   | -1 (2)  |              |
500      |  20    |    200   |   3     |              |
5000     |  50    |    200   | -1 (2)  |      45      |  79.0


#### Boosting

...
    

#### Deep neural networks

...

### Conclusions

...

