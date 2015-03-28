
### Simple/limited/incomplete benchmark for scalability/speed and accuracy of machine learning libraries for classification

This project aims at a *minimal* benchmark for scalability, speed and accuracy of commonly used implementations
of a few machine learning algorithms. The target is 2-way classification with numeric and categorical inputs (of 
limited cardinality i.e. not very sparse) and no missing data. If the input matrix is of *n* x *p*, *n* is 
varied as 10K, 100K, 1M, 10M, while *p* is maximum a few thousands (even after expanding the categoricals into dummy 
variables/one-hot encoding).

The algorithms studied are 
- linear (logistic regression, linear SVM)
- random forest
- boosting 
- deep learning 

in various commonly used open source implementations like 
- R packages
- Python scikit-learn
- Vowpal Wabbit
- H2O 
- Spark MLlib.

Random forest, boosting and more recently deep neural networks are the algos expected to perform the best on the structure/sizes
described above (e.g. vs alternatives such as *k*-nearest neighbors, naive-Bayes, decision trees etc). 
Non-linear SVMs are also among the best in accuracy in general, but become slow/cannot scale for the larger *n*
sizes we want to deal with. The linear models are less accurate in general and are used here only 
as a baseline (but they can scale better and some of them can deal with very sparse features). 

Scalability here means the algos are able to complete (in decent time) for the given *n* sizes. 
The main contraint is RAM (a given algo/implementation can crash if running out of memory), but some 
of the algos/implementations can scale beyond 1 machine (node). Speed is determined by computational
complexity but also some algos/implementations can use multiple processor cores and even multiple nodes.
Accuracy is measured by AUC. The interpretability of models is not of concern in this project.

In summary, we are focusing on which algos/implementations can be used to train accurate 2-way classifiers for data
with millions of observations and thousands of features processed on commodity hardware (mainly one machine with decent RAM and several cores).



