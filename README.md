
### Simple/basic/limited/incomplete benchmark of machine learning libraries for scalability/speed and accuracy

This project aims at a *minimal* benchmark for scalability, speed and accuracy of commonly used implementations
of a few machine learning algorithms. The focus is 2-way classification with numeric and categorical inputs (of 
limited cardinality i.e. not very sparse) and no missing data. If the input matrix is of *n* x *p*, *n* is 
varied as 10K, 100K, 1M, 10M, while *p* is maximum a few thousands even after expanding the categoricals into dummy 
variables/one-hot encoding.

The algorithms studied are linear (logistic regression, linear SVM), random forest, boosting and deep learning in
various commonly used implementations like R packages, Python scikit-learn, Vowpal Wabbit, H2O and Spark MLlib.
Random forest, boosting and more recently deep neural networks are expected to perform the best on the data/sizes
described above. Non-linear SVMs are also good in accuracy in general but become slow/cannot scale for the larger *n*
sizes we want to deal with. The linear models are less accurate in general and used here as a base-line (they also scale 
very well and some of them can deal with very sparse features).


