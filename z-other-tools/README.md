
#### How to Benchmark Your Tool of Choice with Minimal Work

##### aka The Absolute Minimal Benchmark

If your favorite software tool for machine learning (either open source or commercial) is not benchmarked here, 
you can get an idea of speed/accuracy with minimal work by following the instructions below.

Get the [training data of 100K records](https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv)
and the [test data](https://s3.amazonaws.com/benchm-ml--main/test.csv) (csv files).

I suggest you benchmark random forests, it's both easy to train and provides
pretty good accuracy. Use `100` trees and max depth `20`. Keep the categorical
variables non-ordinal and do not do any feature engineering to improve accuracy.

Do the following (pseudocode) in your system:
```
train_data = read(train_data_file)
test_data = read(test_data_file)

model = train_random_forest(train_data, n_trees = 100, depth = 20)

predictions = predict(model, test_data_without_labels)
calculate_AUC(predictions, test_data_labels_only)
```
For example in H2O it is [this easy](../2-rf/4-h2o.R).

Once this works, try the larger [training data of 1M records](https://s3.amazonaws.com/benchm-ml--main/train-1m.csv)
and next the [training data of 10M records](https://s3.amazonaws.com/benchm-ml--main/train-10m.csv).

Compare time and AUC with xgboost run on EC2 32 cores instance:

Size  | Time (sec) |  AUC
------|------------|---------
100K  |    4       |   0.726
1M    |    30      |   0.749
10M   |    600     |   0.763

If your software crashes for the larger size, you need more RAM. You can get up to
250GB on EC2.

Here are the results for `n = 1M` for a few software tools:

Tool    | Time (sec)  |   AUC
--------|-------------|----------
Python  |             |
H2O     |             |
xgboost |   30        |  0.749
Spark   |             |

If you have results for other software tool(s), contact me by submitting a github issue.


