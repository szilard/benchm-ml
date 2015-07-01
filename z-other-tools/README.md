
#### How to Benchmark Your Tool of Choice with Minimal Work

If your favorite tool is not listed here (either open source or commercial), 
you can get an idea of speed/accuracy with minimal work following the instructions below.

Get the [training data of 100K](https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv)
and the [test data](https://s3.amazonaws.com/benchm-ml--main/test.csv) CSV files.

I suggest you benchmark random forests, it's both easy to train and provides
pretty good accuracy. Use `100` trees and max depth `20`. Keep the categorical
variables non-ordinal and do not do any feature engineering to improve accuracy.

Write the following code for your system:
```
train_data = read(train_data_file)
test_data = read(test_data_file)

model = train_random_forest(train_data, n_trees = 100, depth = 20)

predictions = predict(model, test_data_without_labels)
calculate_AUC(predictions, test_data_labels_only)
```

Once this works, try larger [training data of 1M](https://s3.amazonaws.com/benchm-ml--main/train-1m.csv)
and next [training data of 10M](https://s3.amazonaws.com/benchm-ml--main/train-10m.csv).

Compare time and AUC with xgboost ran on EC2 32 cores instance:

Size  | Time (sec) |  AUC
------|------------|---------
100K  |    4       |   0.726
1M    |    30      |   0.749
10M   |    600     |   0.763

If your software crashes for the larger size, you need more RAM. You can get up to
250GB on EC2.

If your software still crashes (especially on the smaller size), 
takes 10x more time to run or provides AUC lower by 0.05 than
the above results, then use better software.


