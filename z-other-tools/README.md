
#### Other tools

If your favorite tool is not listed here (either open source or commercial), 
you can get an idea of speed/accuracy with minimal work following the instructions below.

Get the [training data of 100K](https://s3.amazonaws.com/benchm-ml--main/train-1m.csv)
and the [test data](https://s3.amazonaws.com/benchm-ml--main/test.csv) CSV files.

I suggest you benchmark random forests, it's both easy to train and provides
pretty good accuracy. Use 100 trees and (max) depth 20 (if you can).

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

Size  |   Time   |  AUC
------|----------|---------
100K  |    4     |   0.726
1M    |    30    |   0.749
10M   |          |





