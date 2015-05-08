library(data.table)
library(ROCR)
library(xgboost)

set.seed(123)

d_train <- as.data.frame(fread("train-0.1m.csv"))
d_test <- as.data.frame(fread("test.csv"))

## "Can not handle categorical predictors with more than 53 categories."
## so need dummy variables/1-hot encoding
## - but then RF does not treat them as 1 variable
system.time({
  X_train_test <- Matrix::sparse.model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test))
  X_train <- X_train_test[1:nrow(d_train),]
  X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
})
dim(X_train)


system.time({
  n_proc <- detectCores()
  bst <- xgboost(data = X_train, label = d_train$dep_delayed_15min,nthread = n_proc)
})


system.time({
  phat <- predict(bst, newdata = X_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")

gc()
sapply(ls(),function(x) object.size(get(x))/1e6)

