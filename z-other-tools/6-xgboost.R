library(readr)
library(ROCR)
library(xgboost)
library(parallel)
library(Matrix)

set.seed(123)

d_train <- read_csv("train-1m.csv")
d_test <- read_csv("test.csv")


system.time({
  X_train_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
  X_train <- X_train_test[1:nrow(d_train),]
  X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
})
dim(X_train)


# random forest with xgboost
system.time({
  n_proc <- detectCores()
  md <- xgboost(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0),
                 nthread = n_proc, nround = 1, max_depth = 20,
                 num_parallel_tree = 100, subsample = 0.632,
                 colsample_bytree = 1/sqrt(length(X_train@x)/nrow(X_train)))
})


system.time({
  phat <- predict(md, newdata = X_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")

