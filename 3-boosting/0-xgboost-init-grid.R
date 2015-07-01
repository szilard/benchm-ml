library(readr)
library(ROCR)
library(xgboost)
library(parallel)
library(Matrix)

set.seed(123)

d_train <- read_csv("train-0.1m.csv")
d_valid <- read_csv("valid.csv")
d_test <- read_csv("test.csv")


system.time({
  X_train_valid_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_valid, d_test))
  n1 <- nrow(d_train)
  n2 <- nrow(d_valid)
  n3 <- nrow(d_test)
  X_train <- X_train_valid_test[1:n1,]
  X_valid <- X_train_valid_test[(n1+1):(n1+n2),]
  X_test <- X_train_valid_test[(n1+n2+1):(n1+n2+n3),]
})
dim(X_train)

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))
dxgb_valid <- xgb.DMatrix(data = X_valid, label = ifelse(d_valid$dep_delayed_15min=='Y',1,0))
dxgb_test  <- xgb.DMatrix(data = X_test,  label = ifelse(d_test$dep_delayed_15min =='Y',1,0))



params <- expand.grid(max_depth = c(1,2,3,5,8,12,16), eta = c(0.3,0.1,0.03,0.01), 
      min_child_weight = c(1,10), subsample = c(1,0.5))

for (k in 1:nrow(params)) {
  prm <- params[k,]
  print(prm)
  print(system.time({
    n_proc <- detectCores()
    md <- xgb.train(data = dxgb_train, nthread = n_proc, 
                 objective = "binary:logistic", nround = 5000, 
                 max_depth = prm$max_depth, eta = prm$eta, 
                 min_child_weight = prm$min_child_weight, subsample = prm$subsample, 
                 watchlist = list(valid = dxgb_valid, train = dxgb_train), eval_metric = "auc",
                 early_stop_round = 100, printEveryN = 100)
  }))
}


system.time({
  phat <- predict(md, newdata = X_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")


