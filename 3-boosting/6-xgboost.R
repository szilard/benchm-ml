library(readr)
library(ROCR)
library(xgboost)
library(parallel)
library(Matrix)

set.seed(123)

d_train <- read_csv("train-10m.csv")
d_test <- read_csv("test.csv")


system.time({
  X_train_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
  n1 <- nrow(d_train)
  n2 <- nrow(d_test)
  X_train <- X_train_test[1:n1,]
  X_test <- X_train_test[(n1+1):(n1+n2),]
})
dim(X_train)

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))



system.time({
n_proc <- detectCores()
md <- xgb.train(data = dxgb_train, nthread = n_proc, 
                 objective = "binary:logistic", nround = 1000, 
                 max_depth = 16, eta = 0.01, subsample = 0.5,
                 min_child_weight = 1)
})



system.time({
  phat <- predict(md, newdata = X_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")


