library(readr)
library(ROCR)
library(mxnet)
library(Matrix)

d_train <- read_csv("train-1m.csv")
d_valid <- read_csv("valid.csv")
d_test <- read_csv("test.csv")


## normalization
d_train$DepTime <- d_train$DepTime/2500
d_valid$DepTime <- d_valid$DepTime/2500
d_test$DepTime <- d_test$DepTime/2500

d_train$Distance <- log10(d_train$Distance)/4
d_valid$Distance <- log10(d_valid$Distance)/4
d_test$Distance <- log10(d_test$Distance)/4


system.time({
  ##X_train_test_valid <- Matrix::sparse.model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test, d_valid))
  X_train_test_valid <- model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test, d_valid))
  X_train <- X_train_test_valid[1:nrow(d_train),]
  X_test <- X_train_test_valid[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
  X_valid <- X_train_test_valid[(nrow(d_train)+nrow(d_test)+1):(nrow(d_train)+nrow(d_test)+nrow(d_valid)),]
})
##dim(X_train_test_valid)
##object.size(X_train_test_valid)/1e6


mx.set.seed(0)
system.time({
md <- mx.mlp(X_train, as.numeric(d_train$dep_delayed_15min=="Y"), 
          array.layout = "rowmajor", out_node = 2, 
          device = mx.gpu(),
          hidden = c(128,128), activation = "relu", ##dropout = 0.1,        
          num.round = 2, array.batch.size = 100,
          learning.rate = 0.07, momentum = 0.9, initializer = mx.init.uniform(0.07),
          eval.metric = mx.metric.accuracy, 
          eval.data = list(data = X_valid, label = as.numeric(d_valid$dep_delayed_15min=="Y"))
          )
})


phat <- t(predict(md, X_test, array.layout = "rowmajor"))[,2]
rocr_pred <- prediction(phat, as.numeric(d_test$dep_delayed_15min=="Y"))
performance(rocr_pred, "auc")



