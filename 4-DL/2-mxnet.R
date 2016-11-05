library(readr)
library(ROCR)
library(mxnet)
library(Matrix)

d_train <- read_csv("train-1m.csv")
d_test <- read_csv("test.csv")


## normalization
d_train$DepTime <- d_train$DepTime/2500
d_test$DepTime <- d_test$DepTime/2500

d_train$Distance <- log10(d_train$Distance)/4
d_test$Distance <- log10(d_test$Distance)/4


system.time({
  X_train_test <- model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test))
  X_train <- X_train_test[1:nrow(d_train),]
  X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
})


mx.set.seed(123)

system.time({
md <- mx.mlp(X_train, as.numeric(d_train$dep_delayed_15min=="Y"), 
          array.layout = "rowmajor", out_node = 2, 
          device = mx.gpu(),
          hidden = c(200,200), activation = "relu", 
          num.round = 1, array.batch.size = 128,
          learning.rate = 0.01, momentum = 0.9, initializer = mx.init.uniform(0.1),
          eval.metric = mx.metric.accuracy)
})


phat <- t(predict(md, X_test, array.layout = "rowmajor"))[,2]
rocr_pred <- prediction(phat, as.numeric(d_test$dep_delayed_15min=="Y"))
performance(rocr_pred, "auc")



