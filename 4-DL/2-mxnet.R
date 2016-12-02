library(readr)
library(ROCR)
library(mxnet)
library(Matrix)
library(magrittr)

d_train <- read_csv("train-1m.csv")
d_test <- read_csv("test.csv")


## normalization (without normalization garbage result AUC 0.5)
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

md_spec <- mx.symbol.Variable('data') %>%
  mx.symbol.FullyConnected(num_hidden = 200) %>% mx.symbol.Activation(act_type = "relu") %>%
  mx.symbol.FullyConnected(num_hidden = 200) %>% mx.symbol.Activation(act_type = "relu") %>%
  mx.symbol.FullyConnected(num_hidden = 2) %>% mx.symbol.SoftmaxOutput()

system.time({
  md <- mx.model.FeedForward.create(md_spec, 
               X = X_train, y = as.numeric(d_train$dep_delayed_15min=="Y"), array.layout = "rowmajor",
               initializer = mx.init.normal(0.1),
               eval.metric = mx.metric.accuracy,
               ##optimizer = mxnet:::mx.opt.sgd(learning.rate = 0.05, momentum = 0.9),  ## bug?
               learning.rate = 0.01, momentum = 0.9,  
               ctx = mx.gpu(), 
               ##ctx = mx.cpu(), 
               num.round = 1, array.batch.size = 128,
               epoch.end.callback = mx.callback.log.train.metric(100))
})

#   user  system elapsed 
# 50.665   7.177  33.925     ## GPU (P2)
# AUC 0.7125609  (0.5!!! if no normalization is used)

#    user   system  elapsed
# 289.609 1332.340   66.775    ## CPU (r3 32cores, openblas)
#    90.755   6.027  81.520    ## no BLAS uses 1 core only
# AUC 0.7125485


phat <- t(predict(md, X_test, array.layout = "rowmajor"))[,2]
rocr_pred <- prediction(phat, as.numeric(d_test$dep_delayed_15min=="Y"))
performance(rocr_pred, "auc")



