library(data.table)
library(ROCR)
library(Matrix)
library(Rborist)

set.seed(123)


d_train <- as.data.frame(fread("train-1m.csv"))
d_test <- as.data.frame(fread("test.csv"))


system.time({
  X_train_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
  X_train <- X_train_test[1:nrow(d_train),]
  X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
})

system.time({
    md <- Rborist(X_train,
                  ifelse(d_train$dep_delayed_15min=='Y',1.0,0.0),
                  nLevel=20, nTree=100, predProb = 1/sqrt(length(X_train@x)/nrow(X_train)), thinLeaves=TRUE)
})

system.time({
  phat <- predict(md, newdata = X_test)
})

rocr_pred <- prediction(phat$yPred, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")

