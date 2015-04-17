
library(data.table)
library(ROCR)
library(randomForest)
library(parallel)

set.seed(123)

d_train <- as.data.frame(fread("train-0.1m.csv"))
d_test <- as.data.frame(fread("test.csv"))

## "Can not handle categorical predictors with more than 53 categories."
## so need dummy variables/1-hot encoding
## - but then RF does not treat them as 1 variable
system.time({
X_train_test <-  model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test))
X_train <- X_train_test[1:nrow(d_train),]
X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
})
dim(X_train)


system.time({
n_proc <- detectCores()
mds <- mclapply(1:n_proc,
        function(x) randomForest(X_train, as.factor(d_train$dep_delayed_15min), 
              ntree = floor(500/n_proc)), mc.cores = n_proc)
md <- do.call("combine", mds)
})


system.time({
phat <- predict(md, newdata = X_test, type = "prob")[,"Y"]
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")

gc()
sapply(ls(),function(x) object.size(get(x))/1e6)


