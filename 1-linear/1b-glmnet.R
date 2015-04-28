
library(readr)
library(ROCR)
library(glmnet)

d_train <- read_csv("train-10m.csv")
d_test <- read_csv("test.csv")

system.time({
X_train_test <-  model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test))
X_train <- X_train_test[1:nrow(d_train),]
X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
})
dim(X_train)


system.time({
  md <- glmnet( X_train, d_train$dep_delayed_15min, family = "binomial", lambda = 0)
})


system.time({
  phat <- predict(md, newx = X_test, type = "response")
})

rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")



