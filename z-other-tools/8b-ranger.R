library(readr)
library(ranger)
library(ROCR)

d_train <- read_csv("train-1m.csv")
d_test <- read_csv("test.csv")

d_train$dep_delayed_15min <- as.factor(d_train$dep_delayed_15min)
d_test$dep_delayed_15min  <- as.factor(d_test$dep_delayed_15min)

system.time({
  md <- ranger(dep_delayed_15min ~ ., d_train, num.trees = 100, probability = TRUE, write.forest = TRUE)
})

system.time({
  phat <- predictions(predict(md, data = d_test))[,"Y"]
})

rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")


