
library(readr)
library(ROCR)
library(gbm)

set.seed(123)

d_train <- read_csv("train-1m.csv")
d_test <- read_csv("test.csv")

d_train$dep_delayed_15min <- ifelse(d_train$dep_delayed_15min=="Y",1,0)
d_test$dep_delayed_15min <- ifelse(d_test$dep_delayed_15min=="Y",1,0)


system.time({
  md <- gbm(dep_delayed_15min ~ ., data = d_train, distribution = "bernoulli", 
            n.trees = 1000, 
            interaction.depth = 16, shrinkage = 0.01, n.minobsinnode = 1,
            bag.fraction = 0.5)
})


phat <- predict(md, newdata = d_test, n.trees = md$n.trees, type = "response")
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]






