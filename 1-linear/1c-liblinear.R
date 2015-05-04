
library(LiblineaR)
library(readr)
library(ROCR)

d_train <- read_csv("train-10m.csv")
d_test <- read_csv("test.csv")

for (k in c("Month","DayofMonth","DayOfWeek")) {
  d_train[[k]] <- as.character(d_train[[k]])
  d_test[[k]] <- as.character(d_test[[k]])
}
sapply(d_train, class)


system.time({
X_train_test <-  model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test))
X_train <- X_train_test[1:nrow(d_train),]
X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]
})


system.time({
md <- LiblineaR( X_train, d_train$dep_delayed_15min, epsilon=1e-5, cost=1000)
})


system.time({
phat <- predict(md, newx = X_test, proba = TRUE)$probabilities[,"Y"]
})

rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")


