library(readr)
library(ROCR)
library(glmnet)
library(Matrix)

d_train <- read_csv("spark-train-1m.csv", col_names = FALSE)
d_test <- read_csv("spark-test-1m.csv", col_names = FALSE)

X_train <- as(as.matrix(d_train[,-1]), "sparseMatrix")
X_test <- as(as.matrix(d_test[,-1]), "sparseMatrix")


system.time({
  md <- glmnet(X_train, d_train[[1]], family = "binomial", lambda = 0)
})
## 5sec

phat <- predict(md, newx = X_test, type = "response")
rocr_pred <- prediction(phat, d_test[[1]])
performance(rocr_pred, "auc")
## 0.7107


