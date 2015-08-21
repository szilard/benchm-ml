library(readr)
library(ROCR)
library(glmnet)


for (size in c(0.0001,0.001,0.01,0.1,1,10)) {
print(size)


d_train <- read_csv(paste0("higgs-train-",format(size, scientific=FALSE),"m.csv"), col_names = FALSE)
d_test <- read_csv("higgs-test.csv", col_names = FALSE)

d_train[[1]] <- as.factor(d_train[[1]])
d_test[[1]] <- as.factor(d_test[[1]])


print(system.time({
  md <- glmnet( as.matrix(d_train[,2:ncol(d_train)]), d_train$X1, family = "binomial", lambda = 0)
}))


system.time({
  phat <- predict(md, newx = as.matrix(d_test[,2:ncol(d_test)]), type = "response")
})

rocr_pred <- prediction(phat, d_test$X1)
print(performance(rocr_pred, "auc"))

}



