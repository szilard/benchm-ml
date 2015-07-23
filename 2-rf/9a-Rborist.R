library(data.table)
library(Rborist)
library(ROCR)

dx_train <- as.data.frame(fread("train-1m.csv"))
dx_test <- as.data.frame(fread("test.csv"))

dt_train <- dx_train
dt_test <- dx_test

# Rborist 0-1.1 only accepts factor and numeric predictors or response:
#
facCols <- c("UniqueCarrier", "Origin","Dest", "Month", "DayofMonth", "DayOfWeek")

responseCol <- "dep_delayed_15min"
numCols <- c("DepTime","Distance")

for (k in facCols) {
  dt_train[[k]] <- as.factor(dx_train[[k]])
  dt_test[[k]] <- as.factor(dx_test[[k]])
}

dt_train[[responseCol]] <- as.factor(dx_train[[responseCol]])
dt_test[[responseCol]] <- as.factor(dx_test[[responseCol]])

for (k in numCols) {
  dt_train[[k]] <- as.numeric(dx_train[[k]])
  dt_test[[k]] <- as.numeric(dx_test[[k]])
}

Xnames <- names(dt_train)[which(names(dt_train) != responseCol)]



system.time({
  md <- Rborist(dt_train[, Xnames], dt_train[, responseCol], nTree = 100)
})

system.time({
  phat <- predict(md, newdata=dt_test[, Xnames], ctgCensus="prob")$ctgCensus[,"Y"]
})

rocr_pred <- prediction(phat, dt_test$dep_delayed_15min == "Y")
performance(rocr_pred, "auc")




