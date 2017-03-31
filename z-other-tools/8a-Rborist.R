library(data.table)
library(Rborist)
library(ROCR)

dx_train <- as.data.frame(fread("train-1m.csv"))
dx_test <- as.data.frame(fread("test.csv"))

# Rborist only accepts factor and numeric predictors or response:
#
facCols <- c("UniqueCarrier", "Origin","Dest", "Month", "DayofMonth", "DayOfWeek")
numCols <- c("DepTime","Distance")
responseCol <- "dep_delayed_15min"

for (k in facCols) {
  dx_train[[k]] <- as.factor(dx_train[[k]])
  dx_test[[k]] <- as.factor(dx_test[[k]])
}

for (k in numCols) {
  dx_train[[k]] <- as.numeric(dx_train[[k]])
  dx_test[[k]] <- as.numeric(dx_test[[k]])
}

dx_train[[responseCol]] <- as.factor(dx_train[[responseCol]])
dx_test[[responseCol]] <- as.factor(dx_test[[responseCol]])

Xnames <- names(dx_train)[which(names(dx_train) != responseCol)]



print(system.time({
  md <- Rborist(dx_train[, Xnames], dx_train[, responseCol], nTree = 100, nLevel = 20, thinLeaves=TRUE)
}))

system.time({
  phat <- predict(md, newdata=dx_test[, Xnames], ctgCensus="prob")$prob[,"Y"]
})

rocr_pred <- prediction(phat, dx_test$dep_delayed_15min == "Y")
performance(rocr_pred, "auc")




