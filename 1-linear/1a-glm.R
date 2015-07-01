
library(readr)
library(ROCR)

d_train <- read_csv("train-0.01m.csv")
d_test <- read_csv("test.csv")


system.time({
md <- glm( ifelse(dep_delayed_15min == "Y", 1, 0) ~ ., data = d_train, family = binomial())
})


d_test <- d_test[d_test$UniqueCarrier %in% unique(d_train$UniqueCarrier),]
d_test <- d_test[d_test$Origin %in% unique(d_train$Origin),]
d_test <- d_test[d_test$Dest %in% unique(d_train$Dest),]

system.time(
phat <- predict(md, newdata = d_test, type = "response")
)

rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")



