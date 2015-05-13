
library(h2o)

## java -Xmx50g -jar h2o-2.8.6.2/h2o.jar -flatfile h2o-hosts.txt
h2oServer <- h2o.init(startH2O = FALSE)

dx_train <- h2o.importFile(h2oServer, path = paste0("train-10m.csv"))
dx_test <- h2o.importFile(h2oServer, path = "test.csv")

for (k in c("Month","DayofMonth","DayOfWeek")) {
  dx_train[[k]] <- as.factor(dx_train[[k]])
  dx_test[[k]] <- as.factor(dx_test[[k]])
}

Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]



system.time({
  md <- h2o.randomForest(x = Xnames, y = "dep_delayed_15min", data = dx_train, 
               ntree = 10, depth=20, nbins=50, type="BigData")
})


system.time({
phat <- h2o.predict(md, dx_test)[,"Y"]
})
h2o.performance(phat, dx_test[,"dep_delayed_15min"])@model$auc



