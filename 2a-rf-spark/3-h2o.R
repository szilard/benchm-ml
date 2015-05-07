
library(h2o)

h2oServer <- h2o.init(max_mem_size="200g", nthreads=-1)

dx_train <- h2o.importFile(h2oServer, path = "milsongs-cls-train.csv")
dx_test <- h2o.importFile(h2oServer, path = "milsongs-cls-test.csv")

system.time({
  md <- h2o.randomForest(y = 1, x = 2:91, data = dx_train, 
  ntree = 10,  depth = 20, nbins = 50, type="BigData")
})

system.time({
phat <- h2o.predict(md, dx_test)[,3]
})
h2o.performance(phat, dx_test[,1])@model$auc



