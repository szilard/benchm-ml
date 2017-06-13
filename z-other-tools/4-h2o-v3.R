
library(h2o)

h2o.init(max_mem_size="60g", nthreads=-1)


dx_train <- h2o.importFile(path = "train-1m.csv")
dx_test <- h2o.importFile(path = "test.csv")


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]

system.time({
  md <- h2o.randomForest(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, ntrees = 100)
})

system.time({
  print(h2o.performance(md, dx_test)@metrics$AUC)
})


