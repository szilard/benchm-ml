
library(h2o)

h2o.init(max_mem_size="60g", nthreads=-1)

dx_train <- h2o.importFile(path = "train-1m.csv")
dx_test <- h2o.importFile(path = "test.csv")


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]


system.time({
  md <- h2o.gbm(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, distribution = "bernoulli", 
          ntrees = 1000, 
          max_depth = 16, learn_rate = 0.01, min_rows = 1,
          nbins = 100)
})


system.time({
  print(h2o.auc(h2o.performance(md, dx_test)))
})



