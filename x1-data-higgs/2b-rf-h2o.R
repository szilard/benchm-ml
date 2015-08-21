
library(h2o)

for (size in c(0.0001,0.001,0.01,0.1,1,10)) {
print(size)

h2oServer <- h2o.init(max_mem_size="250g", nthreads=-1)
Sys.sleep(5)

dx_train <- h2o.importFile(h2oServer, path = paste0("higgs-train-",format(size, scientific=FALSE),"m.csv"))
dx_test  <- h2o.importFile(h2oServer, path = "higgs-test.csv")

dx_train[,1] <- as.factor(dx_train[,1])
dx_test[,1] <- as.factor(dx_test[,1])


print(system.time({
  md <- h2o.randomForest(x = 2:ncol(dx_train), y = 1, data = dx_train, ntree = 500, type="BigData")
}))


system.time({
  phat <- h2o.predict(md, dx_test)[,3]
  print(h2o.performance(phat, dx_test[,1])@model$auc)
})

h2o.shutdown(h2oServer, prompt = FALSE)
Sys.sleep(5)

}

