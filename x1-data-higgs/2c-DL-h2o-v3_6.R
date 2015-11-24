
library(h2o)

for (size in c(0.0001,0.001,0.01,0.1,1,10)) {
print(size)

h2o.init(max_mem_size="250g", nthreads=-1)
Sys.sleep(3)


dx_train <- h2o.importFile(paste0("higgs-train-",format(size, scientific=FALSE),"m.csv"))
dx_valid  <- h2o.importFile("higgs-valid.csv")
dx_test  <- h2o.importFile("higgs-test.csv")

dx_train[,1] <- as.factor(dx_train[,1])
dx_valid[,1] <- as.factor(dx_valid[,1])
dx_test[,1] <- as.factor(dx_test[,1])



print(system.time({
  md <- h2o.deeplearning(x = 2:ncol(dx_train), y = 1, training_frame = dx_train,
            validation_frame = dx_valid,
            activation = "RectifierWithDropout", hidden = c(200,200,200,200), epochs = 100,
            l1 = 1e-5, l2 = 1e-5, hidden_dropout_ratios=c(0.2,0.1,0.1,0),
            stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
}))

print(h2o.performance(md, dx_test)@metrics$AUC)



h2o.shutdown(prompt = FALSE)
Sys.sleep(3)

}

