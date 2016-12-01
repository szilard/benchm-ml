

library(h2o)
h2o.init(max_mem_size = "50g", nthreads = -1)


dx_train <- h2o.importFile("train-1m.csv")
dx_test <- h2o.importFile("test.csv")


## to have same normalization as for the other DL libs that don't auto normalize 
dx_train$DepTime <- dx_train$DepTime/2500
dx_test$DepTime <- dx_test$DepTime/2500

dx_train$Distance <- log10(dx_train$Distance)/4
dx_test$Distance <- log10(dx_test$Distance)/4


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 0,
            momentum_start = 0.9, momentum_stable = 0.9, nesterov_accelerated_gradient = FALSE,
            epochs = 1) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed 
#  0.750   0.008  51.358     ## 32 cores
# AUC 0.7226275   (with no explicit normalization 0.7237816)


