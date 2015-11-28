
## h2o 3.6.0.8
library(h2o)

h2o.init(max_mem_size = "50g", nthreads = -1)


dx_train <- h2o.importFile("train-1m.csv")
##dx_train <- h2o.importFile("train-10m.csv")
dx_valid <- h2o.importFile("valid.csv")
dx_test <- h2o.importFile("test.csv")


## to have same normalization as for the other DL libs that don't auto normalize 
dx_train$DepTime <- dx_train$DepTime/2500
dx_valid$DepTime <- dx_valid$DepTime/2500
dx_test$DepTime <- dx_test$DepTime/2500

dx_train$Distance <- log10(dx_train$Distance)/4
dx_valid$Distance <- log10(dx_valid$Distance)/4
dx_test$Distance <- log10(dx_test$Distance)/4


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]


system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train,
            validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), epochs = 100,
            ## activation = "RectifierWithDropout", hidden = c(200,200,200,200), epochs = 100,
            ## l1 = 1e-5, l2 = 1e-5, hidden_dropout_ratios=c(0.2,0.1,0.1,0),
            stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_AUC==max(d_scoring$validation_AUC, na.rm=TRUE),]$epochs[2:3]


