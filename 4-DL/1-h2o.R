
## h2o 3.10.0.8

library(h2o)
h2o.init(max_mem_size = "50g", nthreads = -1)


dx_train <- h2o.importFile("train-10m.csv")
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
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            ## DEFAULT: activation = "Rectifier", hidden = c(200,200), 
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  2.891   0.247 275.616 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7307861
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 0.1698963 1.8196927



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(50,50,50,50), input_dropout_ratio = 0.2,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  1.701   0.191 144.301 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7321902
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 1.549485 2.719795



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(50,50,50,50), 
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  1.393   0.203 108.415 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7315689
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 0.1498604 1.9090842



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(20,20),
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  1.406   0.174 104.139 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7306146
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 1.870624 4.640307



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(20),
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  1.482   0.097 124.429 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7315768
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 6.399249 6.669262




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(10),
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  1.685   0.117 152.939 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7325564
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1]  3.220322 11.859830



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(5),
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  1.345   0.173 107.650 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7294642
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 1.679884 9.310295




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(1),
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  1.435   0.225 121.663 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7121256
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1]  3.579505 13.149986






system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), l1 = 1e-5, l2 = 1e-5, 
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  2.740   0.314 267.694 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7312349
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 0.5702015 1.8402960



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "RectifierWithDropout", hidden = c(200,200,200,200), hidden_dropout_ratios=c(0.2,0.1,0.1,0),
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  4.324   0.418 443.058 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7332076
#> ## print epochs:  1: best AUC (on validation)  2: early stopping
#[1] 1.280414 2.000519


                        
