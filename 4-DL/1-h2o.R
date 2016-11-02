
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





system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.95, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  2.823   0.615 242.150 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7110229
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.3701672 1.7100978
    



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  2.989   0.362 272.354 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7334915
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.4300052 1.9097695      



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.9999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  2.778   0.228 264.193 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7265394
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.270231 1.960119




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.999, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed
#  3.094   0.711 287.199
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7219705
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.6599353 2.3602646





system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.999, epsilon = 1e-09,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  2.826   0.505 261.192 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7317512
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.2597259 1.8000915




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, ## default: rate = 0.005, rate_decay = 1, momentum_stable = 0,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  3.335   0.238 340.291 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7302248
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.2198763 1.1098714      




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.001, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  4.056   0.383 412.792 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7327167
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.2098676 0.7397312




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  2.844   0.222 276.620 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7330604
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.1199309 0.9497759





system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  3.802   0.494 361.527 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7352361
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.3296777 1.0097397




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-04, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#    user   system  elapsed 
#  37.083    6.723 3762.738 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7275267
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 8.629915 8.780104



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.9,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  3.850   0.220 355.542 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7348393
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.4099075 0.9699861



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e4, momentum_stable = 0.9,
            epochs = 100, stopping_rounds = 10, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]

#   user  system elapsed 
#  3.560   0.191 337.696 
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7329273
#> d_scoring <- md@model$scoring_history
#> d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]$epochs[2:3]
#[1] 0.2500784 0.9199458




