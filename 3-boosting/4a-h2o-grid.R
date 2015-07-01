
library(h2o)

h2oServer <- h2o.init(max_mem_size="250g", nthreads=-1)

dx_train <- h2o.importFile(h2oServer, path = "train-10m.csv")
dx_valid <- h2o.importFile(h2oServer, path = "valid.csv")
dx_test <- h2o.importFile(h2oServer, path = "test.csv")


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]

system.time({
  mds <- h2o.gbm(x = Xnames, y = "dep_delayed_15min", data = dx_train, distribution = "bernoulli", 
          n.trees = c(100,300,1000,3000), 
          interaction.depth = c(2,5,10,20,50), shrinkage = c(0.01,0.03), n.minobsinnode = c(1,10),
          n.bins = c(100,1000))
})


d_auc <- data.frame()
for (md in mds@model) {
  phat <- h2o.predict(md, dx_valid)[,"Y"]
  auc <- h2o.performance(phat, dx_valid[,"dep_delayed_15min"])@model$auc
  d_auc <- rbind(d_auc, cbind(as.data.frame(
      md@model$params[c("n.trees","interaction.depth","shrinkage","n.minobsinnode","n.bins")]), auc = auc))
}
d_auc



i_sort <- order(d_auc$auc, decreasing = TRUE)
d_auc[i_sort,]
d_auc[i_sort,][1:10,]


md <- mds@model[[i_sort[1]]]
phat <- h2o.predict(md, dx_test)[,"Y"]
h2o.performance(phat, dx_test[,"dep_delayed_15min"])@model$auc


