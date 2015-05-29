
library(ggplot2)
library(readr)

d <- read_csv("x-overfit.csv")

ggplot(d, aes(x = n_trees, y = AUC_test)) +
  geom_point() + geom_line() 



