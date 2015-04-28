
library(ggplot2)
library(readr)

d <- read_csv("z-auc-lin-rf.csv")

ggplot(d, aes(x = n, y = AUC, color = Model)) +
  geom_point() + geom_line() + scale_x_log10(breaks = c(0.01,0.1,1,10))


