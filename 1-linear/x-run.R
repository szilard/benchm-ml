
library(ggplot2)
library(readr)

d <- read_csv("x-run.csv")

ggplot(d, aes(x = n, y = Time, color = Tool)) +
  geom_point() + geom_line() + scale_x_log10(breaks = c(0.01,0.1,1,10)) + scale_y_log10()

ggplot(d, aes(x = n, y = AUC, color = Tool)) +
  geom_point() + geom_line() + scale_x_log10(breaks = c(0.01,0.1,1,10))


