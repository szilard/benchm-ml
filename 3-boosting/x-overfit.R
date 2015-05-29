
library(ggplot2)
library(readr)
library(dplyr)
library(reshape2)

d <- read_csv("x-overfit.csv")

d %>% melt(id.vars = "n_trees") %>%
  ggplot(aes(x = n_trees, y = value, color = variable)) +
  geom_point() + geom_line() + 
  facet_wrap(~ variable, ncol = 1, scales = "free") +
  theme(legend.position = "none")



