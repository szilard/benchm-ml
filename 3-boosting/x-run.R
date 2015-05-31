
library(ggplot2)
library(readr)
library(dplyr)
library(reshape2)

d <- read_csv("x-run.csv")

d %>% select(1:4) %>% melt(id.vars=1:2) %>%
  ggplot(aes(x = n, y = value, color = Tool)) +
  geom_point() + geom_line() + 
  facet_wrap(~variable, scales = "free", ncol = 2) +
  scale_x_log10(breaks = c(0.01,0.1,1,10)) + 
  scale_y_log10(breaks = c(1,10,100,1000,10000))

d %>% select(c(1,2,5,6)) %>% melt(id.vars=1:2) %>%
  ggplot(aes(x = n, y = value, color = Tool)) +
  geom_point() + geom_line() + 
  facet_wrap(~variable, scales = "free", ncol = 2) +
  scale_x_log10(breaks = c(0.01,0.1,1,10)) +
  scale_y_continuous(breaks = seq(60,80,2))


