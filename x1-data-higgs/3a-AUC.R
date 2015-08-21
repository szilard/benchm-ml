
library(ggplot2)
library(readr)
library(dplyr)
library(reshape2)

d <- read_csv("3a-AUC.csv") %>% melt(id.vars = "n")
names(d) <- c("n","Model","AUC")
 
ggplot(d, aes(x = n, y = AUC, color = Model)) +
  geom_point() + geom_line() + 
  scale_x_log10(breaks = c(0.0001,0.001,0.01,0.1,1,10), 
                labels = c(0.0001,0.001,0.01,0.1,1,10))



