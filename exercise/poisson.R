#
# Explore Poisson distributions.
#

library(ggplot2)

dpoi <- function(lambda, k) {
  # The probability mass function of the Poisson distribution
  p <- lambda ** k * exp(-lambda) / factorial(k)
  return(p)
}


dpoi.to.dataframe <- function(lambda) {
  # Calculate Poisson probablity of a given lambda 
  # against multiple k and save it to a data frame.
  p <- vector(length = 20)

  # Calculate the probability of X=k given lambda=1, for k=1:20. 
  for (i in 1:20) {
    p[i] <- dpoi(lambda, i)
  }
  # Deposit the vector p into a data frame
  p.data.frame <- data.frame(p=p, k=1:20, lambda=lambda)
  return(p.data.frame)
}

# Make three data frames for lambda=1, 4, 10.
pois.dataframe <- do.call(rbind, lapply(c(1, 4, 10), dpoi.to.dataframe))
pois.dataframe$lambda <- factor(pois.dataframe$lambda)

pdf("poisson.pdf", width=7, height=5)
# Plot p against k with both points and lines.
ggplot(data=pois.dataframe, aes(x=k, y=p, fill=lambda, group=lambda, 
    shape=lambda, colour=lambda)) + geom_line() + geom_point() + 
    theme(legend.position=c(0.9,0.9))
dev.off()