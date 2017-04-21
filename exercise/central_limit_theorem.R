#
# Explore the Central Limit Theorem.
# The Central Limit Theorem: This states that if you 1) sample n times from 
# a distribution, 2) calculate the mean, and 3) repeat that 1000 or so times, 
# the collection of your estimates of the mean will become more normally 
# distributed as the sample size n increases. This is true regardless of the 
# shape of the distribution from which you are sampling.
#

pdf("central_limit_theorem.pdf", width=7, height=5)
par(mfrow=c(2,3))

for (lambda_ in c(0.1, 5)) {
  for (n_ in c(10, 100, 1000)) {
    sample.mean <- vector(length = 1000)
    for (i in 1:1000) {
      sample.mean[i] <- mean(rpois(n = n_, lambda = lambda_))
    }
    hist(sample.mean, main = paste("lambda=", lambda_,", sample_size=", n_))
  }
}

dev.off()
