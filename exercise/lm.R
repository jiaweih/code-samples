# The objective is to practice testing the assumptions behind general linear models, 
# making data alterations to better satisfy assumptions, and practice indexing results
# from linear model outputs and creating summary tables.

library(ggplot2)

ggplotRegression <- function (fit) {

  ggplot(fit$model, aes_string(x = names(fit$model)[2], y = names(fit$model)[1])) + 
  geom_point() +
  stat_smooth(method = "lm", col = "red") +
  labs(title = paste("Adj R2 = ",signif(summary(fit)$adj.r.squared, 5),
                     "Intercept =",signif(fit$coef[[1]],5 ),
                     " Slope =",signif(fit$coef[[2]], 5),
                     " P =",signif(summary(fit)$coef[2,4], 5)))
} 
  # theme(plot.title = element_text(family = "Trebuchet MS", color="#666666",
  #                                 face="bold", size=8, hjust=0))
# }

plot_lm <- function(lm, name) {
  # Plot the linear model  
  p1 <- ggplotRegression(lm)

  # Plot histogram of the residuals
  range.residuals = range(lm$residuals)
  binwidth = (range.residuals[2] - range.residuals[1])/10
  p2 <- ggplot(data.frame(lm$residuals), aes(lm.residuals)) + 
        geom_histogram(binwidth = binwidth)

  # Check the independence of the residuals by plotting residuals 1 through the
  # second-to-last residual, against residuals 2 through the last residual.
  lag.1 <- lm$residuals[-length(lm$residuals)]
  lag.2 <- lm$residuals[-1]
  fit <- lm(lag.2 ~ lag.1, data=data.frame(c(lag.1, lag.2)))
  p3 <- ggplotRegression(fit)

  # Save the graphs
  dir.create("graphs", showWarnings = FALSE)
  pdfs <- list(pdf(paste0("graphs/", name, "_residuals_hist.pdf")),
               pdf(paste0("graphs/", name, "_lag.pdf")),
               pdf(paste0("graphs/", name, ".pdf")))
  ps <- list(p1, p2, p3)
  
  for (i in 1:3) {
    pdfs[[i]]
    print(ps[i])
    dev.off()
  }
}


### Read and pre-process data
birds <- read.csv ("http://130.111.193.18/stats/birdsdiet.csv")
names(birds) <- tolower(names(birds))
# Put a period before each "abund" to separate from max and avg.
names(birds) <- gsub('abund', '.abund', names(birds))
# Change "Pelecans" to "Pelicans" in the family column
birds$family <- gsub('Pelecans', 'Pelicans', birds$family)

# Run a linear model of 'max.abund' (y) against 'mass' (x)
data <- birds
lm1 <- lm('max.abund ~ mass', data)
plot_lm(lm1, 'lm1')

# log transform both variables
data <- birds
data$max.abund <- log(data$max.abund)
data$mass <- log(data$mass)
lm2 <- lm('max.abund ~ mass', data)
plot_lm(lm2, 'lm2')

# Exclude row 32 (outlier) and rerun
data <- birds
data <- data[-c(32),]
data$max.abund <- log(data$max.abund)
data$mass <- log(data$mass)
lm3 <- lm('max.abund ~ mass', data)
plot_lm(lm3, 'lm3')

# Make a table to deposit some comparative results
# from different approaches above
results.lm <- function(lm, name) {
    # Deposit the slope and intercept values.
    slope <- signif(lm$coef[[2]], 5)
    intercept <- signif(lm$coef[[1]], 5)
    # Deposit the r-sq values.
    r.sq <- signif(summary(lm)$r.squared, 5)
    # Deposit the p-values.
    p.slope <- signif(summary(lm1)$coefficients['mass', 'Pr(>|t|)'], 5)
    p.intercept <- signif(summary(lm1)$coefficients['(Intercept)', 'Pr(>|t|)'], 5)
    return(c(model=name, slope=slope, intercept=intercept, r.sq=r.sq,
                p.slope=p.slope, p.intercept=p.intercept))
}


results.lm1 <- results.lm(lm1, 'lm1')
results.lm2 <- results.lm(lm2, 'lm2')
results.lm3 <- results.lm(lm3, 'lm3')
results <- as.data.frame(do.call(rbind, lapply(list(results.lm1, results.lm2, results.lm3), t)))

write.table(results, file = "models_compare.csv")