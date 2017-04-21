#
# Build a manual t-test function and a function for a bootstrap power analysis.
#

sig_t <- function(data, data.col, group.col, group.1, group.2) {
    # t-test, returning a two-tailed significance value.
    #
    # Args:
    #   data: A dataframe containing the data to be tested
    #   data.col: The name of the column in 'data' containing the data for the
    #             comparison of means.
    #   group.col: The name of the column in 'data' containing the grouping variables.
    #   group.1: The first grouping variable in 'group.col'.
    #   group.2: The second grouping variable in 'group.col'.
    #
    # Returns:
    #   A two-tailed significance value.
    data <- data[!is.na(data[,data.col]),]
    # Make row-indexing objects containing the ROW NUMBERS for the two groups.
    grp1.rows <- row.names(data[data[, group.col] == group.1,])
    grp2.rows <- row.names(data[data[, group.col] == group.2,])
    n1 <- length(grp1.rows)
    n2 <- length(grp2.rows)
    # Get the mean value for each group.
    mean1 <- mean(data[grp1.rows, data.col])
    mean2 <- mean(data[grp2.rows, data.col])
    # Calculate the degrees of freedom
    df <- n1 + n2 - 2
    # Calculate the pooled variance of the data.
    ss1 <- sum((data[grp1.rows, data.col] - mean1) ^ 2)
    ss2 <- sum((data[grp2.rows, data.col] - mean2) ^ 2)
    # Get the pooled variance 
    pooled.var <- (ss1 + ss2)/df
    # Calculate the t statistic. 
    t <- (mean1 - mean2) / sqrt(pooled.var*(1/n1 + 1/n2))
    # Calculate the p value from the cumulative distribution function for the
    # t-distribution. 
    p <- pt(t, df)
    # Convert the p value to a two-tailed p.
    p <- ifelse(p >= 0.5, (1 - p)*2, p*2)
    
    return(p)
}

boot_power <- function(data, data.col, group.col, group1, group2, n.boot=1000, alpha=0.5) {
    # Perform a bootstrap power analysis
    #
    # Args:
    #   data: The dataframe containing the data of interest
    #   data.col: The name of the column containing the data to analyze.
    #   group.col: The name of the column containing the grouping variables
    #   group1: The grouping variable in group.col for group1
    #   group2: The grouping variable in group.col for group2
    #   n.boot: the number of bootstrap iterations to run. 
    #   n1: the sample size for the randomized sample for group1
    #   n2: the sample size for the randomized sample for group2.
    #   alpha: The significance threshold. Give it a default value of 0.05
    #
    # Returns:
    #   The statistical power.
    data <- data[!is.na(data[,data.col]),]
    
    # Calculate the observed mean value for each group
    m1 <- mean(data[data[, group.col]==group1, data.col])
    m2 <- mean(data[data[, group.col]==group2, data.col])
    
    # Calculate the degrees of freedom
    n1 <- length(data[data[, group.col]==group1, data.col])
    n2 <- length(data[data[, group.col]==group2, data.col])
    df <- n1 + n2 - 2
    
    # Calculate the pooled variance
    ss1 <- sum((data[data[, group.col]==group1, data.col] - m1) ^ 2) 
    ss2 <- sum((data[data[, group.col]==group2, data.col] - m2) ^ 2)
    pooled.var <- (ss1 + ss2)/df
    
    # Use a loop with ‘n.boot’ iterations to fill a vector of p-values obtained
    # from t-tests on randomly generated data.
    p_vec <- vector(length = n.boot)
    
    for (i in seq(n.boot)) {
        # Generate a random sample from a normal distribution 
        rand1 <- rnorm(n = n1, mean = m1, sd = sqrt(pooled.var))
        rand2 <- rnorm(n = n2, mean = m2, sd = sqrt(pooled.var))
        # Make a data frame with a group column and a data column. 
        dataframe <- data.frame(group.col = c(rep(group1, n1), rep(group2, n2)), 
                                data.col = c(rand1, rand2))
        colnames(dataframe) <- c(group.col, data.col)
        # Use the sig_t() function to return the p-value from a t-test
        p <- sig_t(dataframe, data.col, group.col, group1, group2)

        p_vec[i] <- p
    }
    # The proportion of p-values in your vector are less than ‘alpha’. 
    power <- sum(p_vec < alpha) / n.boot
    return(power)
}


# sla <- read.csv('sla.csv')
# sig_t(sla, 'l', 'soil', 'C', 'S')
# 0.433438618353941

# boot_power(sla, 'l', 'soil', 'C', 'S', n.boot=1000, alpha=0.5)
# 0.622