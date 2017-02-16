# Forecast age-specific-fertility-rates (ASFR) with linear mixed-effects model.
# Fixed effects: ldi, edu; random effects: location.
# LDI is defined as lag distributed income per capita: gross domestic product 
# per capita that has been smoothed over the preceding 10 years.
library(lme4)
library(plyr)

# Read and clean the data.
data <- read.csv('/ihme/forecasting/data/fertility/inputs/modeling_data.csv')
data <- data[!(data$asfr==0),]
mean.age <- read.csv('/ihme/forecasting/data/fertility/inputs/mean_fertile_age_by_population.csv')
mean.age <- rename(mean.age,c('year' = 'year_id'))
data <- merge.data.frame(x = data, y = mean.age, by = c('location_id', 'year_id'))
data$sex <- NULL
data$iso3 <- NULL

edu_forecast = read.csv('/ihme/forecasting/data/fertility/inputs/edu_forecast.csv')
ldi_forecast = read.csv('/ihme/forecasting/data/fertility/inputs/ldi_forecast.csv')
ldi.edu <- merge(edu_forecast, ldi_forecast, by = c('location_id', 'year_id'))

hold.out.year <- 2000

regression <- function(formula) {
	# Fit the model for each age_group_id and return a list of fitted model.
    lm_lst <- list()
    
    for (age_group_id in 8:14) {
        
        train.data <- data[(data$age_group_id == age_group_id) & (data$year_id < hold.out.year), ]
        lm_lst[[length(lm_lst) + 1]] <- formula(train.data)
        
    }
    return(lm_lst)
}

ldi.edu.country.formula <- function(data) {
	# Return the model formula.
    formula <- lmer('log(asfr) ~ 1 + ldi + edu + (1 | location_id)', data=data)
    return(formula)
}

ldi.edu.country.lmer <- regression(ldi.edu.country.formula) 
names(ldi.edu.country.lmer) <- 8:14

get_betas <- function(age_group_id) {
	# Simulate betas from a Multivariate Normal Distribution
	model <- ldi.edu.country.lmer[[as.character(age_group_id)]] 
	betas <- mvrnorm(1000, mu = fixef(model), Sigma = vcov(model))
	betas <- as.data.frame(betas)
	names(betas) <- c('fix.intercept', 'ldi', 'edu')
	betas$age_group_id <- age_group_id
	return(betas)
}

get_random_alphas <- function(age_group_id){
	# Get random effect alphas by countries
	model <- ldi.edu.country.lmer[[as.character(age_group_id)]]
	random.alphas <- ranef(model)
	random.alphas <- as.data.frame(random.alphas$location_id)
	random.alphas$location_id <- rownames(random.alphas)
	random.alphas$age_group_id <- age_group_id
	return(random.alphas)
}

random.alphas <- lapply(8:14, get_random_alphas)
random.alphas <- do.call(rbind.data.frame, random.alphas)
names(random.alphas) <- c('rand.alpha', 'location_id', 'age_group_id')
ldi.edu <- merge(ldi.edu, random.alphas, by=c('location_id', 'age_group_id'))

predict.asfr <- function(age_group_id){
	# Predict ASFR using simulated alphas and betas
	betas <- get_betas(age_group_id)
	ldi.edu.age.specific <- ldi.edu[ldi.edu$age_group_id==age_group_id, ]
	ldi.intercept <- betas$ldi * ldi.edu.age.specific[, paste0('edu_forecast_draw', as.character(0:999))]
	edu.intercept <- betas$edu * log(ldi.edu.age.specific[, paste0('ldipc_', as.character(0:999))])
	fix.intercept <- betas$fix.intercept
	ran.intercept <- ldi.edu.age.specific[, c('rand.alpha')]
	predict.results <- ldi.intercept + edu.intercept + fix.intercept + ran.intercept
	names(predict.results) <- paste0('asfr_draw_', as.character(0:999))
	indexes <- ldi.edu.age.specific[, c('location_id', 'age_group_id', 'year_id')]
	predict.results <- cbind(indexes, predict.results)
	return(predict.results)
}

predict.results <- lapply(8:14, predict.asfr)
predict.results <- do.call(rbind.data.frame, predict.results)

write.csv(predict.results, file='/ihme/forecasting/data/fertility/outputs/predict_results_log_space.csv')
