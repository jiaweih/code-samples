#
# Predict Age Specific Fertility Rates (ASFR) for age/location/year with 
# random effects model.
#
#
library(lme4)
library(plyr)

# Read observed predictors and target variables
data.past <- read.csv('/ihme/forecasting/data/fertility/inputs/modeling_data.csv')
data.past <- data.past[!(data.past$asfr==0),]

# Read forecasted predictors
edu.forecast <- read.csv('/ihme/forecasting/data/fertility/inputs/edu_forecast.csv')
ldi.forecast <- read.csv('/ihme/forecasting/data/fertility/inputs/ldi_forecast.csv')
ldi.edu <- merge(edu.forecast, ldi.forecast, by = c('location_id', 'year_id'))

hold.out.year <- 2010

regression <- function(formula) {
	# Construct a list of regression formulas for each age group.
	#
	# Args:
	#	formula: The general formula to be used in regression.
	#
	# Returns:
	#	A list of regression formulas for each age group.
	lm_lst <- list()

	for (age_group_id in 8:14) {
    
    train.data <- data.past[(data.past$age_group_id == age_group_id) & 
    						(data.past$year_id < hold.out.year),]
    lm_lst[[length(lm_lst) + 1]] <- formula(train.data)
	    
	}
	return(lm_lst)
}

ldi.edu.country.formula <- function(data) {
	# Linear random effects model by location
	#
	# Args:
	#	data: Dataframe of observed data, containing independent 
	#  		  and dependent variables.
	#
	# Returns:
	# 	Formula of linear random intercept model.
  formula <- lmer('log(asfr) ~ 1 + ldi + edu + (1 | location_id)', data=data)
}


ldi.edu.country.lmer <- regression(ldi.edu.country.formula) 
names(ldi.edu.country.lmer) <- 8:14


get.betas <- function(age_group_id) {
	# Simulate betas from a Multivariate Normal Distribution
	#
	# Args:
	#	age_group_id: 8 to 14
	#
	# Returns:
	#	Dataframe of regression betas for each age_group_id.
	model <- ldi.edu.country.lmer[[as.character(age_group_id)]] 
	betas <- mvrnorm(1000, mu = fixef(model), Sigma = vcov(model))
	betas <- as.data.frame(betas)
	names(betas) <- c('fix.intercept', 'ldi', 'edu')
	betas$age_group_id <- age_group_id
	return(betas)
}

get.random.alphas <- function(age_group_id){
	# Get random effect alphas by country
	# 
	# Args:
	#	age_group_id: 8 to 14
	#
	# Returns:
	# 	Dataframe of random intercepts for each age_group_id.
	#
	model <- ldi.edu.country.lmer[[as.character(age_group_id)]]
	random.alphas <- ranef(model)
	random.alphas <- as.data.frame(random.alphas$location_id)
	random.alphas$location_id <- rownames(random.alphas)
	random.alphas$age_group_id <- age_group_id
	return(random.alphas)
}

random.alphas <- lapply(8:14, get.random.alphas)
random.alphas <- do.call(rbind.data.frame, random.alphas)
names(random.alphas) <- c('rand.alpha', 'location_id', 'age_group_id')
ldi.edu <- merge(ldi.edu, random.alphas, by=c('location_id', 'age_group_id'))

predict.asfr <- function(age_group_id){
	# Predict ASFR using simulated alphas and betas.
	# 
	# Args:
	#	age_group_id: 8 to 14
	# 
	# Returns:
	#	Dataframe of predicted ASFR
	#
	betas <- get.betas(age_group_id)
	ldi.edu.age.specific <- ldi.edu[ldi.edu$age_group_id==age_group_id,]
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

write.csv(predict.results, 
	file='/ihme/forecasting/data/fertility/outputs/predict_results_log_space.csv')
