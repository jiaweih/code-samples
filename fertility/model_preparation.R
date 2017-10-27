#############################
## Stage-one fertility forecasting: data preparation.
#############################
## DRIVE MACROS
rm(list=ls())
source("/home/j/temp/central_comp/libraries/current/r/get_location_metadata.R")
source("/home/j/temp/central_comp/libraries/current/r/get_covariate_estimates.R")
library(ggplot2)
library(magrittr)
library(RColorBrewer)
library(lme4)
library(boot)
library(plyr)
library(parallel)
library(argparse)

parser <- ArgumentParser()
parser$add_argument("--version", help = "Date to specify model run", type = "character")
args <- parser$parse_args()
print(args)
list2env(args, environment()); rm(args)
setwd(paste0("/ihme/forecasting/data/fbd_scenarios_data/forecast/asfr/", version))

## LOAD LOCS TFR AND MATERNAL EDUCATION
locsdf <- get_location_metadata(location_set_id = 22)[level == 3]

## AGE SPECIFIC DATA, model_version_id from Jonathan.
asfrdf <- get_covariate_estimates('ASFR', age_group_id = 8:14, sex_id = 2,
                        location_id = locsdf$location_id, year_id = 1980:2016, model_version_id=13961)
asmedudf <- get_covariate_estimates('education_yrs_pc', age_group_id = 8:14,
                        sex_id = 2, location_id = locsdf$location_id, year_id = 1980:2016, model_version_id=14855)
# New met_demand estimates from Caleb.
ascontradf <- fread("/ihme/forecasting/data/fbd_scenarios_data/past/met_demand/met_demand.csv")

#### Maternal education forecasts from Jonathan/Vinay.
asmedudf.for <- fread("/ihme/forecasting/data/fbd_scenarios_data/forecast/maternal_education/medu_forecasts.csv")
# From met_demand forecasts.
ascontradf.for <- fread("/ihme/forecasting/data/fbd_scenarios_data/forecast/met_demand/best/met_demand_forecasts.csv")
asdatadf.for <- merge(asmedudf.for, ascontradf.for,
                      by = c("location_id", "year_id", "age_group_id"))
asdatadf.for[, sex_id := NULL]

## PREP
asdatadf <- merge(asfrdf[, .(location_id, year_id, age_group_id, age_group_name, asfr = mean_value)], 
                  asmedudf[, .(location_name, location_id, year_id, age_group_id, asmedu = mean_value)],
                  by = c("location_id", "year_id", "age_group_id"))
asdatadf <- merge(asdatadf, ascontradf[, .(location_id, year_id, age_group_id, met_demand)],
                by = c("location_id", "year_id", "age_group_id"))
#######
# From cljm's email about asfr lower/upper bounds
# 15-19 max 0.22 Min 0.0005
# 20-24 max 0.35 Min 0.01
# 25-29 max 0.4 min 0.05
# 30-34 max 0.35 min 0.02
# 35-39 max 0.3 min 0.007
# 40-44 max 0.18 min 0.0005
# 45-49 max 0.10 min 0.00001
asfr.lb <- c(0.0005, 0.01, 0.01, 0.02, 0.007, 0.0005, 0.00001)
asfr.ub <- c(0.22, 0.35, 0.4, 0.35, 0.3, 0.18, 0.10)
age_groups <- seq(8, 14)

## Scale ASFR
for (i in seq_along(age_groups)){
    asdatadf[age_group_id == age_groups[i],
             scaled_asfr := (asfr - asfr.lb[i]) / (asfr.ub[i] - asfr.lb[i])]
}
## Bound scaled ASFR; take logit of ASFR
asdatadf[scaled_asfr > 0.999, scaled_asfr := 0.999][scaled_asfr < 0.001,
         scaled_asfr := 0.001][, logit_asfr := logit(scaled_asfr)]
asdatadf[, decade := round_any(year_id, 10, floor)][, decade := factor(decade, levels = seq(1950, 2010, 10))]
# Add location_id, region_id, super_region_id after merging.
asdatadf <- merge(asdatadf, locsdf[, .(location_id, region_id, super_region_id)], by = "location_id")

## Three knots of maternal education: 6, 10, 14; from cljm.
knots <- c(min(asdatadf$asmedu), 6, 10, 14, max(asdatadf$asmedu))
print(paste0("Min asmedu: ", min(asdatadf$asmedu)))
print(paste0("Max asmedu: ", max(asdatadf$asmedu)))
min.max.asmedu <- data.frame(min_asmedu = min(asdatadf$asmedu),
                             max_asmedu = max(asdatadf$asmedu))
# Saved for later use in produce_draws.py
write.table(min.max.asmedu, 'min_max_asmedu.csv', sep=',')
## Names of pieces as predictors
knots_names <- c('asmedu_below_6', 'asmedu_between_6_10', 'asmedu_between_10_14', 'asmedu_above_14')

## Create pieces; could wrap up in a function for later use
for(i in 1:(length(knots) - 1)) {
    varname <- knots_names[i]
    asdatadf[asmedu >= knots[i] & asmedu < knots[i+1], (varname) := asmedu - knots[i]]
    asdatadf[asmedu >= knots[i+1], (varname) := knots[i+1] - knots[i]]
    asdatadf[is.na(get(varname)), (varname) := 0]
}

### Piecewise for forecast data.
## Use the same min/max of knots as observed data so that the transition
## wll be smooth.
knots <- c(min(asdatadf$asmedu), 6, 10, 14, max(asdatadf$asmedu))
knots_names <- c('asmedu_below_6', 'asmedu_between_6_10', 'asmedu_between_10_14', 'asmedu_above_14')

for(i in 1:(length(knots) - 1)) {
    varname <- knots_names[i]
    asdatadf.for[asmedu >= knots[i] & asmedu < knots[i+1], (varname) := asmedu - knots[i]]
    asdatadf.for[asmedu >= knots[i+1], (varname) := knots[i+1] - knots[i]]
    asdatadf.for[is.na(get(varname)), (varname) := 0]
}


asdatadf.for <- merge(asdatadf.for,
            locsdf[, .(location_id, location_name, region_id, super_region_id)],
            by = "location_id")

asdatadf <- rbind(asdatadf, asdatadf.for, fill=TRUE)
# Save the data for modeling.
fwrite(asdatadf, "asdatadf.csv")

all_locs <- unique(asdatadf$location_name)

# Plot of logit_asfr ~ asmedu
pdf("asfr_maternal_edu.pdf", width = 10, height = 10)
    for (loc in all_locs) {
        plot.data <- asdatadf[location_name == loc,]
        asscatter <- ggplot(data = plot.data, aes(x = asmedu, y=logit_asfr)) +
                geom_point(alpha = .6,) + 
                facet_wrap(~age_group_name) + scale_color_brewer(palette = "Spectral") +
                theme_bw() + labs(x = "Maternal Education", y = "Logit ASFR", color = "") + 
                theme(legend.position = "bottom") + ggtitle(loc)
        print(asscatter)
    }

dev.off()
