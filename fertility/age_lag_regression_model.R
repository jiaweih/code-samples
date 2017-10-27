rm(list=ls())

library(ggplot2)
library(magrittr)
library(RColorBrewer)
library(lme4)
library(boot)
library(plyr)
library(parallel)
library(data.table)
library(argparse)
library(mvtnorm)

parser <- ArgumentParser()
parser$add_argument("--age", help = "Age for regression",
                    type = "character")
parser$add_argument("--year_id", help = "For age group 8 only; year_id for regression",
                    type = "character")
parser$add_argument("--version", help = "Version to specify model runs.",
                    type = "character")
args <- parser$parse_args()
print(args)

list2env(args, environment()); rm(args)
age <- as.integer(age)
year_id <- as.integer(year_id)
setwd(paste0("/ihme/forecasting/data/fbd_scenarios_data/forecast/asfr/", version))


coreNum <- function(p = 1) {
    # Get slots
    if (interactive()) {
        slots <- readline(prompt="Slots in use: ")
    } else slots <- strtoi(Sys.getenv("NSLOTS"))
    slots <- 20
    slots <- as.numeric(slots)
    
    # Assign cores
    if (grepl("Intel", system("cat /proc/cpuinfo | grep 'name'| uniq", intern=TRUE))) {
        cores <- slots * 0.86
    } else {
        cores <- slots * 0.64
    }
    use_cores <- round(cores / p, 0)
    return(use_cores)
}

asformlogit <- "logit_asfr ~ asmedu_below_6 + asmedu_between_6_10 + 
                    asmedu_between_10_14 + asmedu_above_14 + lag_logit_asfr + met_demand" %>% as.formula
# For age_group 8, 9, don't use spline. The lag_logit_asfr for age_group 8 is period lag.
asformlogit.no.spline <- "logit_asfr ~ asmedu + lag_logit_asfr + met_demand" %>% as.formula
asformlogit.no.met_demand <- "logit_asfr ~ asmedu + lag_logit_asfr" %>% as.formula


if (age %in% c(8)) {
    print("In loop")
    asdatadf <- fread("asdatadf.csv")
    asdatadf <- asdatadf[age_group_id == age]
    asdatadf <- asdatadf[asdatadf$year_id > 1989]
    lag.asfr <- fread(paste0('asfr_lag_', age, '_', year_id, '.csv'))
    asdatadf <- merge(asdatadf, lag.asfr, by=c("location_id", "year_id", "age_group_id"))
    asdatadfs <- split(asdatadf, by = "age_group_id")
    asmodelraw <- mclapply(asdatadfs, function(df) {
        model <- lm(formula = asformlogit.no.spline, data = df)
    }, mc.cores = coreNum())
    ages <- unique(asdatadf$age)
    print(ages)

    asmodels <- mclapply(1:length(asdatadfs), function(i){
        df <- asdatadfs[[i]]
        model <- asmodelraw[[i]]
        df <- df[, pred := predict(model, newdata=df)]
        df <- df[, resid := logit_asfr - pred]
        message(paste0("AGID: ", unique(df$age_group_id)))
        cat("\n\n")
        return(df)
    }, mc.cores = coreNum()) %>% rbindlist(use.names=T, fill=T)
    fwrite(asmodels, paste0('lag_asmodels_', age, '_', year_id, '.csv'))
    if(year_id == 2036) {
        fwrite(asmodels, paste0('lag_asmodels_', age, '.csv'))
    }

} else {
    # Covariate data for all years.
    asdatadf <- fread(paste0('asdatadf.csv'))
    asdatadf[, lag_logit_asfr := NULL]
    asdatadf <- asdatadf[age_group_id == age]
    # cohort lag data; from age_lag_asfr.py
    lag.asfr <- fread(paste0('asfr_lag_', age, '.csv'))
    asdatadf <- merge(asdatadf, lag.asfr, by=c("location_id", "year_id", "age_group_id"))
    asdatadf <- asdatadf[year_id > 2016, logit_asfr := NA]
    asdatadfs <- split(asdatadf, by = "age_group_id")

    asmodelraw <- mclapply(asdatadfs, function(df) {
        if (unique(df$age_group_id) == 9) {
            model <- lm(formula = asformlogit, data = df)
        } else {
            model <- lm(formula = asformlogit, data = df)
        }
    }, mc.cores = coreNum())

    asmodels <- mclapply(1:length(asdatadfs), function(i){
    df <- asdatadfs[[i]]
    model <- asmodelraw[[i]]
    df <- df[, pred := predict(model, newdata=df)]
    df <- df[, resid := logit_asfr - pred]
    return(df)
    }, mc.cores = coreNum()) %>% rbindlist(use.names=T, fill=T)

    fwrite(asmodels, paste0('lag_asmodels_', age, '.csv'))

}

print("Write summary")
logfile <- file(paste0("summary_", age, ".txt"), open="wt")
sink(logfile)
sink(logfile, type="message")
ages <- unique(asdatadf$age_group_id)


for(i in 1:length(asmodelraw)){
    m <- asmodelraw[[i]]
    age_group <- ages[i]
    print("Age Group ID")
    print(age_group)
    print(summary(m))
}

sink()
sink(type="message")

print("Write coefficients")
for(i in 1:length(asmodelraw)){
    m <- asmodelraw[[i]]
    # Save coefficients and vcov for later draws generation
    coeff.file <- paste0("coefficents_", age, ".csv")
    write.csv(m$coefficients, coeff.file)
    vcov.file <- paste0("vcov_", age, ".csv")
    write.csv(vcov(m), vcov.file)
}