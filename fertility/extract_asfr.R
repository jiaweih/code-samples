# Extract the outputs of bayesPop, trajectories of 
# Age Specific Fertility Rates (ASFR), for population forecasting.
# Outputs are stored as arrays in .rda format by countries. This script would
# convert the data to dataframe and save them as csv files.

array.to.dataframe <- function(asfert, country.code) {
  # Convert ASFR in array format to dataframe
  # Args:
  #   asfert: Arrays of ASFR data
  #   country.code: Integer code indicating each country
  #
  # Returns:
  #   Dataframe of ASFR draws.
  # age_group_id 8-14
  dimnames(asfert)[[1]] <- 8:14
  # 1000 draws
  dimnames(asfert)[[3]] <- 0:999
  df <- expand.grid(dimnames(asfert))
  names(df) <- c("age_group_id", "year_id", "draws")
  df$country.code <- c(country.code)
  df$asfert <- c(asfert)
  # convert births per women in five years to births per 1000 women in one year
  df$asfert <- df$asfert * 1000 / 5
  return(df)
}


indir <- "/ihme/forecasting/data/fertility/bayes_pop/pop_projections/predictions"
outdir <- "/ihme/forecasting/data/fertility/bayes_asfr"
my_files <- list.files(indir)
# Get files like vital_events_*, which contain age-specific fertility rates.
vital_files <- grep("vital_events_", my_files, value=T)
# Get the country codes.
names(vital_files) <- gsub("[^0-9]", "", vital_files)

for (i in 1:length(vital_files)) {
  file <- vital_files[i]
  country.code <- names(file)

  inpath <- file.path(indir, file)
  load(inpath)
  df <- array.to.dataframe(asfert, country.code)

  outpath <- file.path(outdir, paste0('asfr_', country.code, '.csv'))
  write.csv(df, file=outpath, row.names=FALSE)
}