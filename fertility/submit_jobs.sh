#!/bin/bash
python_shell=/ihme/forecasting/envs/jiawei_conda/bin/python
r_shell="/share/local/singularity/bin/singularity exec \
          /share/singularity-images/rstudio/ihme_rstudio_3401.img \
          /usr/local/bin/Rscript"
DRAW=1000
version="last_run"
$r_shell model_preparation.R --version $version

for year in 2016 2021 2026 2031 2036
do
    echo $year
    echo "age_lag_asfr.py \n"
    $python_shell age_lag_asfr.py --age_group_id 8 --year_id $year --version $version
    echo "age_lag_regression_model.R \n"
    $r_shell age_lag_regression_model.R --age 8 --year_id $year --version $version
    echo "arima_model.py \n"
    $python_shell arima_model.py --age_group_id 8 --year_id $year --draws $DRAW --version $version
    echo "add_arima_residuals.R \n"
    $r_shell add_arima_residuals.R --age_group_id 8 --year_id $year --version $version 
done

for age_group_id in {9..14}
do
    echo "age_lag_asfr.py \n"
    $python_shell age_lag_asfr.py --age_group_id $age_group_id --version $version
    echo $age_group_id
    echo "age_lag_regression_model.R \n"
    $r_shell age_lag_regression_model.R --age $age_group_id --version $version
    echo "arima_model.py \n"
    $python_shell arima_model.py --age_group_id $age_group_id --draws $DRAW --version $version
    echo "add_arima_residuals.R \n"
    $r_shell add_arima_residuals.R --age_group_id $age_group_id --version $version
done

$python_shell produce_draws.py --draws $DRAW --version $version
$r_shell draws_extend_asfr_age_groups.R --version $version
$python_shell save_plot_fertility.py --version $version
$python_shell arc_quantiles_tfr.py --version $version
# Another new script is supposed to be here; after we got TFR scenarios, we 
# need to get ASFR scenarios and combine them with ASFR reference produced
# in produce_draws.py; save them for cohort component model.