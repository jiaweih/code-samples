'''
This script aims to calculate risk-acause specific PAFs
Inputs:
0. Command line arguments: a single set of (acause, DATE, sex_id)
1. forecasting database: get the relationship between cause and risk
2. SEV: '/ihme/forecasting/data/sev/{risk}/{sex_id}/best'
3. rrmax: "{rrmax_dir}/{rrmax_file}" where
    rrmax_dir = '/ihme/forecasting/data/gbd2013_riskfactors_updated/rrmax/'
    rrmax_file = '{risk}_{sex_id}.hdf'
    OR
    rrmax_dir = '/ihme/forecasting/data/rr_max'
4. complete index:
    '/ihme/forecasting/ref/jiawei/complete_index.csv'

Outputs:
PAF:
    outpath_paf = '/ihme/forecasting/data/cluster_sev/{DATE}/paf/'\
    '{acause}_{risk}_{sex_id}.hdf'
'''
import logging
import os
import pandas as pd
import xarray as xr
from fbd_core import argparse
from fbd_core.db import get_cause_id
from fbd_core.demog.draws import truncate_draws
from fbd_core.etl.transformation import resample

from utils import (get_acause_related_risks, get_modeling_risks,
                   dataframe_to_hdf, get_whole_index, xarray_to_dataframe)

from settings import (DRAW_PREFIX, INDIR_SEV, INDIR_RRMAX,
                      OUTDIR_PAF_FORECAST, OUTDIR_PAF_PAST, DEMOGRAPHY_COLS,
                      RR_MAX_DRAW_PREFIX, DEFAULT_YEARS, VACCINE_RISKS)


__modname__ = "fbd_research.scalars.calculate_pafs"
logger = logging.getLogger(__modname__)


def read_xarray_sev(risk, date):
    """
    Read SEV in an xarray format.

    Args:
        risk (str): risk name.
        date (str): date str indicating the folder where data comes from.

    Returns:
        ds (xarray.Dataset): contains sev values, indexed by demography dims.
    """
    inpath = os.path.join(INDIR_SEV.format(d=date), '{}.nc'.format(risk))
    # We need to use open_dataset if there are more than one variable, like
    # summary data (mean, median, lower, upper).
    ds = xr.open_dataset(inpath)

    num_of_draws_in = len(ds.coords["draw"])
    if num_of_draws_in != NUMBER_OF_DRAWS:
        da_name = ds.data_vars.keys()[0]
        da = resample(ds[da_name], NUMBER_OF_DRAWS)
        ds = da.to_dataset()

    return ds


def get_past_sev(risk, date, years=None):
    """
    TODO: What does this function do?

    Args:
        risk (str): name of risk to get past sev data for.
        date (str): date indicating the data folder to pull data from.
        years (list[int]): three years for past start, forecast start, and
            forecast end.

    Returns:
        pandas.DataFrame: past sev data for the risk.
    """
    years = years or DEFAULT_YEARS
    ds = read_xarray_sev(risk, date)
    ds = ds.loc[{'year_id': range(years[0], years[1])}]
    raw_sev =\
        xarray_to_dataframe(ds, DEMOGRAPHY_COLS, DRAW_PREFIX, NUMBER_OF_DRAWS)
    full_index = get_whole_index("past", years)
    sev = full_index.merge(raw_sev, on=DEMOGRAPHY_COLS, how='left')
    return sev


def get_sev(risk, version, date, years=None):
    """Get SEV data for past or future for the given risk.

    Args:
        risk (str): risk to get sev data for.
        version (str): past or future.
        date (str): date indicating the data folder to pull data from.
        years (list[int]): three years for past start, forecast start, and
            forecast end.

    Returns:
        pandas.DataFrame: SEV data.
    """
    years = years or DEFAULT_YEARS
    if version == 'past':
        year_ids = range(years[0], years[1])
    elif version == 'forecast':
        year_ids = range(years[1], years[2]+1)
    else:
        raise ValueError("Version should be 'past' or 'forecast'.")
    ds = read_xarray_sev(risk, date)
    ds = ds.loc[{'year_id': year_ids}]
    raw_sev =\
        xarray_to_dataframe(ds, DEMOGRAPHY_COLS, DRAW_PREFIX, NUMBER_OF_DRAWS)
    # To get the whole sets of demographies.
    full_index = get_whole_index(version, years)
    # Merge SEV with complete demographies,
    # so no demography will be missing.
    sev = full_index.merge(raw_sev, on=DEMOGRAPHY_COLS, how='left')
    return sev


def get_rrmax(risk, cause_id):
    """ Get rrmax.

        Parameters
        ----------
        risk_id: int
        cause_id: int
        version: str, 'past' or 'forecast'

        Returns
        ----------
        df: dataframe of rrmax.
    """
    inpath = os.path.join(INDIR_RRMAX, '{}.h5'.format(risk))
    df = pd.read_hdf(inpath)
    df = df.loc[(df.cause_id == cause_id)] \
           .drop(['risk_id', 'cause_id'], axis=1)
    # Set rrmax below 1 to 1.001.
    rr_cols = ['rr_{i}'.format(i=i) for i in range(NUMBER_OF_DRAWS)]
    values = df.loc[:, rr_cols].values
    values[values < 1] = 1.001
    df.loc[:, rr_cols] = values
    return df


def merge_sev_rrmax(sev, rr_max):
    """ Merge SEV and rrmax.

        Parameters
        ----------
        sev: dataframe of SEV.
        rr_max: dataframe of RR(max).

        Returns
        ----------
        sev_rr_max: dataframe of merged SEV and RR(max) draws.
    """

    # For metab risks, rrmax are age/sex specific.
    share_cols = ['location_id', 'age_group_id', 'sex_id', 'year_id']

    for col in share_cols:
        if col not in rr_max.columns:
            share_cols.remove(col)

    sev_rr_max = pd.merge(sev, rr_max, on=share_cols, how='outer')
    # We can replace NA with 0 because these will not affect the result.
    sev_rr_max[SEV_COLS] = sev_rr_max[SEV_COLS].fillna(0, axis=1)
    # Filling rr_max's NA with 1 will also ensure calculation runs
    # and the 1 will not change the result.
    sev_rr_max[RR_MAX_COLS] = sev_rr_max[RR_MAX_COLS].fillna(1, axis=1)
    # sev_rr_max = sev_rr_max.loc[~ sev_rr_max.year_id.isnull()]
    sev_rr_max = sev_rr_max.sort_values(DEMOGRAPHY_COLS)
    return sev_rr_max


def save_paf(risk, acause, df_paf, version, date):
    ''' Save PAF under the directory specified in settings.py

        Parameters
        ----------
        risk: str
        acause: str
        df_paf: dataframe of PAF
        version: "past" or "forecast"
        date: the version of this run
    '''
    if version == 'past':
        outdir = OUTDIR_PAF_PAST
    elif version == 'forecast':
        outdir = OUTDIR_PAF_FORECAST.format(d=date)
    outpath_paf = os.path.join(outdir,
                               '{acause}_{risk}.h5'.format(acause=acause,
                                                           risk=risk))
    if not os.path.exists(os.path.dirname(outpath_paf)):
        os.makedirs(os.path.dirname(outpath_paf))

    dataframe_to_hdf(df_paf, outpath_paf, DEMOGRAPHY_COLS)
    logger.info('{} pafs saved {}'.format(version, outpath_paf))


def calculate_paf(risk, acause, version, date, years=None):
    """Calculate PAF for (risk, acause).

    Args:
        risk (str): risk to calculate paf for.
        acause (str): acause to calculate paf for.
        version (str): "past" or "forecast".
        date (str): date string indicating the folder to pull data from.
        years (list[int]): three years for past start, forecast start, and
            forecast end.

    Returns:
        pandas.DataFrame: PAF data.
    """
    years = years or DEFAULT_YEARS
    cause_id = get_cause_id(acause)
    sev = get_sev(risk, version, date, years)
    rr_max = get_rrmax(risk, cause_id)
    sev_rr_max = merge_sev_rrmax(sev, rr_max)
    sev_values = sev_rr_max[SEV_COLS].values
    rrmax_values = sev_rr_max[RR_MAX_COLS].values
    # Calculate scalars.
    scalar = sev_values * (rrmax_values - 1) + 1
    scalar = truncate_draws(scalar)

    # Calculate PAF.
    paf = 1 - 1.0/scalar
    # Save PAF.
    df_paf = sev_rr_max.loc[:, DEMOGRAPHY_COLS].copy()
    paf_draws = ['paf_{draw}'.format(draw=i) for i in range(NUMBER_OF_DRAWS)]
    df_paf[paf_draws] = pd.DataFrame(paf, index=df_paf.index)

    # Sort columns and reset index
    df_paf = df_paf.sort_values(DEMOGRAPHY_COLS) \
                   .reset_index(drop=True) \
                   .drop_duplicates(DEMOGRAPHY_COLS)
    return df_paf


def main(acause, date, years=None):
    """
    TODO: what does this script do?

    Args:
        acause (str): the cause whcih we are doing this thing to.
        date (str): not sure, but I think this is the version string?
        years (list[int]): three years for past start, forecast start, and
            forecast end.
    """
    years = years or DEFAULT_YEARS

    # All risks contributing to acause.
    risks = get_acause_related_risks(acause)
    modeling_risks = get_modeling_risks()
    vaccine_risks = VACCINE_RISKS

    for risk in risks:
        # PAFs of vaccine_risks are calculated differently.
        if risk in vaccine_risks:
            continue
        elif risk in modeling_risks:
            outpath_paf_past =\
                os.path.join(OUTDIR_PAF_PAST,
                             '{acause}_{risk}.h5'.format(acause=acause,
                                                         risk=risk))
            # If past PAF exists, we don't need to recalculate.
            if not os.path.exists(outpath_paf_past):
                version = 'past'
                df_paf = calculate_paf(risk, acause, version, date, years)
                save_paf(risk, acause, df_paf, version, date)

            version = 'forecast'
            df_paf = calculate_paf(risk, acause, version, date, years)
            save_paf(risk, acause, df_paf, version, date)
        else:
            logger.error('No {risk} available for {acause}'.
                         format(risk=risk, acause=acause))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate PAFS")
    parser.add_argument("--acause", type=str, required=True,
                        help="It's `A Cause', get it?")
    parser.add_argument("--date", type=str, required=True,
                        help="String denoting file directory. Ex: 2015_03_21")
    parser.add_arg_years()
    parser.add_arg_draws()

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.debug("arguments {}".format(args))

    year_args = list(args.years)  # ourgparse arg with default value

    NUMBER_OF_DRAWS = args.draws or 100

    SEV_COLS = [DRAW_PREFIX + '{}'.format(i) for i in range(NUMBER_OF_DRAWS)]
    RR_MAX_COLS = [RR_MAX_DRAW_PREFIX + '{}'.format(i)
                   for i in range(NUMBER_OF_DRAWS)]

    main(args.acause, args.date, year_args)
    logger.debug("exit from script")
