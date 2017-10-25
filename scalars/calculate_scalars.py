"""
This script calculates aggregated acause specific PAFs and scalars
"""
import argparse
from collections import defaultdict
import cPickle as pickle
import logging
import numpy as np
import os
import pandas as pd
import xarray as xr

from fbd_core.etl.extraction import subset_and_index, df_to_xr

from plot_tools import plot_scalars
from utils import (get_acause_related_risks, dataframe_to_hdf,
                   read_risk_table_from_db)

from settings import (INPATH_MEDIATION, INPATH_PAF_SET_ONE,
                      INDIR_PAF_FORECAST, INDIR_PAF_PAST,
                      INDIR_VACCINE_PAF, OUTDIR_SCALAR_PAST,
                      OUTDIR_SCALAR_FORECAST, OUTDIR_SCALAR,
                      OUTDIR_AGG_PAF_PAST, OUTDIR_AGG_PAF_FORECAST,
                      OUTDIR_SCALAR_PLOTS, DEMOGRAPHY_COLS,
                      DEMOGRAPHY_INDICES, DEFAULT_YEARS, VACCINE_RISKS,
                      SCALAR_DRAW_PREFIX, PAF_DRAW_PREFIX)


__modname__ = "fbd_research.scalars.calculate_scalars"
logger = logging.getLogger(__modname__)


def bound_zero_one(paf):
    ''' Bound PAFs between 0 and 1.

        Parameters
        ----------
        paf: dataframe

        Returns
        ----------
        paf_bounded: dataframe
    '''
    paf_bounded = paf.fillna(0)
    matrix = paf_bounded.loc[:, PAF_COLS].values
    replacements = ((matrix).T == 1).T
    matrix[replacements] = 0.9999
    paf_bounded.loc[:, PAF_COLS] = matrix
    return paf_bounded


def read_paf(acause, risk, version, date, years):
    """
    Read past or forecast PAF.

    Args:
        acause (str): cause name.
        risk (str): risk name.
        version (str): "past" or "forecast".
        date (str): date str indiciating folder where data comes from.
        years (list[int]): three years for past start, forecast start, and
            forecast end.

    Returns
        paf (pandas.DataFrame): dataframe of PAF.
    """

    try:  # TODO need better flow, and perhaps put the risks in settings.py
        if risk in VACCINE_RISKS:
            infile = os.path.join(INDIR_VACCINE_PAF.format(d=date),
                                  '{}_{}.h5'.format(acause, risk))
        else:
            if version == 'past':
                infile = os.path.join(INDIR_PAF_PAST,
                                      '{}_{}.h5'.format(acause, risk))
            elif version == 'forecast':
                infile = os.path.join(INDIR_PAF_FORECAST.format(d=date),
                                      '{}_{}.h5'.format(acause, risk))
        if not os.path.exists(infile):
            raise ValueError("Path {} doesn't exist! ".format(infile))

        paf = pd.read_hdf(infile, 'data')

    except:
        logger.error("{}, {} Error: "
                     "read_paf broken".format(acause, risk))
        return pd.DataFrame()

    paf = bound_zero_one(paf)
    paf = paf.sort_values(DEMOGRAPHY_COLS).reset_index(drop=True)
    paf = paf[paf['year_id'] >= years[0]]
    return paf


def product_of_mediation(acause, risk, cause_risks):
    """ Return the mediation factor for (acause, risk).

        Parameters
        ----------
        acause: str, acause.
        risk: str, risk.
        cause_risks: risks related to acause.

        Returns
        ----------
        mediation_products: float, mediation products for (acause, risk).
    """
    logger.info("Doing some mediation.")
    # Read the dataframe of mediation for risk factors.
    med = pd.read_csv(INPATH_MEDIATION)
    mediation_factors = med.loc[(med.acause == acause) &
                                (med.mediator == risk) &
                                (med.risk.isin(cause_risks))
                                ]['mean'].values

    if len(mediation_factors):
        mediation_products = np.prod(1 - mediation_factors)
    else:
        # The mediation of each risk factor through itself is
        # assumed to be zero.
        mediation_products = 1.0
    return mediation_products


def get_id_risk(risk_table):
    ''' Return dict of (id, risk) for level 1, 2, 3 risks.
        {104: 'metab', 202: '_env', 203: '_behav'}

        Parameters
        ----------
        risk_table: dataframe with columns of rei, rei_id,
                    path_to_top_parent, level
    '''
    id_risk = dict(risk_table.loc[risk_table.level.isin([1, 2, 3])]
                   [['rei_id', 'rei']].values)
    return id_risk


def get_cluster_risks(cause_risks, id_risk, risk_table):
    ''' Return a dictionary. key: level 1, 2, 3 risk; value: list of sub-risks.
        {'_env': ['air_hap'], 'metab': ['metab_bmi'], '_behav': ['activity']}

        Parameters
        ----------
        cause_risks: list of risks contributing to 'cause'.
        id_risk: dictionary of risk_id, risk.
        risk_table: dataframe with columns of rei, rei_id,
                    path_to_top_parent, level.

    '''
    risk_lst = defaultdict(list)

    for level in [1, 2, 3]:
        for risk in cause_risks:
            try:
                parent_id = int(risk_table.loc[risk_table.rei == risk]
                                ['path_to_top_parent'].values[0].
                                split(',')[level])
                parent_risk = id_risk[parent_id]
                risk_lst[parent_risk].append(risk)
            except:
                continue
    return risk_lst


def aggregate_paf(acause, cause_risks, version, date, years,
                  cluster_risk=None):
    ''' Aggregate PAFs through mediation.

        Args:
            acause (str): acause
            cause_risks (list[str]): risks associated with acause
            version (str): 'past' or 'forecast'
            date (str): indicating folder where data comes from/goes to
            years (list[int]): three years for past start, forecast start, and
                forecast end.
            cluster_risk: whether this is a cluster risk.
                Impacts how it's saved.

        Returns:
            paf_aggregated (pandas.DataFrame): dataframe of aggregated PAF.
    '''
    logger.info('Start aggregating {} PAF:'.format(version))
    # Read the list of risk-acause pairs that are supposed to have PAF of one.
    with open(INPATH_PAF_SET_ONE, 'rb') as f:
        paf_set_one_tuple = pickle.load(f)

    index = 0

    logger.info("Risks: {}".format(cause_risks))

    for risk in cause_risks:
        logger.info('Doing risk: {}'.format(risk))
        # Ignore (acause, risk) if it's in paf_set_one_tuple.
        # So we wouldn't get flat scalars for the acause.
        if (acause, risk) in paf_set_one_tuple:
            logger.info("{}, {} in paf_set_one_tuple".format(acause, risk))
            continue
        paf = read_paf(acause, risk, version, date, years)

        if not len(paf):
            logger.info("len(paf) == 0: {}".format(risk))
            continue

        mediation_prod = product_of_mediation(acause, risk, cause_risks)

        if index == 0:
            logger.info("Index is 0. Starting the paf_prod.")
            paf_prod = paf[DEMOGRAPHY_COLS].copy()
            paf_prod[PAF_COLS] = 1 - paf[PAF_COLS] * mediation_prod
            index += 1
        else:
            logger.info("Index is {}.".format(index))
            paf[PAF_COLS] = 1 - paf[PAF_COLS] * mediation_prod
            paf_prod = paf_prod.merge(paf, on=DEMOGRAPHY_COLS)
            paf_values = paf_prod[PAF_X_COLS].values *\
                paf_prod[PAF_Y_COLS].values
            paf_prod[PAF_COLS] = pd.DataFrame(paf_values)
            keep_cols = DEMOGRAPHY_COLS + PAF_COLS
            paf_prod = paf_prod[keep_cols]
            index += 1

    if index > 0:
        logger.info("We got some pafs.")
        paf_prod[PAF_COLS] = 1 - paf_prod[PAF_COLS]
        paf_aggregated = paf_prod.copy()

        # Cap PAF at 0.9999.
        array = paf_aggregated.loc[:, PAF_COLS].values
        if len(np.where(array > 0.9999)[0]):
            logger.info("Gotta scale some down to 0.9999")
            array[array > 0.9999] = 0.9999
            paf_aggregated.loc[:, PAF_COLS] = array
    else:
        logger.error('No risks available for {}'.format(acause))
        paf_aggregated = pd.DataFrame()
        return paf_aggregated

    paf_mediated = scale_paf(paf_aggregated)
    # TODO move save functions outside of aggregation
    save_paf(paf_mediated, version, acause, date, cluster_risk=cluster_risk)
    return paf_mediated


def generate_scalars_from_aggregated_paf(paf):
    '''Calculate scalars from aggregated PAF.

        Parameters
        ----------
        paf: dataframe of aggregated PAF.

        Returns
        ----------
        scalar: dataframe of scalar.
    '''
    logger.info("Generating scalars from paf.")
    scalar = paf.loc[:, DEMOGRAPHY_COLS].copy()
    scalar_values = 1 / (1 - paf.loc[:, PAF_COLS].values)
    scalar[SCALAR_COLS] = pd.DataFrame(scalar_values, index=scalar.index)
    scalar = scalar.sort_values(DEMOGRAPHY_COLS).reset_index(drop=True)
    return scalar


def scale_paf(paf):
    '''Bound PAF at 0.95 if the maximum cause-specific PAF is over 0.95.
       Multiply PAFs by scale factor 0.95/max(PAF).

       Parameters
       ----------
       paf: dataframe of PAF.

       Returns
       ----------
       scaled_paf: dataframe of scaled PAF.
    '''
    scaled_paf = paf.copy()
    paf_values = paf.loc[:, PAF_COLS].values
    max_paf = np.max(paf_values)

    if max_paf > 0.95:
        scale = 0.95 / max_paf
        paf_values = paf_values * scale
        scaled_paf.loc[:, PAF_COLS] = paf_values
    return scaled_paf


def save_paf(paf, version, acause, date, cluster_risk=None):
    ''' Save mediated PAF at cause level.

        Parameters
        ----------
        paf: dataframe of PAF.
        version: 'past' or 'forecast'.
        cluster_risk: if none, it will be just risk.
    '''
    if version == 'past':
        if cluster_risk is not None:
            outpath =\
                os.path.join(OUTDIR_AGG_PAF_PAST,
                             'risk_acause_specific/{}_{}.h5'.
                             format(acause, cluster_risk.strip('_')))
        else:
            outpath = os.path.join(OUTDIR_AGG_PAF_PAST, '{}.h5'.format(acause))
    elif version == 'forecast':
        if cluster_risk is not None:
            outpath =\
                os.path.join(OUTDIR_AGG_PAF_FORECAST.format(d=date),
                             'risk_acause_specific/{}_{}.h5'.
                             format(acause, cluster_risk.strip('_')))
        else:
            outpath = os.path.join(OUTDIR_AGG_PAF_FORECAST.format(d=date),
                                   '{}.h5'.format(acause))

    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    logger.info("Saving some pafs: {}".format(outpath))
    dataframe_to_hdf(paf, outpath, DEMOGRAPHY_COLS)


def main(acause, date, years=None, update_past=True):
    """
    The mother function that runs scalars calculations

    Args:
        acause (str): cause to compute scalars for
        date (str): date string pointing to folder to pull data from
        years (list[int]): three years for past start, forecast start, and
            forecast end.
        update_past (boolean): whether to update past scalars by overwriting.
    """
    years = years or DEFAULT_YEARS
    risk_table = read_risk_table_from_db()
    id_risk = get_id_risk(risk_table)
    lst_scalar = []
    for version in ['past', 'forecast']:
        logger.info("OH BOY WE'RE DOING THE: {}".format(version))
        if version == 'past':
            outpath_scalar = os.path.join(OUTDIR_SCALAR_PAST,
                                          '{}.nc'.format(acause))
            # We only need to calculate past scalars once.
            if os.path.exists(outpath_scalar) and not update_past:
                scalar = xr.open_dataset(outpath_scalar)
                # we just want the past here:
                scalar = scalar.loc[{'year_id': range(years[0], years[1])}]
                lst_scalar.append(scalar)
                continue
        elif version == 'forecast':
            outpath_scalar =\
                os.path.join(OUTDIR_SCALAR_FORECAST.format(d=date),
                             '{}.nc'.format(acause))

        cause_risks = get_acause_related_risks(acause)
        # Aggregate PAF for level-1 cluster risks
        # We don't need to use the PAF for scalar.
        risk_lst = get_cluster_risks(cause_risks, id_risk, risk_table)

        for key in risk_lst.keys():
            logger.info("In the middle of a for-loop.")
            subrisks = risk_lst[key]
            # If there's only one subrisk and it's itself
            if len(subrisks) == 1 and subrisks[0] == key:
                continue

            logger.info('Start aggregating cluster risk: {}'.format(key))
            _ = aggregate_paf(acause, subrisks, version, date, years,
                              cluster_risk=key)

        # Aggregate PAF for all risks.
        # We need to use the PAF for scalar.
        paf_mediated = aggregate_paf(acause, cause_risks, version, date, years)
        if len(paf_mediated) == 0:
            logger.info("No paf_mediated. Early return.")
            return

        scalar_df = generate_scalars_from_aggregated_paf(paf_mediated)
        # now convert df to xr and save as .nc
        if not os.path.exists(os.path.dirname(outpath_scalar)):
            os.makedirs(os.path.dirname(outpath_scalar))
        # TODO could we deprecate subset_and_index, given the latest df_to_xr?
        subset_df = subset_and_index(scalar_df, index_columns=DEMOGRAPHY_COLS,
                                     index_vals_to_keep=DEMOGRAPHY_INDICES,
                                     draw_prefix='scalar_')
        subset_df.rename({'value': acause}, inplace=True)
        xr_scalar = df_to_xr(subset_df)
        xr_scalar.to_netcdf(outpath_scalar)

        # We just want the future here for forecast data:
        if version == 'forecast':
            xr_scalar = xr_scalar.loc[{'year_id': range(years[1], years[2]+1)}]
        elif version == 'past':
            xr_scalar = xr_scalar.loc[{"year_id": range(years[0], years[1])}]
        else:
            raise Exception(
                    "Version is {}. Can only be 'forecast' or 'past'".format(
                        version))
        lst_scalar.append(xr_scalar.to_dataset())

    logger.info("A big thing: {}".format(lst_scalar))

    # Save past and forecast scalars.
    outpath_scalar = os.path.join(OUTDIR_SCALAR.format(d=date),
                                  '{}.nc'.format(acause))
    scalar_all_years = xr.concat(lst_scalar, dim='year_id')
    logger.info("big thing concatted: {}".format(scalar_all_years))
    scalar_all_years =\
        scalar_all_years.loc[{'year_id':
                              sorted(scalar_all_years['year_id'].values)}]
    logger.info("big thing with sorted years: {}".format(scalar_all_years))
    scalar_all_years.to_netcdf(outpath_scalar)

    logger.info("Saved! Doing plots now.")

    # Plot scalars; maybe put the parameters in settings?
    sex_id = 2
    location_ids = [102]
    scenario = 0
    plot_scalars(scalar_all_years, acause, sex_id, location_ids, scenario,
                 date, OUTDIR_SCALAR_PLOTS,
                 start_age_group_id=10, end_age_group_id=22)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Scalars")
    parser.add_argument("--acause", type=str, required=True,
                        help="It's `A Cause', get it?")
    parser.add_argument("--date", type=str, required=True,
                        help="String denoting file directory. Ex: 2015_03_21")
    parser.add_arg_years()
    parser.add_arg_draws()

    args = parser.parse_args()

    year_args = list(args.years)  # ourgparse arg with default value

    NUMBER_OF_DRAWS = args.draws or 100

    SCALAR_COLS = [SCALAR_DRAW_PREFIX + '{}'.format(i)
                   for i in range(NUMBER_OF_DRAWS)]
    PAF_COLS = [PAF_DRAW_PREFIX + '{}'.format(i)
                for i in range(NUMBER_OF_DRAWS)]
    PAF_X_COLS = [PAF_DRAW_PREFIX + '{}_x'.format(i)
                  for i in range(NUMBER_OF_DRAWS)]
    PAF_Y_COLS = [PAF_DRAW_PREFIX + '{}_y'.format(i)
                  for i in range(NUMBER_OF_DRAWS)]

    logging.basicConfig(level=logging.INFO)
    logger.debug("Arguments {}".format(args))

    to_update_past = True  # O M G

    main(args.acause, args.date, years=year_args, update_past=to_update_past)

    logger.debug("Exit from script")
