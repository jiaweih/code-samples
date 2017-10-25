import os
import pandas as pd
import xarray as xr
from fbd_core.db import db_engine, get_cause_risk_pairs
from settings import (RISKS_NOT_AVAILABLE, DEMOGRAPHY_INDICES, DEFAULT_YEARS,
                      FORECAST_RISK_ATTRIBUTABLE_DIR,
                      PAST_RISK_ATTRIBUTABLE_DIR)

# TODO wholesale outsource subsequent methods to fbd_core.
# TODO can we minimize all this xr-to-df-to-xr conversion bs?


def get_acause_related_risks(acause):
    '''Return a list of risks contributing to certain acause.
    '''
    if acause in ['rotavirus']:  # TODO refactor this mapping to settings.py?
        risks = ['rota']
    else:
        df_acause_risk = get_cause_risk_pairs()
        risks = list(df_acause_risk.loc[df_acause_risk.acause == acause]['rei']
                                   .unique())
    return [risk for risk in risks if risk not in RISKS_NOT_AVAILABLE]


def get_risk_related_acauses(rei):
    '''Return a list of acauses associated with certain risk.
    '''
    df_acause_risk = get_cause_risk_pairs()
    acauses = list(df_acause_risk.loc[df_acause_risk.rei == rei]['acause']
                                 .unique())
    return acauses


def get_modeling_causes():
    '''Return the causes we are modeling
    '''
    df_acause_risk = get_cause_risk_pairs()
    acauses = list(df_acause_risk.acause.unique())
    return acauses


def get_modeling_risks():
    '''Return risks we are modeling
    '''
    df_acause_risk = get_cause_risk_pairs()
    modeling_risks = list(df_acause_risk.rei.unique())

    for risk in RISKS_NOT_AVAILABLE:
        if risk in modeling_risks:
            modeling_risks.remove(risk)
    return modeling_risks


def read_risk_table_from_db():  # TODO move to fbd_core.db?
    '''Query risks from forecasting rei table.
    '''
    engine = db_engine('fbd-dev-read', database='forecasting')
    query = '''select rei_id, rei, path_to_top_parent, level
               from forecasting.risks;'''
    df_risk = pd.read_sql_query(query, engine)
    return df_risk


def dataframe_to_hdf(df, outpath, demography_cols, key='data', mode='w'):
    '''Save dataframe to disk as HDF format.'''
    df.to_hdf(outpath,
              mode=mode,
              key=key,
              data_columns=demography_cols,
              format='table',
              complib='blosc',
              complevel=9)
    return


def get_whole_index(version="all", years=None):
    """
    Gets a dataframe with a complete set of modeling demographics.

    Args:
        version: "all", "past", or "forecast".
        years (list[int]): three years for past start, forecast start, and
            forecast end.

    Returns:
        pandas.DataFrame: a dataframe with a complete set of modeling
            demographics.
    """
    years = years or DEFAULT_YEARS
    whole_index = pd.DataFrame(index=pd.MultiIndex.
                               from_product(DEMOGRAPHY_INDICES.values(),
                                            names=DEMOGRAPHY_INDICES.keys()))
    whole_index.reset_index(inplace=True)
    if version == 'past':
        whole_index =\
            whole_index.loc[whole_index.year_id < years[1]]
    elif version == 'forecast':
        whole_index =\
            whole_index.loc[whole_index.year_id >= years[1]]
    return whole_index


def get_risk_attributable_mortality(acause, risk):
    """
    Reads already-computed past/future risk-attributable outputs,
    combines them, and returns an xarray object.

    Args:
        acause (str): name of acause
        risk (str): name of risk

    Returns:
        xarray.Dataset: combined past/future risk-attributable data.
    """
    attri_future =\
        xr.open_dataset(os.path.join(FORECAST_RISK_ATTRIBUTABLE_DIR,
                        '{}_{}.nc'.format(acause, risk)))
    attri_past =\
        xr.open_dataset(os.path.join(PAST_RISK_ATTRIBUTABLE_DIR,
                        '{}_{}.nc'.format(acause, risk)))
    attri_past = attri_past.drop(['acause', 'measure'])
    attri = xr.concat([attri_past, attri_future], dim='year_id')
    return attri


def xarray_to_dataframe(ds, cols, draw_prefix, num_draws):
    """
    Converts xarray to dataframe.
    It is assumed that a draw-like dimension exists in the input xarray object.

    Args:
        ds (xarray.Dataset): input xarray Dataset
        cols (list of str): columns used for indexing upon calling .pivot_table
        draw_prefix (str): name prefix of the draw dimension in xarray object.
            Example: "draw_", "sev_", "paf_", "paf_x_"
            (underscore always assumed)
        num_draws (int): number of labels along the draw dimension

    Returns:
        Pandas.DataFrame
    """
    df = ds.to_dataframe().reset_index()
    # the draw column name
    draw_dim_name =\
        ["_".join(draw_prefix.split("_")[0:-1])]  # "paf_x_" -> ["paf_x"]
    # The data variable name
    data_var_name = str(ds.data_vars.keys()[0])

    df = pd.pivot_table(df,
                        values=data_var_name,
                        index=cols,
                        columns=draw_dim_name)\
           .reset_index()\
           .rename(columns={i: draw_prefix + '{}'.format(i)
                            for i in range(num_draws)})
    return df
