import os
import sys
sys.path.append('/homes/jiaweihe/fbd_scenarios/')
import logging
import pandas as pd
import xarray as xr
import numpy as np

from fbd_scenarios import forecast_methods
import settings
import fbd
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from fbd_core.db import get_ages, get_modeled_locations


__modname__ = "fbd.demography.fertility"
logger = logging.getLogger(__modname__)
sns.despine()
sns.set_style('ticks')


def get_met_demand(met_demand_file):
    ''' Read and clean met_demand data.
    '''
    df_met_demand = \
        pd.read_csv(met_demand_file)\
          .loc[:, ['location_id', 'year_id', 'age_group_id', 'met_demand']]\
          .drop_duplicates(['location_id', 'year_id', 'age_group_id'])
    return df_met_demand


def met_demand_to_array(df_met_demand):
    ''' Xarray dataset to dataarray.
    '''
    da_met_demand = xr.Dataset\
                    .from_dataframe(df_met_demand.set_index(
                        ['location_id', 'year_id', 'age_group_id']))\
                    .to_array()
    return da_met_demand


def forecast_met_demand(da_met_demand):
    ''' Forecast met_demand using arc_quantiles methods.
    '''
    # To avoid the error "ValueError: conflicting sizes for dimension
    # 'age_group_id': length 755 on the data but length 7 on coordinate
    # 'age_group_id'"".
    da_met_demand = da_met_demand.transpose('variable', 'year_id',
                                            'age_group_id', 'location_id')
    forecasts_met_demand = forecast_methods.arc_quantiles(da_met_demand, \
                            years=[1990, 2017, 2041], weight_exp=0.4)\
                                           .mean('draw')

    return forecasts_met_demand


def save_met_demand(met_demand, outfile):
    ''' Save met_demand in past and forecast directory;
        Convert xarray to dataframe for fertility forecasts.
    '''
    past_met_demand = met_demand.loc[{'year_id': xrange(1990, 2017)}]
    past_met_demand.to_netcdf(os.path.join(settings.MET_DEMAND_PAST_DIR,
                                           'met_demand.nc'))
    forecasts_met_demand = met_demand.loc[{'year_id': xrange(2017, 2041)}]
    if not os.path.exists(settings.MET_DEMAND_FORECAST_DIR):
        os.makedirs(settings.MET_DEMAND_FORECAST_DIR)
    forecasts_met_demand.to_netcdf(os.path.join(settings.MET_DEMAND_FORECAST_DIR,
                                           'met_demand.nc'))
    forecasts_met_demand = forecasts_met_demand.to_dataframe()\
          .reset_index()\
          .rename(columns={'value': 'met_demand'})\
          .loc[:, ['location_id', 'age_group_id', 'year_id', 'met_demand']]
    forecasts_met_demand.to_csv(outfile, index=False)


def plot_met_demand(forecasts):
    ''' Plot forecasted met_demand.
    '''
    loc_map = get_modeled_locations().\
        set_index('location_id')['location_name'].to_dict()
    age_map = get_ages().set_index('age_group_id')['age_group_name'].to_dict()
    locations = [l for l in forecasts.location_id.values if l in loc_map.keys()]
    outfile = os.path.join(settings.MET_DEMAND_FORECAST_DIR, 'graphs/met_demand.pdf')
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    with PdfPages(outfile) as pp:
        for location_id in locations:
            location = loc_map[location_id]
            nrow = 3
            ncol = 3
            fig = plt.figure(figsize=(12, 10))
            grid = gridspec.GridSpec(nrow, ncol)
            for ix, age_group_id in enumerate(np.arange(8, 15)):
                ax = fig.add_subplot(grid[ix])
                tmp = forecasts.loc[{'location_id': location_id, 'age_group_id': age_group_id}]
                tmp_past = tmp.loc[{'year_id': np.arange(1990, 2017)}]
                tmp_future = tmp.loc[{'year_id': np.arange(2017, 2041)}]
                ax.plot(tmp_past.year_id, tmp_past.values[0], color='b')
                ax.plot(tmp_future.year_id, tmp_future.values[0], color='g')
                ax.text(0.7, 0.95, age_map[age_group_id],
                        verticalalignment='top',
                        horizontalalignment='left', transform=ax.transAxes,
                        color='black', fontsize=12)
            fig.suptitle(location, fontsize=15)
            pp.savefig(bbox_inches='tight')



def main():
    logging.info("Getting met_demand")
    # From Caleb
    met_demand_file = "/ihme/forecasting/data/fbd_scenarios_data/past/"\
                      "met_demand/met_demand.csv"
    df_met_demand = get_met_demand(met_demand_file)
    da_met_demand = met_demand_to_array(df_met_demand)
    logging.info("Forecasting met_demand")
    forecasts = forecast_met_demand(da_met_demand)
    logging.info("Saving met_demand")
    outfile = '/ihme/forecasting/data/fbd_scenarios_data/'\
              'forecast/met_demand/best/met_demand_forecasts.csv'
    save_met_demand(forecasts, outfile)
    logging.info("Plotting met_demand")
    plot_met_demand(forecasts)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
    logger.debug("Exit from script")