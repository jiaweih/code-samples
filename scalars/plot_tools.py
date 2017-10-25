"""
A module that stores all the scalars-related plotting tools.
"""
import logging
import os

import numpy as np

from fbd_core.db import get_ages, get_modeled_locations
from fbd_core.demog.construct import get_gbd_demographics

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


__modname__ = "fbd_research.scalars.plot_tools"
logger = logging.getLogger(__modname__)


# TODO this needs to be refactored, and possibly moved to fbd_core


def plot_scalars(scalar_ds, acause, sex_id, location_ids, scenario, date,
                 outdir, start_age_group_id=10, end_age_group_id=22):
    ''' Plot scalars with uncertainties.

        Parameters
        ----------
        scalar_ds: scalar in xarray.Dataset format.
        acause: str
        sex_id: int, 1 or 2
        location_ids: list
        scenario: -1 or 0 or 1
        date: date to version-control plots
        outdir: output directory
        start_age_group_id: int, default 10
        end_age_group_id: int, default 22
    '''
    outfile = os.path.join(outdir.format(d=date),
                           '{}_{}.pdf'.format(acause, scenario))
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    ds = scalar_ds.loc[{'scenario': scenario, 'sex_id': sex_id}]
    # Calculate statistics.
    ds['mean'] = ds['value'].mean(dim='draw')
    ds['lower'] = ds['value'].quantile(0.025, dim='draw')
    ds['upper'] = ds['value'].quantile(0.975, dim='draw')

    age_map = get_ages().set_index('age_group_id')['age_group_name'].to_dict()

    with PdfPages(outfile) as pp:
        for location_id in location_ids:
            loc_ds = ds.loc[{"location_id": location_id}]
            nrow = 4
            ncol = 3
            fig = plt.figure(figsize=(12, 10))
            grid = gridspec.GridSpec(nrow, ncol)

            for ix, age_group_id in enumerate(xrange(start_age_group_id,
                                                     end_age_group_id)):
                ax = fig.add_subplot(grid[ix])
                age_ds = loc_ds.loc[{"age_group_id": age_group_id}]
                ax.plot(age_ds['year_id'], age_ds['mean'])
                ax.fill_between(age_ds['year_id'], age_ds['lower'],
                                age_ds['upper'])
                ax.text(0.7, 0.95, age_map[age_group_id],
                        verticalalignment='top',
                        horizontalalignment='left', transform=ax.transAxes,
                        color='black', fontsize=12)

            location_map = get_modeled_locations().\
                set_index('location_id')['location_name'].to_dict()
            location = location_map[location_id]
            suptitle = "{location}, {acause}; Scenario: {scenario}".format(
                        location=location, acause=acause, scenario=scenario)
            fig.suptitle(suptitle, fontsize=15)
            pp.savefig(bbox_inches='tight')
    logger.info('Scalars plotting finished: {}'.format(acause))


def plot_paf(df, acause, risk, date, paf_cols, demography_cols):
    ''' Plot cause specific scalars.'''

    age_map = get_ages().\
        set_index('age_group_id')['age_group_name'].to_dict()
    location_map = get_modeled_locations().\
        set_index('location_id')['location_name'].to_dict()

    df['mean'] = np.mean(df.loc[:, paf_cols].values, axis=1)
    df['lower'] = np.percentile(df.loc[:, paf_cols].values, 2.5, axis=1)
    df['upper'] = np.percentile(df.loc[:, paf_cols].values, 97.5, axis=1)
    df = df[demography_cols + ['mean', 'lower', 'upper']]
    df = df.loc[df.scenario == 0]

    outfile = ('/ihme/forecasting/data/paf/{date}/plots/'
               '{acause}_{risk}_2.pdf'.format(date=date,
                                              risk=risk,
                                              acause=acause))

    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    location_ids = get_gbd_demographics().location_id.unique()

    with PdfPages(outfile) as pp:
        for location_id in location_ids:  # [6, 102]
            test = df.loc[df.location_id == location_id]
            nrow = 4
            ncol = 3
            fig = plt.figure(figsize=(12, 10))
            grid = gridspec.GridSpec(nrow, ncol)

            for ix, age_group_id in enumerate(xrange(2, 14)):
                ax = fig.add_subplot(grid[ix])
                tmp = test.loc[test.age_group_id == age_group_id]
                tmp = tmp.drop_duplicates(demography_cols)
                ax.plot(tmp.year_id.unique(),
                        tmp.loc[tmp.sex_id == 2]['mean'].values, 'b',
                        label='Female')
                ax.fill_between(tmp.year_id.unique(),
                                tmp.loc[tmp.sex_id == 2]['lower'].values,
                                tmp.loc[tmp.sex_id == 2]['upper'].values)

                ax.text(0.7, 0.95, age_map[age_group_id],
                        verticalalignment='top',
                        horizontalalignment='left', transform=ax.transAxes,
                        color='black', fontsize=12)
            location = location_map[location_id]
            suptitle = '{location}, {acause}'.format(location=location,
                                                     acause=acause)
            fig.suptitle(suptitle, fontsize=15)
            pp.savefig()


def plot_risk_attributable(ds, acause, sex_id, location_ids, risk, outdir,
                           start_age_group_id=10, end_age_group_id=22):
    ''' Plot risk attributable mortality across scenarios.

        # NOTE: this is called within plot_risk_attr_mort()

        Parameters
        ----------
        ds: risk attributable mortality in xarray format.
        acause: str
        risk: str
        sex_id: int, 1 or 2
        location_ids: list
        outdir: output directory
        start_age_group_id: int, default 10
        end_age_group_id: int, default 22
    '''
    # TODO Kendrick said: This is kind of weird (I think), because if I call
    # a different plotting function plot_thing(), and then this plotting
    # function, and then call plot_thing() again,
    # it could potentially change plot_thing() because of the sns stuff.
    sns.despine()
    sns.set_style('ticks')

    age_map = get_ages().\
        set_index('age_group_id')['age_group_name'].to_dict()
    location_map = get_modeled_locations().\
        set_index('location_id')['location_name'].to_dict()
    color_map = ['r', 'b', 'g']

    sexn = 'male' if sex_id == 1 else 'female'
    outfile = os.path.join(outdir, '{}_{}_{}.pdf'.format(acause, risk, sexn))
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    # Calculate statistics.
    ds['mean'] = ds['value'].mean(dim='draw')
    ds = ds.loc[dict(sex_id=sex_id)]

    with PdfPages(outfile) as pp:
        for location_id in location_ids:
            loc_ds = ds.loc[{"location_id": location_id}]
            nrow = 4
            ncol = 3
            fig = plt.figure(figsize=(15, 12))
            grid = gridspec.GridSpec(nrow, ncol)

            for ix, age_group_id in enumerate(xrange(start_age_group_id,
                                                     end_age_group_id)):
                ax = fig.add_subplot(grid[ix])
                age_ds = loc_ds.loc[{"age_group_id": age_group_id}]
                for scenario in [-1, 1, 0]:
                    scenario_ds = age_ds.loc[dict(scenario=scenario)]
                    ax.plot(scenario_ds['year_id'], scenario_ds['mean'],
                            color=color_map[scenario+1])
                    ax.text(0.7, 0.95, age_map[age_group_id],
                            verticalalignment='top',
                            horizontalalignment='left',
                            transform=ax.transAxes,
                            color='black', fontsize=12)
                    ax.axvline(x=2015, color='k')

            location = location_map[location_id]
            suptitle = "{location}, {acause}, {risk}".format(
                        location=location, acause=acause,
                        scenario=scenario, risk=risk)
            fig.suptitle(suptitle, fontsize=15)
            pp.savefig()
