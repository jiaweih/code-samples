'''Code snippets related to population forecasting using cohort component model.
'''
import pandas as pd
import numpy as np


DRAW = 1000


def merge_pop_nqx_under_one(pop, weekly_nqx, sex_id):
    ''' Merge age under one population with nqx(probability of dying).

        Parameters
        ----------
        pop: dataframe
            Contains population data with columns location_id, age_group_id,
            sex_id, year_id, age.
        weekly_nqx: dataframe
            Contains probability of dying for age group enn, lnn, pnn.
        sex_id: int
            1 indicates male, 2 indicates female.

        Returns
        -------
        x: dataframe
            Contains 52 rows with one row for each week's population
            and probability of dying.
    '''
    # 'enn': week 1; 'lnn': week 2 - 4; 'pnn': week 5 - 52
    under_one_pop = pop.loc[(pop.age.isin(['enn', 'lnn', 'pnn'])) & (pop.sex_id==sex_id)]
    under_one_pop = under_one_pop.merge(weekly_nqx, \
                    on=['location_id', 'age_group_id', 'sex_id', 'year_id', 'age'])
    # Replicate rows for each age group.
    enn = under_one_pop.iloc[[0]]
    lnn = under_one_pop.iloc[[1]*3]
    pnn = under_one_pop.iloc[[2]*48]
    x = pd.concat([enn, lnn, pnn], ignore_index=True)
    return x


def average_pop(df):
    '''Get the population at week0; assume weekly population within the same
       age group are the same and the number of weekly new borns are the same.
    '''
    for i in range(DRAW):
        df['wk0_{i}'.format(i=i)] = df['pop_{i}'.format(i=i)] / len(df)
    return df


def repeat_weeks(x, new_borns):
    '''Simulation of the weekly population prediction by pushing 
       the envelop forward for 52 weeks.

       Parameters
       ----------
       x: dataframe
        Contains draws of population and probability of dying for each week age.
       new_borns: 1-D array
        Draws of new borns.

       Returns
       -------
       c_x: dataframe
        Population of each week after pushing the envelop forward for 52 weeks.

    '''
    c_x = x.copy()

    for _ in np.arange(1, 52+1):
        for i in np.arange(DRAW):
            # The population of each age(week) of the new week(t) depends on the 
            # population and probability of survival of last week(t-1).
            c_x['wk1_{i}'.format(i=i)] = c_x['wk0_{i}'.format(i=i)].shift(1) * \
                                        (1 - c_x['draw{i}'.format(i=i)].shift(1))
            # The population of starting age(week) of the new week(t) depends 
            # on the number of weekly new borns and their probability of survival.
            c_x.loc[0, 'wk1_{i}'.format(i=i)] = \
                (1 - c_x.loc[0, 'draw{i}'.format(i=i)]) * new_borns[i-1]
            # The population of each age(week) in the new week(t) becomes that of 
            # the past week(t-1) for calculation of new cycle.
            c_x['wk0_{i}'.format(i=i)] = c_x['wk1_{i}'.format(i=i)]
    return c_x


def aggregate_weeks_pop(x, week_str):
    '''Aggregate the population of each age(week) into each age_group.
       Week 2-4 will be age_group_id 3, week 5-52 will be age_group_id 4.

       Parameters
       ----------
       x: dataframe
        Contains population of each week age after simulation for 52 weeks.
       week_str: str
        'week_0' or 'week_1'
    '''
    pop_draws = [\
        x.groupby(['location_id', 'sex_id', 'year_id', \
            'age_group_id', 'age'])['{week_str}_{i}'.format(week_str=week_str, i=i)].sum() \
                                        for i in np.arange(DRAW)]
    pop = pd.concat(pop_draws, axis=1) \
            .reset_index() \
            .rename(columns={'{week_str}_{i}'.format(week_str=week_str, i=i): \
                        'pop_{i}'.format(i=i) for i in np.arange(DRAW)})
    return pop


def forecast_under_one(pop_t, weekly_nqx, newborns_df, sex_id):
    '''Forecast population for age under one. There are three age_groups for 
       age under one: 2(enn, early neonatal, week 1),
       3(lnn, late neonatal, week 2 - 4), 4(pnn, post neonatal, week 5 - 52).

       Parameters
       ----------
       pop_t: dataframe
        Population of age under one in year t.
       weekly_nqx: dataframe
        Probability of dying for age_group enn, lnn, pnn
       newborns_df: dataframe
        Draws of new borns for male and female.
       sex_id: int
        1 indicates male, 2 indicates female.

       Returns
       -------
       pop_tplus1: dataframe
        Population for age group under one in year t+1.

    '''
    pop_nqx = merge_pop_nqx_under_one(pop_t, weekly_nqx, sex_id)
    # Obtain population at week 0.
    pop_nqx = pop_nqx.groupby(['age', 'year_id', 'location_id', 'sex_id', 'age_group_id']) \
                     .apply(average_pop)
    # Assign 1 - 52 for each week ages.
    pop_nqx['age_wks'] = np.arange(1, 52+1)
    sex_id = pop_nqx.loc[0, 'sex_id']
    # Column names of draws of weekly live births for both sexes.
    draws_live_births_males_weekly = \
        ['live_births_males_weekly_{i}'.format(i=i) for i in np.arange(DRAW)]
    draws_live_births_females_weekly = \
        ['live_births_females_weekly_{i}'.format(i=i) for i in np.arange(DRAW)]

    if sex_id == 1:
        new_borns = newborns_df[draws_live_births_males_weekly].values.sum(axis=0)
    else:
        new_borns = newborns_df[draws_live_births_females_weekly].values.sum(axis=0)

    x = repeat_weeks(pop_nqx, new_borns)
    pop_tplus1 = aggregate_weeks_pop(x, 'wk1')

    return pop_tplus1
