'''Code snippets related to modeling of deaths caused by natural disaster.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import bernoulli

import seaborn as sns
from patsy import dmatrices, dmatrix
import statsmodels.discrete.discrete_model as sm

import rpy2
from rpy2.robjects import pandas2ri

# Number of draws
DRAW = 1000
# Column names of draws
DRAW_COLS = ['draw_{}'.format(i) for i in range(DRAW)]


def fit_model_disaster(df, model):
    '''Fit logistic regression model for outcome that if a disaster occurr.

       Parameters
       ----------
       df: dataframe
        Contains columns of outcome and predictors.
       model: str
        A formula string used to construct a design matrix.
        Ex. 'has_disaster ~ C(iso3) + C(region_id) + year_id + cz_prop'

       Returns
       -------
       modeled_df: dataframe
        Contains draws of predicted events.
        1 indicates event occurs; 0 indicates otherwise.

    '''
    # Return matrix of y and X based on model formula.
    y, X = dmatrices(model, df, return_type='dataframe')
    # Construct logit model.
    logit = sm.Logit(y, X)
    # Fit model
    result = logit.fit()
    modeled_df = df.copy()
    # Return predicted probability based on fitted model.
    modeled_df['predicted_prob'] = result.predict()
    # Create columns to store samples of sampled Bernoulli variables.
    modeled_df = modeled_df.reindex(columns=list(modeled_df.columns) + DRAW_COLS)
    # Draw random samples from Bernoulli distribution
    # based on predicted probability.
    modeled_df[DRAW_COLS] = np.array([bernoulli.rvs(p, size=DRAW) \
                                for p in modeled_df.predicted_prob.values])
    return modeled_df


def fit_model_death_rates(df):
    '''Fit model for the death rates and extract coefficients.

       Parameters
       ----------
       df: dataframe
        Contains columns of the log of death rates, sdi,
        age_group_id, region_id, year_id

       Returns
       -------
       feff_df: dataframe
        Contains the draws of coefficients of fixed effects.
       reff_df: dataframe
        Contains the draws of coefficients of random effects.
    '''

    rpy2.robjects.globalenv['data'] = pandas2ri.py2ri(df)
    # Fit linear mixed-effects model in R, fixed effects on sdi, random effects 
    # on age_group_id and region_id.
    rpy2.robjects.r('''
        model = lmer(log_death_rate ~ 1 + sdi + (1|age_group_id) + (1|region_id), data, REML=F)
        capture.output(summary(model),
            file = paste("/ihme/forecasting/data/disaster/outputs/disaser_model.txt", sep=""))

        feff_means = fixef(model)
        reff_means = ranef(model)
        feff_var = attr(vcov(model), "x")
        reff_var = VarCorr(model)
    ''')
    # Extract the coefficients of fixed effects of the model.
    feffects = ['sdi']
    param_list = ['intercept'] + feffects
    fixed_means = pandas2ri.ri2py(rpy2.robjects.globalenv['feff_means'])
    fixed_var = np.reshape(np.array(rpy2.robjects.globalenv['feff_var']),
                          (len(param_list), len(param_list)))
    # Draw random samples from a multivariate normal distribution.
    feff_df = pd.DataFrame(np.random.multivariate_normal( \
                            fixed_means, fixed_var, 1000), \
                            index=np.arange(1000), \
                            columns=[param_list])

    # Extract the coefficients of random effects of the model
    reffects = ['age_group_id', 'region_id']
    reff_means = {}
    reff_vars = {}
    reff_df = {}
            
    for r in reffects:
        rpy2.robjects.r('''
            reff_means_df = data.frame(reff_means${r})
            reff_means_df['{r}_label'] = row.names(reff_means_df)
        '''.format(r=r))
        reff_means[r] = pandas2ri.ri2py(rpy2.robjects.r['reff_means_df'])
        reff_means[r] = reff_means[r].rename(columns={'X.Intercept.': \
                                                    'mean_{}'.format(r)})

        reff_vars[r] = np.array(rpy2.robjects \
                         .r('attr(reff_var${r}, "stddev")'.format(r=r)))[0]
        reff_means[r]['se_{}'.format(r)] = reff_vars[r]
        # Draw random samples from a normal distribution.
        reff_df[r] = pd.DataFrame(np.random.normal(
                                     reff_means[r]['mean_{}'.format(r)],
                                     reff_means[r]['se_{}'.format(r)],
                                     size=(1000, len(reff_means[r]))),
                                     columns=reff_means[r]['{}_label'.format(r)]) \
                        .transpose() \
                        .reset_index()
    return feff_df, reff_df


def boxplot(beta_name):
    '''Plot and save boxplots of draws of coefficients of predictors to determine
       whether the coefficents are significant.

       Parameters
       ----------
       beta_name: str
            The name of beta's name of the model. Ex. region_id, sdi.

    '''
    infile = '/ihme/forecasting/data/disaster/outputs/coef_{}.csv'.format(beta_name)
    coef = pd.read_csv(infile)
    outfile = '/ihme/forecasting/data/disaster/outputs/{}_coeff_boxplot.pdf'.format(beta_name)
    pp = PdfPages(outfile)

    sns.set_style('ticks')
    f, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(x=beta_name, y="value", data=coef, ax=ax)
    ax.set_xlabel(beta_name)
    ax.set_ylabel('Coefficients')
    pp.savefig()
    pp.close()
    return


def scatter_death_rates_risk_score(death_risk_score, disaster, 
                                    age_panel=False, sex_panel=False):
    ''' Scatter plots of death rates against risk scores.
        Parameters
        ----------
        death_risk_score: dataframe
            Contain columns of risk_score, log of death rates, 
            age_group_id, sex_id.
        disaster: str
            Disaster name
        age_panel: bool, optional
            True if plotting age-specific scatterplots.
        sex_panel: bool, optional
            True if plotting sex-specific scatterplots.
    '''
    if age_panel:
        # Age-specific scatterplot of risk_score against log_death_rates.
        f, ax = plt.subplots(figsize=(15, 12))
        g = sns.lmplot(x="risk_score", y="log_death_rates", col="age_group_id", \
                    data=death_risk_score, col_wrap=4, size=3);
        g.set_axis_labels("Risk Score", "Log(Death Rates)");
    elif sex_panel:
        # Sex-specific scatterplot of risk_score against log_death_rates.
        f, ax = plt.subplots(figsize=(12, 10))
        g = sns.lmplot(x="risk_score", y="log_death_rates", col="sex_id", \
                    data=death_risk_score);
        g.set_axis_labels("Risk Score", "Log(Death Rates)");
    else:
        # Scatterplot of risk_score against log_death_rates for all data.
        f, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x="risk_score", y="log_death_rates", data=death_risk_score, ax=ax);
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Log(Death Rates)')
        ax.set_title(disaster)
    return