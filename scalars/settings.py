from collections import OrderedDict
from fbd_core.db.forecasting_demographics import get_modeled_locations

GBD_ROUND = None
FORECAST_START_YEAR = 2016
START_YEAR = 1990
END_YEAR = 2040

DEFAULT_YEARS = (START_YEAR, FORECAST_START_YEAR, END_YEAR)

# Now we define the demographic indices
LOCATION_IDS = get_modeled_locations().location_id.values.tolist()
AGE_GROUP_IDS = range(2, 22)  # TODO use fbd_core.db.forecasting_demographics
SEX_IDS = [1, 2]
YEAR_IDS = range(START_YEAR, END_YEAR+1)
SCENARIOS = [-1, 0, 1]

DEMOGRAPHY_INDICES = OrderedDict(location_id=LOCATION_IDS,
                                 age_group_id=AGE_GROUP_IDS,
                                 sex_id=SEX_IDS,
                                 year_id=YEAR_IDS,
                                 scenario=SCENARIOS)
DEMOGRAPHY_COLS = DEMOGRAPHY_INDICES.keys()

DRAW_PREFIX = 'draw_'
PAF_DRAW_PREFIX = 'paf_'
SCALAR_DRAW_PREFIX = 'scalar_'
RR_MAX_DRAW_PREFIX = 'rr_'

# Parameters for run_scalars.py.

# these are causes that are not modeled.  As far as the pipeline is concerned,
# the following causes should simply be ignored, vaccine or not
VACCINE_NO_MODEL_CAUSES = ['digest_pud', 'nutrition_pem', 'nutrition_iron',
                           'mental_drug', 'msk_rheumarthritis',
                           'maternal_abort', 'maternal_sepsis',
                           'maternal_hem', 'std']

# these are lowest-level vaccine-related causes that we model
# "rotavirus" is manually added here because our "causes" table
# is not up-to-date.
VACCINE_RELATED_ACAUSES = ['lri_hib', 'lri_pneumo', 'diarrhea_rotavirus',
                           'tetanus', 'measles', 'whooping', 'diptheria',
                           'rotavirus']

# There are no scalars directly from the following.
# In fact, we consider scalars at the level of "lri" and "diarrhea",
# so the following need to be excluded from VACCINE_RELATED_ACAUSES when
# scalars are concerned.
ALREADY_INCLUDED_ACAUSES = ['lri_hib', 'lri_pneumo', 'diarrhea_rotavirus']

# Parameters for calculate_pafs.py.
INDIR_SEV = '/ihme/forecasting/data/fbd_scenarios_data/forecast/sev/{d:s}'
INDIR_RRMAX = '/ihme/forecasting/data/dalynator_gbd_2015_outputs/rrmax'
RISKS_NOT_AVAILABLE = ['unsafe_sex', 'metab_gfr', 'abuse_ipv_exp']

OUTDIR_PAF_FORECAST = '/ihme/forecasting/data/fbd_scenarios_data/forecast/' \
                      'paf/{d:s}/risk_acause_specific'
OUTDIR_PAF_PAST = '/ihme/forecasting/data/fbd_scenarios_data/past/paf/' \
                  'risk_acause_specific'

VACCINE_RISKS = ['rota', 'dtp3', 'pcv', 'measles', 'hib']

# Parameters for calculate_scalars.py.
INPATH_MEDIATION = '/ihme/forecasting/ref/jiawei/mediation.csv'
INPATH_PAF_SET_ONE = '/ihme/forecasting/ref/paf_set_one_risk_outcomes.lst'
# The PAF outputs from calculate_pafs.py
# will be inputs for calculate_scalars.py.
INDIR_PAF_FORECAST = OUTDIR_PAF_FORECAST
INDIR_PAF_PAST = OUTDIR_PAF_PAST

# For vaccine_scalars.py
VACCINE_VERSION = '20170619_haq_ratios'
INDIR_VACCINE_SEV = '/ihme/forecasting/data/fbd_scenarios_data/forecast/'\
                    'vaccine/{}'.format(VACCINE_VERSION)
INDIR_VACCINE_PAF = '/ihme/forecasting/data/fbd_scenarios_data/forecast/'\
                    'paf/{d:s}/risk_acause_specific'
OUTDIR_VACCINE_RRMAX = '/ihme/forecasting/data/paf/'\
                       '{d:s}/rrmax/'
ACAUSE_TO_VACC = {'diptheria': ['dtp3'],
                  'tetanus': ['dtp3'],
                  'whooping': ['dtp3'],
                  'measles': ['measles'],
                  'lri_hib': ['hib'],
                  'lri_pneumo': ['pcv'],
                  'diarrhea_rotavirus': ['rota'],
                  'rotavirus': ['rota']
                  }  # whooping = "pertussis"

PCV_INFO_XSL = ('/ihme/forecasting/data/vaccine_coverage/scalars/inputs/'
                'pcv_introduction_supplementary_info.xlsx')

PCV_EFFECT_XSL = ('/ihme/forecasting/data/vaccine_coverage/scalars/'
                  'inputs/pcv_effect.xlsx')

DIARRHEA_INPUT_CSV = ('/ihme/forecasting/data/vaccine_coverage/scalars/'
                      'inputs/RV_diarrhea.csv')

ROTAVIRUS_INPUT_CSV = ('/ihme/forecasting/data/vaccine_coverage/scalars/'
                       'inputs/rota_sever_rota_diarrhea.csv')

DATA_POOR_LOCATIONS_PKL = ('/ihme/forecasting/data/vaccine_coverage/scalars/'
                           'inputs/non_data_rich_locations.pkl')

SUBMODEL_COEFFS_ERR_ONLY_LOG_CSV = ('/ihme/forecasting/data/vaccine_coverage/'
                                    'scalars/inputs/'
                                    'submodel_coeffs_err_only_log.csv')

HIB_EFFECT_SIZE_XSL = ('/ihme/forecasting/data/vaccine_coverage/scalars/'
                       'inputs/hib_effect_size.xlsx')

VACCINE_COEF_CSV = ('/ihme/forecasting/data/vaccine_coverage/scalars/inputs/'
                    '{}_vaccine_coef.csv')

DISEASE_SHORT_NAMES = {
    'dtp3': 'DTP3_coverage_prop',
    'DTP3': 'DTP3_coverage_prop',
    'measles': 'measles_vacc_cov_prop',
    'Measles': 'measles_vacc_cov_prop',
    'rotavirus': 'ROTA_coverage_prop',
    'pneumococcus': 'PCV3_coverage_prop',
    'lri_hib': 'Hib3_coverage_prop'
}   # Should be pcv, but that's the way the ratios are stored

# These will get the date from inside the function where they are used
PAF_VERSION = '20170508_sdi_loess_adjustment'
OUTDIR_AGG_PAF_FORECAST = '/ihme/forecasting/data/fbd_scenarios_data/' \
                          'forecast/paf/{d:s}'
OUTDIR_AGG_PAF_PAST = '/ihme/forecasting/data/fbd_scenarios_data/' \
                      'past/paf'
OUTDIR_SCALAR_PAST = '/ihme/forecasting/data/fbd_scenarios_data/past/scalar'
OUTDIR_SCALAR_FORECAST = ('/ihme/forecasting/data/fbd_scenarios_data/forecast/'
                          'scalar/{d:s}/future')
OUTDIR_SCALAR = '/ihme/forecasting/data/fbd_scenarios_data/forecast/' \
                         'scalar/{d:s}'
OUTDIR_SCALAR_PLOTS = '/ihme/forecasting/data/fbd_scenarios_data/forecast/' \
                      'scalar/{d:s}/graphs'


# For risk_attributable_mortality.py
#
DEATH_VERSION = '20170524_k2_custom_blend_agg'
PAST_DEATH_DIR = '/ihme/forecasting/data/fbd_scenarios_data/past/death'
FORECAST_DEATH_DIR = '/ihme/forecasting/data/fbd_scenarios_data/forecast/'\
                       'death/{}'.format(DEATH_VERSION)
PAST_RISK_ACAUSE_PAF_DIR = '/ihme/forecasting/data/fbd_scenarios_data/past/'\
                           'paf/risk_acause_specific/'
FORECAST_RISK_ACAUSE_PAF_DIR = '/ihme/forecasting/data/fbd_scenarios_data/'\
                               'forecast/paf/{}/'\
                               'risk_acause_specific/'.format(PAF_VERSION)
PAST_RISK_ATTRIBUTABLE_DIR = '/ihme/forecasting/data/fbd_scenarios_data/past/'\
                             'death/risk_attributable'
FORECAST_RISK_ATTRIBUTABLE_DIR = '/ihme/forecasting/data/fbd_scenarios_data/'\
                'forecast/death/{}/risk_attributable'.format(DEATH_VERSION)
OUTDIR_RISK_ATTRIBUTABLE_PLOTS = '/ihme/forecasting/plot/risk_attributable'
