'''Snippets of code related to SEV calculation.

'''
import pandas as pd
import numpy as np
import scipy.integrate
import ctypes
import fbd  # team repository containing some utility functions


def get_ctypes_wrapper(shared_object_file):
    '''Create a ctypes wrapper.

       Parameters
       ----------
       shared_object_file: a .so file compiled from C file: 
       gcc -shared -o lib.so -fPIC lib.c

       Returns
       -------
       func: a ctypes wrapper.
    '''
    lib = ctypes.CDLL(shared_object_file)
    func = lib.f
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_double)
    return func


def calculate_sev(exp_mean, exp_sd, rr_mean, rr_max, tmrel,
                    integ_min, integ_max, flag, shared_object_file):
    '''Calculate sev with ctypes.

       Parameters
       ----------
       exp_mean: 1-D array
            Draws of the mean of exposure
       exp_sd: 1-D array,
            Draws of the standard deviation of exposure
       rr_mean: 1-D array,
            Draws of the mean of relative risk
       rr_max: 1-D array,
            Draws of the maximum relative risk
       tmrel: int or float,
            Theoretical minimum risk exposure level
       integ_min: int or float,
            Lowest level of exposure, also lower interval of integration
       integ_max: int or float,
            Maximum level of exposure, also upper interval of integration
       flag: int(0 or 1)
            1 indicates beneficial risk factors;
            0 indicates detrimental risk factors.
       shared_object_file: str
            A .so file compiled from C file.

       Returns
       -------
       sev: 1-D array
        summary exposure value.
       paf: 1-D array
        population attributable fraction.
    '''
    assert len(exp_mean) == len(exp_sd) == len(rr_mean) == len(rr_max), \
           print("Length of array doesn't match")

    func = get_ctypes_wrapper(shared_object_file)

    def integ(i):
        '''Integration of the proudct of relative risk and exposure.
           Calculate SEV and PAF.

           Parameters
           ----------
           i: int
            index of array, from 0 to length of array.

           Returns
           -------
           sev: int or float, between 0 and 1.
           paf: int or float, between 0 and 1.
        '''
        integral = scipy.integrate.quad(func, integ_min, integ_max,
            args=(exp_mean[i], exp_sd[i], tmrel, rr_mean[i], rr_max[i], flag),
            epsabs=1.49e-06, epsrel=1.49e-06)

        numerator = integral[0] - 1
        denominator = rr_max[i] - 1
        sev = numerator / denominator
        paf = 1 - 1./(sev * denominator + 1)
        return sev, paf

    len_array = len(exp_mean)
    # Apply integration to an array of data.
    integration = list(map(integ, range(len_array)))
    sev = np.array(integration)[:, 0]
    paf = np.array(integration)[:, 1]
    return sev, paf


def pull_acause_risk(db_name, server):
    '''Query acause, risk pairs from database.
       Return a dataframe with two columns, acause and risk.

       Parameters
       ----------
       db_name: str
            The name of database, ex. 'forecasting'
       server: str
            Server to be queried, ex. 'forecasting-db-d01'

       Returns
       -------
       df_acause_risk: dataframe
            Dataframe with two columns, cause and risk.
    '''
    engine = fbd.utils.get_engine(db_name, server=server)
    query = '''SELECT acause, cause_id, rei, rei_id 
               FROM forecasting.best_cause_risk
               WHERE fbd_round_id = 1;'''
    df_acause_risk = pd.read_sql_query(query, engine)
    return df_acause_risk
