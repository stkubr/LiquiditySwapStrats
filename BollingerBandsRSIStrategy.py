import pandas as pd
import numpy as np
import math
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
import UNI_v3_funcs

class BollingerBandsRSIStrategy:
    def __init__(self,model_data,alpha_param,tau_param,limit_parameter):
    
        self.alpha_param            = alpha_param
        self.tau_param              = tau_param
        self.limit_parameter        = limit_parameter
        
    def check_strategy(self,current_strat_obs,strategy_info):
        pass
            
            
    def set_liquidity_ranges(self,current_strat_obs):
        pass
        
        
    ########################################################
    # Extract strategy parameters
    ########################################################
    def dict_components(self,strategy_observation):
        pass