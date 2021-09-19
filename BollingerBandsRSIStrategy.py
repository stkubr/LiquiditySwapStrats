import pandas as pd
import numpy as np
import math
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
import UNI_v3_funcs

class BollingerBandsRSIStrategy:
    def __init__(self,model_data,alpha_param,tau_param,limit_parameter):

        self.model_data = model_data
        self.alpha_param            = alpha_param
        self.tau_param              = tau_param
        self.limit_parameter        = limit_parameter
        
    def check_strategy(self,current_strat_obs,strategy_info):
        #####################################
        #
        # This strategy rebalances in three scenarios:
        # 1. Leave Reset Range
        # 2. Limit position is too unbalanced (limit_parameter)
        # 3. Volatility has dropped           (volatility_reset_ratio)
        #
        #####################################

        LEFT_RANGE_LOW = current_strat_obs.price < strategy_info['reset_range_lower']
        LEFT_RANGE_HIGH = current_strat_obs.price > strategy_info['reset_range_upper']
        LIMIT_ORDER_BALANCE = current_strat_obs.liquidity_ranges[1]['token_0'] + current_strat_obs.liquidity_ranges[1][
            'token_1'] * current_strat_obs.price
        BASE_ORDER_BALANCE = current_strat_obs.liquidity_ranges[0]['token_0'] + current_strat_obs.liquidity_ranges[0][
            'token_1'] * current_strat_obs.price
        model_forecast = None

        # Rebalance out of limit when have both tokens in self.limit_parameter ratio
        if current_strat_obs.liquidity_ranges[1]['token_0'] > 0.0 and current_strat_obs.liquidity_ranges[1][
            'token_1'] > 0.0:
            LIMIT_SIMILAR = ((current_strat_obs.liquidity_ranges[1]['token_0'] / current_strat_obs.liquidity_ranges[1][
                'token_1']) >= self.limit_parameter) | \
                            ((current_strat_obs.liquidity_ranges[1]['token_0'] / current_strat_obs.liquidity_ranges[1][
                                'token_1']) <= (self.limit_parameter + 1))
            if BASE_ORDER_BALANCE > 0.0:
                LIMIT_REBALANCE = ((LIMIT_ORDER_BALANCE / BASE_ORDER_BALANCE) > (
                        1 + self.limit_parameter)) & LIMIT_SIMILAR
            else:
                LIMIT_REBALANCE = LIMIT_SIMILAR
        else:
            LIMIT_REBALANCE = False

        # if a reset is necessary
        if ((LEFT_RANGE_LOW | LEFT_RANGE_HIGH) | LIMIT_REBALANCE):
            current_strat_obs.reset_point = True

            if (LEFT_RANGE_LOW | LEFT_RANGE_HIGH):
                current_strat_obs.reset_reason = 'exited_range'
            elif LIMIT_REBALANCE:
                current_strat_obs.reset_reason = 'limit_imbalance'

            # Remove liquidity and claim fees
            current_strat_obs.remove_liquidity()

            # Reset liquidity
            # TODO: Clean up returns
            liq_range, strategy_info = self.set_liquidity_ranges(current_strat_obs)
            return liq_range, strategy_info
        else:
            return current_strat_obs.liquidity_ranges, strategy_info


    def generate_Bollinger_Bands_RSI_limits(self, timepoint, period=20):
        current_spot = np.argmin(abs(self.model_data['time_pd'] - timepoint))
        prices = self.model_data['quotePrice'].iloc[:current_spot].iloc[-period:]
        price_returns = self.model_data['price_return'].iloc[:current_spot].iloc[-period:]

        rsi = self.calc_rsi(price_returns, period).tail(1)[0]/100

        upper_band = prices.rolling(period).mean() + 0.8*2*prices.rolling(period).std() + 0.2*2*prices.rolling(period).std() * (rsi)
        lower_band = prices.rolling(period).mean() - 0.8*2*prices.rolling(period).std() - 0.2*2*prices.rolling(period).std() * (1 - rsi)
        mid_point = prices.rolling(period).mean()

        result_dict = {'upper_bb': upper_band.tail(1)[0],
                  'lower_bb': lower_band.tail(1)[0],
                  'mid_bb': mid_point.tail(1)[0],
        }

        return result_dict

    def calc_rsi(self, price_returns, periods=14, ema=True):
        # Make two series: one for lower closes and one for higher closes
        up = price_returns.clip(lower=0)
        down = -1 * price_returns.clip(upper=0)

        if ema == True:
            # Use exponential moving average
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window=periods, adjust=False).mean()
            ma_down = down.rolling(window=periods, adjust=False).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))
        return rsi

            
    def set_liquidity_ranges(self,current_strat_obs):
        ###########################################################
        # STEP 1: Do calculations required to determine base liquidity bounds
        ###########################################################

        bollinger_bands = self.generate_Bollinger_Bands_RSI_limits(current_strat_obs.time)

        strategy_info = dict()
        strategy_info['reset_range_lower'] = bollinger_bands['lower_bb']
        strategy_info['reset_range_upper'] = bollinger_bands['upper_bb']

        # Set the base range
        base_range_lower = bollinger_bands['lower_bb']
        base_range_upper = bollinger_bands['upper_bb']

        save_ranges = []


        ###########################################################
        # STEP 2: Set Base Liquidity
        ###########################################################

        # Store each token amount supplied to pool
        total_token_0_amount = current_strat_obs.liquidity_in_0
        total_token_1_amount = current_strat_obs.liquidity_in_1

        # Lower Range
        TICK_A_PRE = int(math.log(current_strat_obs.decimal_adjustment * base_range_lower, 1.0001))
        TICK_A = int(round(TICK_A_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)

        # Upper Range
        TICK_B_PRE = int(math.log(current_strat_obs.decimal_adjustment * base_range_upper, 1.0001))
        TICK_B = int(round(TICK_B_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)

        liquidity_placed_base = int(
            UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick, TICK_A, TICK_B, current_strat_obs.liquidity_in_0, \
                                       current_strat_obs.liquidity_in_1, current_strat_obs.decimals_0,
                                       current_strat_obs.decimals_1))

        base_0_amount, base_1_amount = UNI_v3_funcs.get_amounts(current_strat_obs.price_tick, TICK_A, TICK_B,
                                                                liquidity_placed_base \
                                                                , current_strat_obs.decimals_0,
                                                                current_strat_obs.decimals_1)

        total_token_0_amount -= base_0_amount
        total_token_1_amount -= base_1_amount

        base_liq_range = {'price': current_strat_obs.price,
                          'lower_bin_tick': TICK_A,
                          'upper_bin_tick': TICK_B,
                          'lower_bin_price': base_range_lower,
                          'upper_bin_price': base_range_upper,
                          'time': current_strat_obs.time,
                          'token_0': base_0_amount,
                          'token_1': base_1_amount,
                          'position_liquidity': liquidity_placed_base,
                          'reset_time': current_strat_obs.time}

        save_ranges.append(base_liq_range)

        ###########################
        # Set Limit Position according to probability distribution
        ############################

        limit_amount_0 = total_token_0_amount
        limit_amount_1 = total_token_1_amount

        # Place singe sided highest value
        if limit_amount_0 * current_strat_obs.price > limit_amount_1:
            # Place Token 0
            limit_amount_1 = 0.0
            limit_range_lower = current_strat_obs.price
            limit_range_upper = base_range_upper

        else:
            # Place Token 1
            limit_amount_0 = 0.0
            limit_range_lower = base_range_lower
            limit_range_upper = current_strat_obs.price

        TICK_A_PRE = int(math.log(current_strat_obs.decimal_adjustment * limit_range_lower, 1.0001))
        TICK_A = int(round(TICK_A_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)

        TICK_B_PRE = int(math.log(current_strat_obs.decimal_adjustment * limit_range_upper, 1.0001))
        TICK_B = int(round(TICK_B_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)

        if TICK_A == TICK_B:
            TICK_B = TICK_B + 1

        liquidity_placed_limit = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick, TICK_A, TICK_B, \
                                                                limit_amount_0, limit_amount_1,
                                                                current_strat_obs.decimals_0,
                                                                current_strat_obs.decimals_1))
        limit_0_amount, limit_1_amount = UNI_v3_funcs.get_amounts(current_strat_obs.price_tick, TICK_A, TICK_B, \
                                                                  liquidity_placed_limit, current_strat_obs.decimals_0,
                                                                  current_strat_obs.decimals_1)

        limit_liq_range = {'price': current_strat_obs.price,
                           'lower_bin_tick': TICK_A,
                           'upper_bin_tick': TICK_B,
                           'lower_bin_price': limit_range_lower,
                           'upper_bin_price': limit_range_upper,
                           'time': current_strat_obs.time,
                           'token_0': limit_0_amount,
                           'token_1': limit_1_amount,
                           'position_liquidity': liquidity_placed_limit,
                           'reset_time': current_strat_obs.time}

        save_ranges.append(limit_liq_range)

        # Update token amount supplied to pool
        total_token_0_amount -= limit_0_amount
        total_token_1_amount -= limit_1_amount

        # Check we didn't allocate more liquidiqity than available
        assert current_strat_obs.liquidity_in_0 >= total_token_0_amount
        assert current_strat_obs.liquidity_in_1 >= total_token_1_amount

        # How much liquidity is not allcated to ranges
        current_strat_obs.token_0_left_over = max([total_token_0_amount, 0.0])
        current_strat_obs.token_1_left_over = max([total_token_1_amount, 0.0])

        # Since liquidity was allocated, set to 0
        current_strat_obs.liquidity_in_0 = 0.0
        current_strat_obs.liquidity_in_1 = 0.0

        return save_ranges, strategy_info
        
        
    ########################################################
    # Extract strategy parameters
    ########################################################
    def dict_components(self,strategy_observation):
        this_data = dict()

        # General variables
        this_data['time'] = strategy_observation.time
        this_data['price'] = strategy_observation.price
        this_data['price_1_0'] = 1 / this_data['price']
        this_data['reset_point'] = strategy_observation.reset_point
        this_data['reset_reason'] = strategy_observation.reset_reason

        # Range Variables
        this_data['base_range_lower'] = strategy_observation.liquidity_ranges[0]['lower_bin_price']
        this_data['base_range_upper'] = strategy_observation.liquidity_ranges[0]['upper_bin_price']
        this_data['limit_range_lower'] = strategy_observation.liquidity_ranges[1]['lower_bin_price']
        this_data['limit_range_upper'] = strategy_observation.liquidity_ranges[1]['upper_bin_price']
        this_data['reset_range_lower'] = strategy_observation.strategy_info['reset_range_lower']
        this_data['reset_range_upper'] = strategy_observation.strategy_info['reset_range_upper']

        # Fee Varaibles
        this_data['token_0_fees'] = strategy_observation.token_0_fees
        this_data['token_1_fees'] = strategy_observation.token_1_fees
        this_data['token_0_fees_accum'] = strategy_observation.token_0_fees_accum
        this_data['token_1_fees_accum'] = strategy_observation.token_1_fees_accum

        # Asset Variables
        this_data['token_0_left_over'] = strategy_observation.token_0_left_over
        this_data['token_1_left_over'] = strategy_observation.token_1_left_over

        total_token_0 = 0.0
        total_token_1 = 0.0
        for i in range(len(strategy_observation.liquidity_ranges)):
            total_token_0 += strategy_observation.liquidity_ranges[i]['token_0']
            total_token_1 += strategy_observation.liquidity_ranges[i]['token_1']

        this_data['token_0_allocated'] = total_token_0
        this_data['token_1_allocated'] = total_token_1
        this_data[
            'token_0_total'] = total_token_0 + strategy_observation.token_0_left_over + strategy_observation.token_0_fees_accum
        this_data[
            'token_1_total'] = total_token_1 + strategy_observation.token_1_left_over + strategy_observation.token_1_fees_accum

        # Value Variables
        this_data['value_position'] = this_data['token_0_total'] + this_data['token_1_total'] * this_data['price_1_0']
        this_data['value_allocated'] = this_data['token_0_allocated'] + this_data['token_1_allocated'] * this_data[
            'price_1_0']
        this_data['value_left_over'] = this_data['token_0_left_over'] + this_data['token_1_left_over'] * this_data[
            'price_1_0']

        this_data['base_position_value'] = strategy_observation.liquidity_ranges[0]['token_0'] + \
                                           strategy_observation.liquidity_ranges[0]['token_1'] * this_data['price_1_0']
        this_data['limit_position_value'] = strategy_observation.liquidity_ranges[1]['token_0'] + \
                                            strategy_observation.liquidity_ranges[1]['token_1'] * this_data['price_1_0']

        return this_data