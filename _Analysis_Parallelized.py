# -*- coding: utf-8 -*-
"""
 - Make sure to include the spread when calculating success
 - Add feautre to incliude analysis parameters in foldername
 - Integrate this whole process into an app that can run on the phone...no need for texting, just push alerts from apps :D
 - Try to do some math to figure out high side instead of a hardcoded %. Like Fib levels, or stDevs, or something
 - Fix import of pairs and creation of conversion table so # isnt harcoded

 
@author: asklett
"""

import yfinance as yf
import pandas as pd
import ta
import numpy as np
import multiprocessing as mp


# Analysis parameters
IMPORT_PERIOD = '1mo'
IMPORT_INTERVAL = '5m'
TARGET_PERCENT = 5 # Target percent gain of margin used
NUM_CANDLESTICKS = 6 # The number of candlesticks to consider a success
SIGNALS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    
class CurrencyPair():
    '''
    Defines objects consiting as data for a single currency pair which are then
    iterated over to collect "winning strategy" data for each pair

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    '''

    def __init__(self, details, period, interval, target_percent, signal_list):
        '''
        valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

        Parameters
        ----------
        details : LIST
            List of pair-specific details including name and margin rate.
        IMPORT_PERIOD : INT
            Period of time over which to analyze data.
        IMPORT_INTERVAL : INT
            Interval over which to analyze data (e.g. 30 min candlesticks).
        target_percent : INT
            Target percent for which success will be assigned a 1 if reached.
            Input as a percent (not fraction), i.e., 95, not 0.95.
        signal_list : LIST
            List of signals over which to iterate.

        Returns
        -------
        None.

        '''
        self.pair = details[0]
        self.margin_rate = details[1]
        self.import_period = period
        self.import_interval = interval
        self.target_percent = target_percent * self.margin_rate
        self.signal_list = ['Signal_' + str(x) for x in signal_list]
        self.winning_strategies = []

    def update_data(self):
        '''
        Update currency data

        Returns
        -------
        None.

        '''
        self.currency_data = yf.Ticker(self.pair).history(period=self.import_period, interval=self.import_interval)

    def calculate_indicators(self):
        '''
        Calculates the values of all indicators and concatinates them to dataframe

        Returns
        -------
        None.

        '''
        high = self.currency_data['High']
        low = self.currency_data['Low']
        close = self.currency_data['Close']

        ##_Momentum Indicators_
        #Awesome Oscillator
        awesomeOscillator = ta.momentum.AwesomeOscillatorIndicator(high, low, window1=5, window2=34, fillna=False)
        self.currency_data['AwesomeOscillator'] = awesomeOscillator.awesome_oscillator()

        #KAMA
        KAMA = ta.momentum.KAMAIndicator(close, window=10, pow1=2, pow2=30, fillna=False)
        self.currency_data['KAMA'] = KAMA.kama()

        #ROC
        ROC_125 = ta.momentum.ROCIndicator(close, window=125, fillna=False)
        ROC_21 = ta.momentum.ROCIndicator(close, window=21, fillna=False)
        self.currency_data['ROC_125'] = ROC_125.roc()
        self.currency_data['ROC_21'] = ROC_21.roc()

        #RSI
        RSI = ta.momentum.RSIIndicator(close, window=14, fillna=False)
        self.currency_data['RSI'] = RSI.rsi()

        #Stochastic Oscillator
        stochastic = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3, fillna=False)
        self.currency_data['Stochastic'] = stochastic.stoch()

        #TSI
        TSI = ta.momentum.TSIIndicator(close, window_slow=25, window_fast=13, fillna=False)
        self.currency_data['TSI'] = TSI.tsi()

        #Williams R
        williams = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14, fillna=False)
        self.currency_data['WilliamsR'] = williams.williams_r()

        ##_Volatility Indicators_
        #AverageTrueRange
        ATR = ta.volatility.AverageTrueRange(high, low, close, window=14, fillna=False)
        self.currency_data['ATR'] = ATR.average_true_range()

        # Bollinger Bands
        bollingerBands = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        self.currency_data['BolBand_low'] = bollingerBands.bollinger_lband()
        self.currency_data['BolBand_high'] = bollingerBands.bollinger_hband()

        #DonchianChannel
        donchianChannel = ta.volatility.DonchianChannel(high, low, close, window=20, offset=0, fillna=False)
        self.currency_data['DonchianChannel_low'] = donchianChannel.donchian_channel_lband()
        self.currency_data['DonchianChannel_high'] = donchianChannel.donchian_channel_hband()

        #KeltnerChannel
        keltnerChannel = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=10, fillna=False, original_version=True)
        self.currency_data['KeltnerChannel_low'] = keltnerChannel.keltner_channel_lband()
        self.currency_data['KeltnerChannel_high'] = keltnerChannel.keltner_channel_hband()

        ##_Trend Indicators_
        #AroonIndicator
        AroonIndicator = ta.trend.AroonIndicator(close, window=25, fillna=False)
        self.currency_data['AroonIndicator_down'] = AroonIndicator.aroon_down()
        self.currency_data['AroonIndicator_up'] = AroonIndicator.aroon_up()

        #CCIIndicator
        CCIIndicator = ta.trend.CCIIndicator(high, low, close, window=20, constant=0.015, fillna=False)
        self.currency_data['CCI'] = CCIIndicator.cci()

        #DPOIndicator
        DPOIndicator = ta.trend.DPOIndicator(close, window=20, fillna=False)
        self.currency_data['DPO'] = DPOIndicator.dpo()

        #EMAIndicator
        EMAIndicator = ta.trend.EMAIndicator(close, window=14, fillna=False)
        self.currency_data['EMA'] = EMAIndicator.ema_indicator()

        #KSTIndicator
        KSTIndicator = ta.trend.KSTIndicator(close, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9, fillna=False)
        self.currency_data['KST'] = KSTIndicator.kst()
        self.currency_data['KST_signal'] = KSTIndicator.kst_sig()

        #MACD
        MACD = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
        self.currency_data['MACD'] = MACD.macd()
        self.currency_data['MACD_signal'] = MACD.macd_signal()

        #MassIndex
        MassIndex = ta.trend.MassIndex(high, low, window_fast=9, window_slow=25, fillna=False)
        self.currency_data['MassIndex'] = MassIndex.mass_index()

        #TRIXIndicator
        TRIXIndicator = ta.trend.TRIXIndicator(close, window=15, fillna=False)
        self.currency_data['TRIX'] = TRIXIndicator.trix()

        # Remove NaN rows at beginning
        self.currency_data = self.currency_data.dropna()
        self.currency_data = self.currency_data.reset_index()

    def assign_success(self, candlesticks):
        '''
        Assigns success to every row as 1 or 0 depending on whether the target percent was
        reached withing a certain number of candlesticks

        Parameters
        ----------
        candlesticks : INT
            Number of candlesticks over which to determine if success was achieved

        Returns
        -------
        None.

        '''
        title = 'Success at {0:1.1%}'.format(self.target_percent/100)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=candlesticks)
        self.currency_data[title] = self.currency_data['High'].rolling(window=indexer).max() >= self.currency_data['Open'] * (1 + self.target_percent/100)
        self.currency_data[title] = self.currency_data[title].shift(-1)
        
    def make_signals(self):
        '''
        Calculate signal data and append to dataframe

        Returns
        -------
        None.

        '''
        # Shifted data used in subsequent signal calculations
        self.currency_data['Open_Shifted_1'] = self.currency_data['Open'].shift(1)
        self.currency_data['Open_Shifted_2'] = self.currency_data['Open'].shift(2)
        self.currency_data['Close_Shifted_1'] = self.currency_data['Close'].shift(1)
        self.currency_data['Close_Shifted_2'] = self.currency_data['Close'].shift(2)
        self.currency_data['AwesomeOscillator_Shifted_1'] = self.currency_data['AwesomeOscillator'].shift(1)
        self.currency_data['AwesomeOscillator_Shifted_2'] = self.currency_data['AwesomeOscillator'].shift(2)
        self.currency_data['KAMA_Shifted_1'] = self.currency_data['KAMA'].shift(1)
        self.currency_data['KAMA_Shifted_2'] = self.currency_data['KAMA'].shift(2)
        self.currency_data['ROC_125_Shifted_1'] = self.currency_data['ROC_125'].shift(1)
        self.currency_data['ROC_125_Shifted_2'] = self.currency_data['ROC_125'].shift(2)
        self.currency_data['ROC_21_Shifted_1'] = self.currency_data['ROC_21'].shift(1)
        self.currency_data['ROC_21_Shifted_2'] = self.currency_data['ROC_21'].shift(2)
        self.currency_data['RSI_Shifted_1'] = self.currency_data['RSI'].shift(1)
        self.currency_data['Stochastic_Shifted_1'] = self.currency_data['Stochastic'].shift(1)
        self.currency_data['Stochastic_Shifted_2'] = self.currency_data['Stochastic'].shift(2)
        self.currency_data['TSI_Shifted_1'] = self.currency_data['TSI'].shift(1)
        self.currency_data['WilliamsR_Shifted_1'] = self.currency_data['WilliamsR'].shift(1)
        self.currency_data['AroonIndicator_up_Shifted_1'] = self.currency_data['AroonIndicator_up'].shift(1)
        self.currency_data['AroonIndicator_up_Shifted_2'] = self.currency_data['AroonIndicator_up'].shift(2)
        self.currency_data['AroonIndicator_down_Shifted_1'] = self.currency_data['AroonIndicator_down'].shift(1)
        self.currency_data['AroonIndicator_down_Shifted_2'] = self.currency_data['AroonIndicator_down'].shift(2)
        self.currency_data['CCI_Shifted_1'] = self.currency_data['CCI'].shift(1)
        self.currency_data['KST_Shifted_1'] = self.currency_data['KST'].shift(1)
        self.currency_data['KST_Shifted_2'] = self.currency_data['KST'].shift(2)
        self.currency_data['KST_signal_Shifted_1'] = self.currency_data['KST_signal'].shift(1)
        self.currency_data['KST_signal_Shifted_2'] = self.currency_data['KST_signal'].shift(2)
        self.currency_data['MACD_Shifted_1'] = self.currency_data['MACD'].shift(1)
        self.currency_data['MACD_Shifted_2'] = self.currency_data['MACD'].shift(2)
        self.currency_data['MACD_signal_Shifted_1'] = self.currency_data['MACD_signal'].shift(1)
        self.currency_data['MACD_signal_Shifted_2'] = self.currency_data['MACD_signal'].shift(2)
        self.currency_data['TRIX_Shifted_1'] = self.currency_data['TRIX'].shift(1)
        self.currency_data['TRIX_Shifted_2'] = self.currency_data['TRIX'].shift(2)

        # Signal 0: 1 candlestick bullish
        self.currency_data['Signal_0'] = False
        mask = self.currency_data['Close'] > self.currency_data['Open']
        self.currency_data.loc[mask,'Signal_0'] = True
        
        # Signal 1: 2 candlestick bullish
        self.currency_data['Signal_1'] = False
        mask = (self.currency_data['Close'] > self.currency_data['Open']) & \
               (self.currency_data['Close_Shifted_1'] > self.currency_data['Open_Shifted_1'])
        self.currency_data.loc[mask,'Signal_1'] = True

        # Signal 2: 3 candlestick bullish
        self.currency_data['Signal_2'] = False
        mask = (self.currency_data['Close'] > self.currency_data['Open']) & \
               (self.currency_data['Close_Shifted_1'] > self.currency_data['Open_Shifted_1']) & \
               (self.currency_data['Close_Shifted_2'] > self.currency_data['Open_Shifted_2'])
        self.currency_data.loc[mask,'Signal_2'] = True
        

        ##__Momentum Indicators__
        # Signal 3: Awesome osillator crosses above 0 within 2 candlesticks
        self.currency_data['Signal_3'] = False
        mask = ((self.currency_data['AwesomeOscillator'] > 0) & \
               ((self.currency_data['AwesomeOscillator_Shifted_1'] < 0) | \
                (self.currency_data['AwesomeOscillator_Shifted_2'] < 0)))
        self.currency_data.loc[mask,'Signal_3'] = True
        
        # Signal 4: Close crosses above KAMA within 2 candlesticks
        self.currency_data['Signal_4'] = False
        mask = ((self.currency_data['Close'] > self.currency_data['KAMA']) & \
               ((self.currency_data['Close_Shifted_1'] < self.currency_data['KAMA_Shifted_1']) | \
                (self.currency_data['Close_Shifted_2'] < self.currency_data['KAMA_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_4'] = True

        # Signal 5: ROC(125) > 0 and ROC(21) < -8 within 2 candlesticks
        self.currency_data['Signal_5'] = False
        mask = (((self.currency_data['ROC_125'] > 0) | (self.currency_data['ROC_125_Shifted_1'] > 0) | (self.currency_data['ROC_125_Shifted_2'] > 0)) & \
                ((self.currency_data['ROC_21'] < -8) | (self.currency_data['ROC_21_Shifted_1'] < -8) | (self.currency_data['ROC_21_Shifted_2'] < -8)))
        self.currency_data.loc[mask,'Signal_5'] = True
        
        # Signal 6: RSI < 40
        self.currency_data['Signal_6'] = False
        mask = self.currency_data['RSI'] < 40
        self.currency_data.loc[mask,'Signal_6'] = True
        
        # Signal 7: RSI increasing
        self.currency_data['Signal_7'] = False
        mask = (self.currency_data['RSI'] > self.currency_data['RSI_Shifted_1'])
        self.currency_data.loc[mask,'Signal_7'] = True
        
        # Signal 8: Stochastic osillator crosses above 0 within 2 candlesticks
        self.currency_data['Signal_8'] = False
        mask = (self.currency_data['Stochastic'] > 0) & \
               ((self.currency_data['Stochastic_Shifted_1'] < 0) | \
                (self.currency_data['Stochastic_Shifted_2'] < 0))
        self.currency_data.loc[mask,'Signal_8'] = True

        # Signal 9: TSI increasing
        self.currency_data['Signal_9'] = False
        mask = self.currency_data['TSI'] > self.currency_data['TSI_Shifted_1']
        self.currency_data.loc[mask,'Signal_9'] = True

        # Signal 10: Williams R < -80
        self.currency_data['Signal_10'] = False
        mask = self.currency_data['WilliamsR'] < -80
        self.currency_data.loc[mask,'Signal_10'] = True
        
        # Signal 11: Williams R increasing
        self.currency_data['Signal_11'] = False
        mask = self.currency_data['WilliamsR'] > self.currency_data['WilliamsR_Shifted_1']
        self.currency_data.loc[mask,'Signal_11'] = True
        
        
        ##__Volatility Indicators__
        
        # Signal 12: Close < Bband low
        self.currency_data['Signal_12'] = False
        mask = self.currency_data['Close'] < self.currency_data['BolBand_low']
        self.currency_data.loc[mask,'Signal_12'] = True
        
        # Signal 13: High > Bband high
        self.currency_data['Signal_13'] = False
        mask = self.currency_data['High'] > self.currency_data['BolBand_high']
        self.currency_data.loc[mask,'Signal_13'] = True
        
        # Signal 14: High touches Donchian channel upper band
        self.currency_data['Signal_14'] = False
        mask = (self.currency_data['High'] == self.currency_data['DonchianChannel_high'])
        self.currency_data.loc[mask,'Signal_14'] = True
        
        # Signal 15: High above high Keltner channel
        self.currency_data['Signal_15'] = False
        mask = (self.currency_data['High'] > self.currency_data['KeltnerChannel_high'])
        self.currency_data.loc[mask,'Signal_15'] = True


        ##_Trend Indicators_
        # Signal 16: Aroon Up crosses above Aroon Down within 2 candlesticks
        self.currency_data['Signal_16'] = False
        mask = ((self.currency_data['AroonIndicator_up'] > self.currency_data['AroonIndicator_down']) & \
               ((self.currency_data['AroonIndicator_up_Shifted_1'] < self.currency_data['AroonIndicator_down_Shifted_1']) | \
                (self.currency_data['AroonIndicator_up_Shifted_2'] < self.currency_data['AroonIndicator_down_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_16'] = True
        
        # Signal 17: Aroon Up > Aroon down
        self.currency_data['Signal_17'] = False
        mask = (self.currency_data['AroonIndicator_up'] > self.currency_data['AroonIndicator_down'])
        self.currency_data.loc[mask,'Signal_17'] = True
        
        # Signal 18: CCI increasing
        self.currency_data['Signal_18'] = False
        mask = (self.currency_data['CCI'] > self.currency_data['CCI_Shifted_1'])
        self.currency_data.loc[mask,'Signal_18'] = True
        
        # Signal 19: CCI > 100
        self.currency_data['Signal_19'] = False
        mask = self.currency_data['CCI'] > 100
        self.currency_data.loc[mask,'Signal_19'] = True
        
        # Signal 20: KST > 0 and KST crosses above KST signal within 2 candlesticks
        self.currency_data['Signal_20'] = False
        mask = ((self.currency_data['KST'] > 0 & (self.currency_data['KST'] > self.currency_data['KST_signal'])) & \
               ((self.currency_data['KST_Shifted_1'] < self.currency_data['KST_signal_Shifted_1']) | \
                (self.currency_data['KST_Shifted_2'] < self.currency_data['KST_signal_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_20'] = True
        
        # Signal 21: MACD crosses above signal line
        self.currency_data['Signal_21'] = False
        mask = ((self.currency_data['MACD'] > self.currency_data['MACD_signal']) & \
               ((self.currency_data['MACD_Shifted_1'] < self.currency_data['MACD_signal_Shifted_1']) | \
                (self.currency_data['MACD_Shifted_2'] < self.currency_data['MACD_signal_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_21'] = True
        
        # Signal 22: TRIX crosses above 0 within 2 candlesticks
        self.currency_data['Signal_22'] = False
        mask = (self.currency_data['TRIX'] > 0 & \
              ((self.currency_data['TRIX_Shifted_1'] < 0) | \
               (self.currency_data['TRIX_Shifted_2'] < 0)))
        self.currency_data.loc[mask,'Signal_22'] = True
        

        # Reformat data. Drop NAN colums, reset the index back to 0, then drop the "old" index data
        self.currency_data = self.currency_data.dropna().reset_index().drop(columns=['index'])


    def correlation_analysis(self):
        '''
        Test all combinations of signals and compare to success values

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        '''
        signal_data = self.currency_data.filter(items=self.signal_list)
        success_data = self.currency_data.filter(like='Success')
        num_signals = len(signal_data.columns)

        format_string = '0' + str(num_signals) + 'b'
        for combination in range(1, 2**num_signals):
            # Make column which is combination (AND) of selected combination strategy data
            testCombination = [bool(int(i)) for i in format(combination, format_string)]
            indices = [i for i, x in enumerate(testCombination) if x]
            test_df = signal_data.loc[:,['Signal_' + str(i) for i in indices]].all(axis='columns')
            
            # Compare column to "Success" data
            total_signals = np.sum(test_df) # Number of times the indicator suggests a "buy"
            if total_signals:
                for criteria in success_data.columns:
                    correct_signals = np.sum((test_df & success_data.loc[:,criteria])) # Number of times the indicator matches a positive criteria
                    
                if correct_signals:
                    self.winning_strategies.append([correct_signals, total_signals, indices])


def calculate_winning_strategies(pair):
    pair.update_data()
    pair.calculate_indicators()
    pair.assign_success(NUM_CANDLESTICKS)
    pair.make_signals()
    pair.correlation_analysis()
    np.save('Data_5min/' + str(pair.pair)+'.npy', np.array(pair.winning_strategies, dtype=object))

if __name__ == '__main__':
    
    np.save('Data_5min/analysis_parameters.npy', np.array([IMPORT_PERIOD, IMPORT_INTERVAL, TARGET_PERCENT, SIGNALS], dtype=object)) 
    
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    
    # Import list of Forex pairs
    ForexPairs = pd.read_excel('ConversionTable.xlsx', sheet_name='MatrixIndices')
    
    # Loop through pairs
    print('Running...')
    object_list = [CurrencyPair(ForexPairs.iloc[i,:].tolist(), IMPORT_PERIOD, IMPORT_INTERVAL, TARGET_PERCENT, SIGNALS) for i in range(len(ForexPairs))]

    pool.map(calculate_winning_strategies, object_list)
    
    pool.close
