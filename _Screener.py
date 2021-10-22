'''
TO DO
- Integrate this whole process into an app that can run on the phone...no need for texting, just push alerts from apps :D
- Add a text alert to alert when something that was suggested goes below a certain price
- Try to do some math to figure out high side instead of a hardcoded %. Like Fib levels, or stDevs, or something. ATR?
- Record which strategy is being used
- Make figure of chart to send with text
- Fix import of pairs and creation of conversion table so # isnt harcoded
- Get live data from Oanda API
- Fix open/close times let user choose "w/ Sun/Fri" or "w/o Sun/Fri"

'''

import logging
from math import floor, ceil
from time import sleep
from datetime import date, datetime
import configparser
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from twilio.rest import Client
from github import Github


def send_mms(file_path, details):
    '''
    Sends a multi-media text message (MMS)

    Parameters
    ----------
    file_path : STRING
        File path of the .png image that is being sent

    details : LIST
        List of items to include in message, including pair name, quantity (units)
        to buy, target pips, and the projected profit

    Returns
    -------
    None.

    '''

    # Read in image file
    with open(file_path, 'rb') as file:
        content = file.read()

    # Upload to github
    git = Github(config['DEFAULT']['GitHubAccessToken'])
    repo = git.get_user().get_repo('ForexCharts')
    git_file = 'Chart.png'
    repo.create_file(git_file, "committing files", content, branch="main")

    # Send MMS message
    account_sid = config['DEFAULT']['Twilio_sid']
    auth_token = config['DEFAULT']['Twilio_token']

    client = Client(account_sid, auth_token)

    msg = ('\n\nPair: {0}' \
        + '\nUnits: {1}' \
        + '\nPips: {2} for ${3:1.2}' \
        + '\nTime: {4} EST').format(details[0], details[1], details[2], \
                                    details[3], str(datetime.now())[5:16])

    client.messages \
        .create(
            body = msg,
            from_= config['DEFAULT']['Twilio_number'],
            media_url = 'http://raw.githubusercontent.com/asklett/ForexCharts/main/Chart.png',
            to = config['DEFAULT']['to_field']
            )

    # Delete file from github
    contents = repo.get_contents('Chart.png')
    repo.delete_file(contents.path, "committing files", contents.sha, branch="main")

def send_sms(details):
    '''
    Sends a standard SMS message

    Parameters
    ----------
    details : LIST
        List of items to include in message, including pair name, quantity (units)
        to buy, target pips, and the projected profit

    Returns
    -------
    None.

    '''

    account_sid = config['DEFAULT']['Twilio_sid']
    auth_token = config['DEFAULT']['Twilio_token']

    client = Client(account_sid, auth_token)

    msg = ('\n\nPair: {0}' \
        + '\nUnits: {1}' \
        + '\nPips: {2} for ${3:1.2}' \
        + '\nTime: {4} EST').format(details[0], details[1], details[2], \
                                    details[3], str(datetime.now())[5:16])

    client.messages \
        .create(
            body= msg,
            from_= config['DEFAULT']['Twilio_number'],
            to = config['DEFAULT']['to_field']
            )

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

    def __init__(self, details, period, interval, target_percent, signals):
        '''
        valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

        Parameters
        ----------
        details : LIST
            List of pair-specific details including name and margin rate.
        period : INT
            Period of time over which to analyze data.
        interval : INT
            Interval over which to analyze data (e.g. 30 min candlesticks).
        target_percent : INT
            Target percent for which success will be assigned a 1 if reached.
            Input as a percent (not fraction), i.e., 95, not 0.95.
        signal_list : LIST
            List of signals over which to iterate. Needs to be the same list
            used in the analysis

        Returns
        -------
        None.

        '''
        self.pair = details[0]
        self.margin_rate = details[1]
        self.import_period = period
        self.import_interval = interval
        self.target_percent = target_percent * self.margin_rate
        self.signal_list = ['Signal_' + str(x) for x in signals]
        self.winning_strategies = np.load('Data_Parallel/' + str(self.pair)+'.npy', allow_pickle=True)

    def update_data(self):
        '''
        Update currency data

        Returns
        -------
        None.

        '''
        self.currency_data = yf.Ticker(self.pair).history(period=self.import_period, \
                                                          interval=self.import_interval)

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
        self.currency_data['Signal_0'] = 0
        mask = self.currency_data['Close'] > self.currency_data['Open']
        self.currency_data.loc[mask,'Signal_0'] = 1
        
        # Signal 1: 2 candlestick bullish
        self.currency_data['Signal_1'] = 0
        mask = (self.currency_data['Close'] > self.currency_data['Open']) & \
               (self.currency_data['Close_Shifted_1'] > self.currency_data['Open_Shifted_1'])
        self.currency_data.loc[mask,'Signal_1'] = 1

        # Signal 2: 3 candlestick bullish
        self.currency_data['Signal_2'] = 0
        mask = (self.currency_data['Close'] > self.currency_data['Open']) & \
               (self.currency_data['Close_Shifted_1'] > self.currency_data['Open_Shifted_1']) & \
               (self.currency_data['Close_Shifted_2'] > self.currency_data['Open_Shifted_2'])
        self.currency_data.loc[mask,'Signal_2'] = 1
        

        ##__Momentum Indicators__
        # Signal 3: Awesome osillator crosses above 0 within 2 candlesticks
        self.currency_data['Signal_3'] = 0
        mask = ((self.currency_data['AwesomeOscillator'] > 0) & \
               ((self.currency_data['AwesomeOscillator_Shifted_1'] < 0) | \
                (self.currency_data['AwesomeOscillator_Shifted_2'] < 0)))
        self.currency_data.loc[mask,'Signal_3'] = 1
        
        # Signal 4: Close crosses above KAMA within 2 candlesticks
        self.currency_data['Signal_4'] = 0
        mask = ((self.currency_data['Close'] > self.currency_data['KAMA']) & \
               ((self.currency_data['Close_Shifted_1'] < self.currency_data['KAMA_Shifted_1']) | \
                (self.currency_data['Close_Shifted_2'] < self.currency_data['KAMA_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_4'] = 1

        # Signal 5: ROC(125) > 0 and ROC(21) < -8 within 2 candlesticks
        self.currency_data['Signal_5'] = 0
        mask = (((self.currency_data['ROC_125'] > 0) | (self.currency_data['ROC_125_Shifted_1'] > 0) | (self.currency_data['ROC_125_Shifted_2'] > 0)) & \
                ((self.currency_data['ROC_21'] < -8) | (self.currency_data['ROC_21_Shifted_1'] < -8) | (self.currency_data['ROC_21_Shifted_2'] < -8)))
        self.currency_data.loc[mask,'Signal_5'] = 1
        
        # Signal 6: RSI < 40
        self.currency_data['Signal_6'] = 0
        mask = self.currency_data['RSI'] < 40
        self.currency_data.loc[mask,'Signal_6'] = 1
        
        # Signal 7: RSI increasing
        self.currency_data['Signal_7'] = 0
        mask = (self.currency_data['RSI'] > self.currency_data['RSI_Shifted_1'])
        self.currency_data.loc[mask,'Signal_7'] = 1
        
        # Signal 8: Stochastic osillator crosses above 0 within 2 candlesticks
        self.currency_data['Signal_8'] = 0
        mask = (self.currency_data['Stochastic'] > 0) & \
               ((self.currency_data['Stochastic_Shifted_1'] < 0) | \
                (self.currency_data['Stochastic_Shifted_2'] < 0))
        self.currency_data.loc[mask,'Signal_8'] = 1

        # Signal 9: TSI increasing
        self.currency_data['Signal_9'] = 0
        mask = self.currency_data['TSI'] > self.currency_data['TSI_Shifted_1']
        self.currency_data.loc[mask,'Signal_9'] = 1

        # Signal 10: Williams R < -80
        self.currency_data['Signal_10'] = 0
        mask = self.currency_data['WilliamsR'] < -80
        self.currency_data.loc[mask,'Signal_10'] = 1
        
        # Signal 11: Williams R increasing
        self.currency_data['Signal_11'] = 0
        mask = self.currency_data['WilliamsR'] > self.currency_data['WilliamsR_Shifted_1']
        self.currency_data.loc[mask,'Signal_11'] = 1
        
        
        ##__Volatility Indicators__
        
        # Signal 12: Close < Bband low
        self.currency_data['Signal_12'] = 0
        mask = self.currency_data['Close'] < self.currency_data['BolBand_low']
        self.currency_data.loc[mask,'Signal_12'] = 1
        
        # Signal 13: High > Bband high
        self.currency_data['Signal_13'] = 0
        mask = self.currency_data['High'] > self.currency_data['BolBand_high']
        self.currency_data.loc[mask,'Signal_13'] = 1
        
        # Signal 14: High touches Donchian channel upper band
        self.currency_data['Signal_14'] = 0
        mask = (self.currency_data['High'] == self.currency_data['DonchianChannel_high'])
        self.currency_data.loc[mask,'Signal_14'] = 1
        
        # Signal 15: High above high Keltner channel
        self.currency_data['Signal_15'] = 0
        mask = (self.currency_data['High'] > self.currency_data['KeltnerChannel_high'])
        self.currency_data.loc[mask,'Signal_15'] = 1


        ##_Trend Indicators_
        # Signal 16: Aroon Up crosses above Aroon Down within 2 candlesticks
        self.currency_data['Signal_16'] = 0
        mask = ((self.currency_data['AroonIndicator_up'] > self.currency_data['AroonIndicator_down']) & \
               ((self.currency_data['AroonIndicator_up_Shifted_1'] < self.currency_data['AroonIndicator_down_Shifted_1']) | \
                (self.currency_data['AroonIndicator_up_Shifted_2'] < self.currency_data['AroonIndicator_down_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_16'] = 1
        
        # Signal 17: Aroon Up > Aroon down
        self.currency_data['Signal_17'] = 0
        mask = (self.currency_data['AroonIndicator_up'] > self.currency_data['AroonIndicator_down'])
        self.currency_data.loc[mask,'Signal_17'] = 1
        
        # Signal 18: CCI increasing
        self.currency_data['Signal_18'] = 0
        mask = (self.currency_data['CCI'] > self.currency_data['CCI_Shifted_1'])
        self.currency_data.loc[mask,'Signal_18'] = 1
        
        # Signal 19: CCI > 100
        self.currency_data['Signal_19'] = 0
        mask = self.currency_data['CCI'] > 100
        self.currency_data.loc[mask,'Signal_19'] = 1
        
        # Signal 20: KST > 0 and KST crosses above KST signal within 2 candlesticks
        self.currency_data['Signal_20'] = 0
        mask = ((self.currency_data['KST'] > 0 & (self.currency_data['KST'] > self.currency_data['KST_signal'])) & \
               ((self.currency_data['KST_Shifted_1'] < self.currency_data['KST_signal_Shifted_1']) | \
                (self.currency_data['KST_Shifted_2'] < self.currency_data['KST_signal_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_20'] = 1
        
        # Signal 21: MACD crosses above signal line
        self.currency_data['Signal_21'] = 0
        mask = ((self.currency_data['MACD'] > self.currency_data['MACD_signal']) & \
               ((self.currency_data['MACD_Shifted_1'] < self.currency_data['MACD_signal_Shifted_1']) | \
                (self.currency_data['MACD_Shifted_2'] < self.currency_data['MACD_signal_Shifted_2'])))
        self.currency_data.loc[mask,'Signal_21'] = 1
        
        # Signal 22: TRIX crosses above 0 within 2 candlesticks
        self.currency_data['Signal_22'] = 0
        mask = (self.currency_data['TRIX'] > 0 & \
              ((self.currency_data['TRIX_Shifted_1'] < 0) | \
               (self.currency_data['TRIX_Shifted_2'] < 0)))
        self.currency_data.loc[mask,'Signal_22'] = 1
        

        # Reformat data. Drop NAN colums, reset the index back to 0, then drop the "old" index data
        self.currency_data = self.currency_data.dropna().reset_index().drop(columns=['index'])

    def correlation_analysis(self):
        '''
        Test combinations of signals supplied by analysis

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        '''
        signal_data = self.currency_data.filter(items=self.signal_list).astype('int64')

        count = 0
        used_strategies = []
        for combination in self.winning_strategies:
            indices = combination
            combinedVarAND = 1

            for j in indices:
                combinedVarAND &= signal_data.iloc[-1, j]

            if combinedVarAND > 0:
                count += 1
                used_strategies.append(combination)
            else:
                pass

        return count, used_strategies


# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Dictionary of weekdays
days = {'Sunday': 6, 'Friday': 4, 'Saturday': 5}

# Create logger
logging.basicConfig(filename='ForexLog.log', format = '%(asctime)s - %(message)s', level=logging.INFO)

# Analysis parameters
ANALYSIS_PARAMETERS = np.load('Data_Parallel/analysis_parameters.npy', allow_pickle=True)
IMPORT_PERIOD = ANALYSIS_PARAMETERS[0]
IMPORT_INTERVAL = ANALYSIS_PARAMETERS[1]
TARGET_PERCENT = ANALYSIS_PARAMETERS[2]
SIGNALS = ANALYSIS_PARAMETERS[3]
MAX_MARGIN = 2.5 # Max margin per position in USD used to calc # units to buy
FREQUENCY = 60 # Time in minutes between scans

# Import pairs list and create conversion table to calculate margin for each pair
ForexPairs = pd.read_excel('ConversionTable.xlsx', sheet_name='MatrixIndices')
conversion_table = np.identity(20)

# Create object list of pairs. Needs to have same signal list used in analysis
pair_obj_list = [CurrencyPair(ForexPairs.iloc[i,:].tolist(), IMPORT_PERIOD, IMPORT_INTERVAL, TARGET_PERCENT, SIGNALS) for i in range(len(ForexPairs))]

# Continually run
print('Running...')
while True:
    # Check to make sure market is open
    print('Verifying open market.')
    while (date.weekday(datetime.now()) == days['Friday'] and datetime.time(datetime.now()).hour > 16) \
            or (date.weekday(datetime.now()) == days['Saturday']) \
            or (date.weekday(datetime.now()) == days['Sunday'] and datetime.time(datetime.now()).hour < 17):
        pass

    # Populate conversion rate matrix with updated data
    print('Populating conversion rate matrix with updated data.')
    ERROR = 0
    try:
        for index, row in ForexPairs.iterrows():
            tickerData = yf.Ticker(row['Pair']).history(period='1d', interval=IMPORT_INTERVAL)
            conversion_table[row['Pair1_Row'],row['Pair1_Column']] = tickerData['Close'][-1]
            conversion_table[row['Pair1_Column'],row['Pair1_Row']] = 1/tickerData['Close'][-1]
    except:
        ERROR = 1

    # Loop through each currency pair and analyze
    if not ERROR:
        print('Starting next scan at ', str(datetime.now())[0:19])
        for index, row in ForexPairs.iterrows():
            try:
                # Import data
                pair_obj_list[index].update_data()
                
                # Check to see if analysis list is empty
                if pair_obj_list[index].winning_strategies.size == 0:
                    continue

                # Calculate indicator values
                pair_obj_list[index].calculate_indicators()

                # Calculate signals
                pair_obj_list[index].make_signals()

                # Compare to strategies
                found_opportunity, strategies = pair_obj_list[index].correlation_analysis()

                # Send text if opportunity is found
                if found_opportunity:
                    # Calculate units and pips to take
                    conversionFactor = conversion_table[row['Pair1_Row'], row['Pair1_Column']] * conversion_table[row['Pair2_Row'], row['Pair2_Column']]
                    pips = floor((TARGET_PERCENT/100 * row['MarginRatio'] * conversion_table[row['Pair1_Row'], row['Pair1_Column']])/row['Pip'])
                    units = ceil(MAX_MARGIN/(row['MarginRatio'] * conversionFactor))
                    profit = pips * units * conversion_table[row['Pair2_Row'], row['Pair2_Column']] * row['Pip']

                    # Log to file
                    logging.info(row['Pair'][0:6] + ', ' + str(units) + ' units, ' + str(pips) + ' pips. Strategies: ' + str(strategies))

                    # Send text message
                    send_sms([row['Pair'][0:6], units, pips, profit])
                    print(row['Pair'][0:6] + ', ' + str(units) + ' units, ' + str(pips) + ' pips')

            except:
                print('Exception encountered in ' + row['Pair'][0:6] + '. Passing to next pair.')
                logging.info('Exception encountered in ' + row['Pair'][0:6] + '. Passing to next pair.')

    if ERROR:
        print('Error populating conversion rate matrix. Sleeping for 1 min.')
        sleep(60)
    else:
        print('Finished another cycle at ', str(datetime.now())[0:19], '. Sleeping for ', str(FREQUENCY), ' mins.')
        sleep(60 * FREQUENCY)
