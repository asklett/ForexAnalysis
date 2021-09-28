'''
DOCSTRING TEXT
'''

import numpy as np
import pandas as pd

def makeStrategyData(tickerData):
    '''
    Defines a strategy based on the value of a single indicator and creates a
    new column in the dataframe that is a '1' if the indicator is singaling, or
    '0' if it isnt

    Parameters
    ----------
    tickerData : DATAFRAME
        Pandas dataframe of OHLC data for a given currency pair.

    Returns
    -------
    tickerData : DATAFRAME
        Pandas dataframe of OHLC data for a given currency pair including
        strategy data.

    '''
    #tickerData['Strategy_1'] = tickerData.apply(lambda row: 1 if tickerData.loc[row.name,'Close'] > tickerData.loc[row.name,'Open'] else 0, axis=1) #1-day bullish - GET RID OF THIS ONE TO MAKE THINGS FASTER
    tickerData['Strategy_2'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'Close'] > tickerData.loc[row.name,'Open']) & (tickerData.loc[row.name-1,'Close'] > tickerData.loc[row.name-1,'Open']) else 0) \
                                                if row.name > 0 else np.nan, axis=1) #2-day bullish - GET RID OF THIS ONE TO MAKE THINGS FASTER
    #tickerData['Strategy_3'] = tickerData.apply(lambda row: (1 if ((tickerData.loc[row.name,'Close'] > tickerData.loc[row.name,'Open']) & \
    #                                                               (tickerData.loc[row.name-1,'Close'] > tickerData.loc[row.name-1,'Open']) & \
    #                                                               (tickerData.loc[row.name-2,'Close'] > tickerData.loc[row.name-2,'Open'])) else 0) \
    #                                            if row.name > 1 else np.nan, axis=1) #3-day bullish - GET RID OF THIS ONE TO MAKE THINGS FASTER
    #tickerData['Strategy_4'] = tickerData.apply(lambda row: 1 if (tickerData.loc[row.name,'Volume'] > 1000000) else 0, axis=1) #high volume
    #tickerData['Strategy_5'] = tickerData.apply(lambda row: 1 if (tickerData.loc[row.name,'Volume'] < 1000000) else 0, axis=1) #low volume - GET RID OF THIS ONE TO MAKE THINGS FASTER
    #tickerData['Strategy_6'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'Volume'] > 2*tickerData.loc[row.name-5:row.name-1,'Volume'].mean()) else 0) \
    #                                            if row.name > 4 else np.nan, axis=1) # high relative volume
    tickerData['Strategy_7'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name, 'AwesomeOscillator'] > 0 &\
                                                                   (tickerData.loc[row.name-1, 'AwesomeOscillator'] < 0 or\
                                                                    tickerData.loc[row.name-2, 'AwesomeOscillator'] < 0)) else 0) \
                                                if row.name > 1 else np.nan, axis=1) # Awesome osillator crosses 0 within 3 days
    tickerData['Strategy_8'] = tickerData.apply(lambda row: (1 if ((tickerData.loc[row.name, 'Close'] > tickerData.loc[row.name, 'KAMA']) &\
                                                                   (tickerData.loc[row.name-1, 'Close'] < tickerData.loc[row.name-1, 'KAMA'] or\
                                                                    tickerData.loc[row.name-2, 'Close'] < tickerData.loc[row.name-2, 'KAMA'])) else 0) \
                                                if row.name > 1 else np.nan, axis=1) # Close crosses above KAMA within 3 days
    #tickerData['Strategy_9'] = tickerData.apply(lambda row: (1 if ((tickerData.loc[row.name, 'ROC_125'] > 0 or tickerData.loc[row.name-1, 'ROC_125'] > 0 or tickerData.loc[row.name-2, 'ROC_125'] > 0) & \
    #                                                               (tickerData.loc[row.name, 'ROC_21'] < -8 or tickerData.loc[row.name-1, 'ROC_21'] < -8 or tickerData.loc[row.name-2, 'ROC_21'] < -8)) else 0) \
    #                                            if row.name > 1 else np.nan, axis=1) # ROC(125) > 0 and ROC(21) < -8 within 3 days
    tickerData['Strategy_10'] = tickerData.apply(lambda row: 1 if tickerData.loc[row.name,'RSI'] < 40 else 0, axis=1) # RSI < 40
    tickerData['Strategy_11'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'RSI'] > tickerData.loc[row.name-1,'RSI']) else 0) \
                                                if row.name > 0 else np.nan, axis=1) # RSI increasing
    tickerData['Strategy_12'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name, 'Stochastic'] > 0 &\
                                                                   (tickerData.loc[row.name-1, 'Stochastic'] < 0 or\
                                                                    tickerData.loc[row.name-2, 'Stochastic'] < 0)) else 0) \
                                                if row.name > 1 else np.nan, axis=1) # stochastic osillator crosses 0 within 3 days
    tickerData['Strategy_13'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'TSI'] > tickerData.loc[row.name-1,'TSI']) else 0) \
                                                if row.name > 0 else np.nan, axis=1) # TSI increasing
    tickerData['Strategy_14'] = tickerData.apply(lambda row: 1 if tickerData.loc[row.name,'WilliamsR'] < -80 else 0, axis=1) # Williams R < -80
    tickerData['Strategy_15'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'WilliamsR'] > tickerData.loc[row.name-1,'WilliamsR']) else 0) \
                                                if row.name > 0 else np.nan, axis=1) # Williams R increasing
    #tickerData['Strategy_16'] = tickerData.apply(lambda row: (1 if ((tickerData.loc[row.name, 'Close'] < tickerData.loc[row.name, 'VWAP']) or\
    #                                                               (tickerData.loc[row.name-1, 'Close'] < tickerData.loc[row.name-1, 'VWAP']) or\
    #                                                                (tickerData.loc[row.name-2, 'Close'] < tickerData.loc[row.name-2, 'VWAP'])) else 0) \
    #                                            if row.name > 1 else np.nan, axis=1) # close < VWAP within 3 days
    #tickerData['Strategy_17'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'ADI'] > tickerData.loc[row.name-1,'ADI']) else 0) \
    #                                            if row.name > 0 else np.nan, axis=1) # ADI increasing
    #tickerData['Strategy_18'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name, 'ChaikinMoneyFlow'] > 0 &\
    #                                                               (tickerData.loc[row.name-1, 'ChaikinMoneyFlow'] < 0 or\
    #                                                                tickerData.loc[row.name-2, 'ChaikinMoneyFlow'] < 0)) else 0) \
    #                                            if row.name > 1 else np.nan, axis=1) # ChaikinMoneyFlow crosses 0 within 3 days
    #tickerData['Strategy_19'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'EoM'] > tickerData.loc[row.name-1,'EoM']) else 0) \
    #                                            if row.name > 0 else np.nan, axis=1) # EoM increasing
    #tickerData['Strategy_20'] = tickerData.apply(lambda row: 1 if tickerData.loc[row.name,'ForceIndex'] > 0 else 0, axis=1) # ForceIndex > 0
    #tickerData['Strategy_21'] = tickerData.apply(lambda row: 1 if tickerData.loc[row.name,'MFI'] < 20 else 0, axis=1) # MFI < 20
    #tickerData['Strategy_22'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'MFI'] > tickerData.loc[row.name-1,'MFI']) else 0) \
    #                                            if row.name > 0 else np.nan, axis=1) # MFI increasing
    #tickerData['Strategy_23'] = # NVI above 255 day EMA
    #tickerData['Strategy_24'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'OBV'] > tickerData.loc[row.name-1,'OBV']) else 0) \
    #                                            if row.name > 0 else np.nan, axis=1) # OBV increasing
    #tickerData['Strategy_25'] = # [ATR(250) / SMA(20,Close) * 100 < 4]
    tickerData['Strategy_26'] = tickerData.apply(lambda row: 1 if (tickerData.loc[row.name,'Close'] < tickerData.loc[row.name,'BolBand_low']) else 0, axis=1) # Close < Bband low
    tickerData['Strategy_27'] = tickerData.apply(lambda row: 1 if (tickerData.loc[row.name,'High'] > tickerData.loc[row.name,'BolBand_high']) else 0, axis=1) # High > Bband high
    #tickerData['Strategy_28'] = tickerData.apply(lambda row: 1 if (tickerData.loc[row.name,'High'] == tickerData.loc[row.name,'DonchianChannel_high']) else 0, axis=1) # high touches donchian channel upper band
    tickerData['Strategy_29'] = tickerData.apply(lambda row: 1 if (tickerData.loc[row.name,'High'] > tickerData.loc[row.name,'KeltnerChannel_high']) else 0, axis=1) # high above high kleltner channel
    tickerData['Strategy_30'] = tickerData.apply(lambda row: (1 if ((tickerData.loc[row.name, 'AroonIndicator_up'] > tickerData.loc[row.name, 'AroonIndicator_down']) &\
                                                                   (tickerData.loc[row.name-1, 'AroonIndicator_up'] < tickerData.loc[row.name-1, 'AroonIndicator_down'] or\
                                                                    tickerData.loc[row.name-2, 'AroonIndicator_up'] < tickerData.loc[row.name-2, 'AroonIndicator_down'])) else 0) \
                                                if row.name > 1 else np.nan, axis=1) # Aroon Up crosses above Aroon Down within 3 days
    tickerData['Strategy_31'] = tickerData.apply(lambda row: 1 if (tickerData.loc[row.name,'AroonIndicator_up'] > tickerData.loc[row.name,'AroonIndicator_down']) else 0, axis=1) # Aroon Up > Aroon down
    tickerData['Strategy_32'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name,'CCI'] > tickerData.loc[row.name-1,'CCI']) else 0) \
                                                if row.name > 0 else np.nan, axis=1) # CCI increasing
    #tickerData['Strategy_33'] = tickerData.apply(lambda row: 1 if tickerData.loc[row.name,'CCI'] > 100 else 0, axis=1) # CCI > 100
    tickerData['Strategy_34'] = tickerData.apply(lambda row: (1 if ((tickerData.loc[row.name, 'KST'] > 0 & (tickerData.loc[row.name, 'KST'] > tickerData.loc[row.name, 'KST_signal'])) &\
                                                                   (tickerData.loc[row.name-1, 'KST'] < tickerData.loc[row.name-1, 'KST_signal'] or\
                                                                    tickerData.loc[row.name-2, 'KST'] < tickerData.loc[row.name-2, 'KST_signal'])) else 0) \
                                                if row.name > 1 else np.nan, axis=1) # KST > 0 and KST crosses above KST signal within 3 days
    #tickerData['Strategy_35'] = tickerData.apply(lambda row: (1 if ((tickerData.loc[row.name, 'MACD'] > tickerData.loc[row.name, 'MACD_signal']) &\
    #                                                               (tickerData.loc[row.name-1, 'MACD'] < tickerData.loc[row.name-1, 'MACD_signal'] or\
    #                                                                tickerData.loc[row.name-2, 'MACD'] < tickerData.loc[row.name-2, 'MACD_signal'])) else 0) \
    #                                            if row.name > 1 else np.nan, axis=1) # MACD crosses above signaml line
    tickerData['Strategy_36'] = tickerData.apply(lambda row: (1 if (tickerData.loc[row.name, 'TRIX'] > 0 &\
                                                                   (tickerData.loc[row.name-1, 'TRIX'] < 0 or\
                                                                    tickerData.loc[row.name-2, 'TRIX'] < 0)) else 0) \
                                                if row.name > 1 else np.nan, axis=1) # TRIX crosses above 0

    # Reformat data. Drop NAN colums, reset the index back to 0, then drop the "old" index data
    tickerData = tickerData.dropna().reset_index().drop(columns=['index'])

    return tickerData

def checkStrategies(tickerData):
    '''
    Checks all combinations of strategies that were previously identified in the
    analysis portion of this work (found in StrategyIndex.csv) and determines
    if that particular group of strategies is signaling or not

    Parameters
    ----------
    tickerData : DATAFRAME
        Pandas dataframe of OHLC and strategy data

    Returns
    -------
    list
        List of probabilities for all combinations of strategies that the trade
        will be successful

    '''
    strategies = pd.read_csv('StrategyIndex.csv')
    numStrats = len(tickerData.columns)

    todayPicks = np.zeros((len(strategies),2))
    for n, strategy in enumerate(strategies['StratIndex']):
        # Make column which is combination (AND) of selected combination strategy data
        testCombination = ['0' if x == ' ' else x for x in list(('{0:' + str(numStrats) + 'b}').format(strategy))]
        testCombination = [int(i) for i in testCombination]

        #loop through indicies and AND them with todays Data
        strategyMatch = 1
        for m, index in enumerate(testCombination):
            strategyMatch &= index == tickerData.iloc[-1,-(m+1)]

        if strategyMatch:
            todayPicks[n,:] = [1, strategies['Probability'][n]]

    if np.sum(todayPicks[:,0]):
        todaysWeightedPercentage = np.sum(todayPicks[:,1])/np.sum(todayPicks[:,0])
        todaysMaxPercentage = np.max(todayPicks[:,1])
    else:
        todaysWeightedPercentage = 0
        todaysMaxPercentage = 0

    return [todaysWeightedPercentage,todaysMaxPercentage]
