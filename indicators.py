'''
DOCSTRING TEXT
'''

import ta


def calculateIndicators(tickerData):
    '''
    Calculates various indicators for the OHLC data

    Parameters
    ----------
    tickerData : DATAFRAME
        Pandas dataframe of OHLC data for a given currency pair.

    Returns
    -------
    tickerData : DATAFRAME
        Pandas dataframe of OHLC data for a given currency pair including
        indicator data

    '''
    Open = tickerData["Open"]
    high = tickerData["High"]
    low = tickerData["Low"]
    close = tickerData["Close"]

    ##_Momentum Indicators_
    #Awesome Oscillator
    awesomeOscillator = ta.momentum.AwesomeOscillatorIndicator(high, low, window1=5, window2=34, fillna=False)
    tickerData['AwesomeOscillator'] = awesomeOscillator.awesome_oscillator()

    #KAMA
    KAMA = ta.momentum.KAMAIndicator(close, window=10, pow1=2, pow2=30, fillna=False)
    tickerData['KAMA'] = KAMA.kama()

    #ROC
    ROC_125 = ta.momentum.ROCIndicator(close, window=125, fillna=False)
    ROC_21 = ta.momentum.ROCIndicator(close, window=21, fillna=False)
    tickerData['ROC_125'] = ROC_125.roc()
    tickerData['ROC_21'] = ROC_21.roc()

    #RSI
    RSI = ta.momentum.RSIIndicator(close, window=14, fillna=False)
    tickerData['RSI'] = RSI.rsi()

    #Stochastic Oscillator
    stochastic = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3, fillna=False)
    tickerData['Stochastic'] = stochastic.stoch()

    #TSI
    TSI = ta.momentum.TSIIndicator(close, window_slow=25, window_fast=13, fillna=False)
    tickerData['TSI'] = TSI.tsi()

    #Williams R
    williams = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14, fillna=False)
    tickerData['WilliamsR'] = williams.williams_r()

    ##_Volatility Indicators_
    #AverageTrueRange
    ATR = ta.volatility.AverageTrueRange(high, low, close, window=14, fillna=False)
    tickerData['ATR'] = ATR.average_true_range()

    # Bollinger Bands
    bollingerBands = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    tickerData['BolBand_low'] = bollingerBands.bollinger_lband()
    tickerData['BolBand_high'] = bollingerBands.bollinger_hband()

    #DonchianChannel
    donchianChannel = ta.volatility.DonchianChannel(high, low, close, window=20, offset=0, fillna=False)
    tickerData['DonchianChannel_low'] = donchianChannel.donchian_channel_lband()
    tickerData['DonchianChannel_high'] = donchianChannel.donchian_channel_hband()

    #KeltnerChannel
    keltnerChannel = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=10, fillna=False, original_version=True)
    tickerData['KeltnerChannel_low'] = keltnerChannel.keltner_channel_lband()
    tickerData['KeltnerChannel_high'] = keltnerChannel.keltner_channel_hband()

    ##_Trend Indicators_
    #AroonIndicator
    AroonIndicator = ta.trend.AroonIndicator(close, window=25, fillna=False)
    tickerData['AroonIndicator_down'] = AroonIndicator.aroon_down()
    tickerData['AroonIndicator_up'] = AroonIndicator.aroon_up()

    #CCIIndicator
    CCIIndicator = ta.trend.CCIIndicator(high, low, close, window=20, constant=0.015, fillna=False)
    tickerData['CCI'] = CCIIndicator.cci()

    #DPOIndicator
    DPOIndicator = ta.trend.DPOIndicator(close, window=20, fillna=False)
    tickerData['DPO'] = DPOIndicator.dpo()

    #EMAIndicator
    EMAIndicator = ta.trend.EMAIndicator(close, window=14, fillna=False)
    tickerData['EMA'] = EMAIndicator.ema_indicator()

    #KSTIndicator
    KSTIndicator = ta.trend.KSTIndicator(close, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9, fillna=False)
    tickerData['KST'] = KSTIndicator.kst()
    tickerData['KST_signal'] = KSTIndicator.kst_sig()

    #MACD
    MACD = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    tickerData['MACD'] = MACD.macd()
    tickerData['MACD_signal'] = MACD.macd_signal()

    #MassIndex
    MassIndex = ta.trend.MassIndex(high, low, window_fast=9, window_slow=25, fillna=False)
    tickerData['MassIndex'] = MassIndex.mass_index()

    #TRIXIndicator
    TRIXIndicator = ta.trend.TRIXIndicator(close, window=15, fillna=False)
    tickerData['TRIX'] = TRIXIndicator.trix()

    # Remove NaN rows at beginning
    tickerData = tickerData.dropna()

    return tickerData
