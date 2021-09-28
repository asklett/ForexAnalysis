'''
TO DO
- Make the analysis program more automatic so the strategies can be updated for the current market
- Integrate this whole process into an app that can run on the phone...no need for texting, just push alerts from apps :D
- Add a text alert to alert when something that was suggested goes below a certain price
- Try to do some math to figure out high side instead of a hardcoded %. Like Fib levels, or stDevs, or something
- Record which strategy is being used
- Make figure of chart to send with text
- Fix open/close times let user choose "w/ Sun/Fri" or "w/o Sun/Fri"
- Fix import of pairs and creation of conversion table so # isnt harcoded
'''

import logging
from math import floor, ceil
from time import sleep
from datetime import date, datetime
import numpy as np
import pandas as pd
import yfinance as yf
from indicators import calculateIndicators
from correlations import makeStrategyData, checkStrategies
from twilio_messaging import send_sms


# Create logger
logging.basicConfig(filename='ForexLog.log', format = '%(asctime)s - %(message)s', level=logging.INFO)

# Analysis parameters
#valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
IMPORT_PERIOD = '1mo'
IMPORT_INTERVAL = '30m'
TARGET_PERCENT = 0.05 # Target for percent profit (in fraction form) as a percentage of the margin used
MAX_MARGIN = 2.5 # Max margin per position in USD used to calculate # units to buy
FREQUENCY = 60 # Time in minutes between scans

# Import pairs list and create conversion table to calculate margin for each pair
ForexPairs = pd.read_excel('ConversionTable.xlsx', sheet_name='MatrixIndices')
conversionTable = np.identity(20)

# Continually run
print('Running...')
while True:
    # Check to make sure market is open
    while (date.weekday(datetime.now()) == 4 and datetime.time(datetime.now()).hour > 16) \
           or (date.weekday(datetime.now()) == 5) \
           or (date.weekday(datetime.now()) == 6 and datetime.time(datetime.now()).hour < 17):
        pass

    # Populate conversion rate matrix with updated data
    print('Populating conversion rate matrix with updated data')
    ERROR = 0
    try:
        for index, row in ForexPairs.iterrows():
            tickerData = yf.Ticker(row['Pair']).history(period='1d', interval=IMPORT_INTERVAL)
            conversionTable[row['Pair1_Row'],row['Pair1_Column']] = tickerData['Close'][-1]
            conversionTable[row['Pair1_Column'],row['Pair1_Row']] = 1/tickerData['Close'][-1]
    except:
        ERROR = 1

    # Loop through each currency pair and analyze
    if not ERROR:
        print('Starting next scan at ', str(datetime.now())[0:19])
        todayPicks = np.zeros((len(ForexPairs),4), dtype=object)
        for index, row in ForexPairs.iterrows():
            try:
                # Import data
                tickerData = yf.Ticker(row['Pair']).history(period=IMPORT_PERIOD, interval=IMPORT_INTERVAL)

                # Calculate indicator values
                tickerData = calculateIndicators(tickerData)
                tickerData = tickerData.reset_index()

                # Calculate strategy indicators
                tickerData = makeStrategyData(tickerData)

                # Compare to strategies
                filteredTickerData = tickerData.filter(like='Strategy').astype('int64')
                weightedPercentage = checkStrategies(filteredTickerData)

                # Send text if opportunity is found
                if weightedPercentage[1] > 0:
                    # Calculate units and pips to take
                    conversionFactor = conversionTable[row['Pair1_Row'], row['Pair1_Column']] * conversionTable[row['Pair2_Row'], row['Pair2_Column']]
                    pips = floor((TARGET_PERCENT * row['MarginRatio'] * conversionTable[row['Pair1_Row'], row['Pair1_Column']])/row['Pip'])
                    units = ceil(MAX_MARGIN/(row['MarginRatio'] * conversionFactor))
                    profit = pips * units * conversionTable[row['Pair2_Row'], row['Pair2_Column']] * row['Pip']
                    
                    # Log to log file
                    logging.info(row['Pair'][0:6] + ', ' + str(units) + ' units, ' + str(pips) + ' pips')
                    
                    # Send text message
                    send_sms([row['Pair'][0:6], units, pips, profit])

            except:
                print('Exception encountered in ' + row['Pair'][0:6] + '. Passing to next pair.')
                logging.info('Exception encountered in ' + row['Pair'][0:6] + '. Passing to next pair.')

    if ERROR:
        print('Error populating conversion rate matrix. Sleeping for 1 min.')
        sleep(60)
    else:
        print('Finished another cycle at ', str(datetime.now())[0:19], '. Sleeping for ', str(FREQUENCY), ' mins.')
        sleep(60 * FREQUENCY)
