# ForexAnalysis
A place for Forex analysis code

The code is broken up into two parts:
  1) Analysis
  2) Screener


The analysis code uses the yfinance module to pull in historic data, then uses the ta module to calculate various indicators. It then assigns each time (row) as a success or failure based on whether or not a specific gain could have been made if that specific currency pair was purchased at that time. It then loops through all possible combinations of the indiators and calculates a percentage of how often that particular set of indicators "won" over how many times it was triggered. The combinations are then saved as numpy arrays to be used by the screener.

The screener code then takes the numpy arrays and current currency pair data, calculates the appropriate indicator values and compares it to the array to see if that particular strategy (combination of indicators) is currently being triggered. If so, when it will send a text message notifying the user. It will also calculate the unit size and a take profit price based on account size.
