# ForexAnalysis
A place for Forex analysis code

The code can be broken up into two sections:
  1) Analysis
  2) Scanning/application


The analysis code using the yfinance module to pull in historic data, then uses the ta module to calculate various indicators. It then assigns each time (row) as a success or failure based on whether or not a specific gain could have been made if that specific currency pair was purchased at that time. It then loops through all possible combinations of the indiators and calculates a percentage of how often that particular set of indicators "won" over how many times it was triggered. The top combinations are then saved to a csv file called StrategyIndex.

The scanning/application code then takes the StrategyIndex list and current currency pair data, calculates the appropriate indicator values and compares it ti the strategylist to see if that particular strategy (combination of indicators) is currently being triggered. If so, when it will send a text message notifying the user. It will also calculate the unit size and a take profit price based on account size.
