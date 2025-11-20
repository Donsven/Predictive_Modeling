# Logan's Pipeline

## API's
Kalshi: https://docs.kalshi.com/welcome
- Official Kalshi API for data on markets,as well as ability to execute trades

AlphaVantage: https://www.alphavantage.co/?gad_source=1&gad_campaignid=22373825259&gbraid=0AAAAA-6A8dPvpXp69gCs4SmebapE-mJk2&gclid=CjwKCAiAw9vIBhBBEiwAraSATkGjWa40za-SG4O4qmMSt-BuigN8prHF3KIa2RwbuNDU_Y8Qi1SmVhoC_k4QAvD_BwE#about
- provides realtime and historical financial market data includes other asset classes(Stocks,ETFs, Mutual Funds, Crypto)

OddsJam : https://oddsjam.com/odds-api
- Offers real time betting lines for over 100 different sports book

Yfinance : https://ranaroussi.github.io/yfinance/
- unoffical Yahoo finance api to get financial data from Yahoo Finance

SportRadar : https://developer.sportradar.com/getting-started/docs/get-started
- API for all sports data including NBA, NFL, NCAAAF etc

## High Level Control Flow

1. Collect Data from APIs + store it in database
    - MongoDb
    - PostgreSQL
2. Clean and Preprocess Data
    - NumPy
    - Pandas
3. Train the model using data
    - Scikit Learn
    - Tensor Flow
    - Tensorboard (experimentation and testing)
4. Backtest our model on historical data and evaluate performance
    - Matplotlib
5. Deploy Model and adjust bet sizes + handle risk mangement



