# APIs
## Basketbal APIs
NBA: https://github.com/swar/nba_api
Scraping: https://www.basketball-reference.com/

## Baseball APIs
MLB: https://statsapi.mlb.com/

## Kalshi API
https://docs.kalshi.com/welcome?redirect=%2F

# Control Flow

## 1. Collect Data
Get NBA + MLB data from APIs and get injury list too maybe. Use Python for API calls with aiohttp async calls.

## 2. Clean Data and Normalization
Pandas and Numpy to filter and clean data. 

## 3. Feature Engineering
Past performance (face-offs), Offensive/defensive ratings, Player availability, Pitch speed/batting stats (for MLB), Kalshi market metrics (price, volume, spread)

Scikit-Learn

## 4. Model Training
Logistic Regression, Random Forest, Time-series models, Simple NN

PyTorch, Scikit-Learn

## 5. Test Preditctions on Kalshi/Fine-Tuning
For a few days, put custom parameters and see what our model will predict. Wait for outcome to come and see how our model did

FastAPI, Docker

## 6. Track Record
Track PnL

PostgreSQL/SQLite or Amazon S3 Bucket

## 7. Evaluate Performance
Plot different graphs/metrics of wins vs. losses or net money

Pandas/Matplotlib or Streamlit