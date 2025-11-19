**Sport API**

## API sports
site: https://api-sports.io/
- cover multiple sports

## sports data io
site: https://sportsdata.io/
- mainly NFL, NBA, MLB, NHL and others

## ThesportsDB
site: https://www.thesportsdb.com/
- free JSON sports API for scores, teams, players, and artwork

## RapidAPI
site: https://rapidapi.com/user/api-sports


**Finance APIs**

## Alpha Vantage
site: https://www.alphavantage.co  
- for real time and historical stocks and technical indicators 

## Marketstack
site: https://marketstack.com  
- real time and historical EOD price data for many global exchanges


**Workflow**
1. data integration
- pull data from sports API and save the raw data into storage folder

2. clean and organize
- use pandas to clean and standardize values
- create table to save those processed data into data warehouse

- Postgres/BigQuery/Snowflake

3. data preparation
- split data into training, validation, and test sets

4. build and train the model
- use PyTorch to build a simple transformer model 
- train model using deep learning framworkm and predict the target output

5. evaluate
- look at error metrics and some plots to see if predictions look reasonable by using the data outside of training
- analyze the errors and refine the model

6. prediction
- package the final model into an inference service

- FastAPI

7. monitoring and maintenance
- log predictions and track real-world performance over time
- retraining workflows if drift or degradation is detected.
