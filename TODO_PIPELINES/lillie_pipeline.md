## APIs/Data Resources:
1. https://collegefootballdata.com/
- advertises free data on NCAA football
- not sure if it is just FBS or FCS also
- not sure on its speed for updates or accuracy but seems like a starting place
2. https://www.basketball-reference.com/
- another supposedly free site on NBA, ABA, G League, and WNBA
- team, player, league information
- not sure how the data is downloaded
3. https://www.sports-reference.com/
- more general sports from basketball-reference
- includes more sports include baseball, football pro and college, basketball pro and college
- not sure on update time but has good historical data
- not betting specific
4. https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data
- money lines betting information for NBA games
5. https://www.kaggle.com/datasets/scottfree/sports-lines
- betting information for line, over/under, and game results for select seasons of select sports
- offer variety and also an AlphaPy python model to analyze the trend data in the game results

## High Level WorkFlow
1. Collect data from APIs or data resources
2. Filter and Clean Data into desired values and parameters
3. Split the data into train and test
4. Fit a linear regression model
5. Evaluate accuracy --> RSME, MAE, R^2
6. Adjust and improve
