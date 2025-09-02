from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import json
import numpy as np
import openpyxl
import os
import pandas as pd
import random
import requests
import sklearn.linear_model
import time

playerTranscode = pd.read_pickle('czechLigaProTtPlayerTranscode.pkl')

def getTournaments(date):
    dt = datetime.strptime(date, '%Y-%m-%d')
    url = f'https://tt.league-pro.com/tours/?year={dt.year}&month={dt.month}&day={dt.day}'
    r = requests.get(url)
    html = BeautifulSoup(r.text, 'html.parser')
    parseTournaments = html.find_all('td', {'class': 'tournament-name'})
    tournaments = []
    for t in parseTournaments:
        try:
            name = t.text
            tournamentId = int(t.a['href'].split('tours/')[1])
            tournaments.append([name, tournamentId])
        except:
            pass
    return tournaments

def getMatchIds(tournamentId):
    url = f'https://tt.league-pro.com/tours/{tournamentId}'
    r = requests.get(url)
    html = BeautifulSoup(r.text, 'html.parser')
    parseMatches = html.find_all('a', {'class': 'undrr bold'})
    matches = []
    for m in parseMatches:
        try:
            matches.append(int(m['href'].split('games/')[1]))
        except:
            pass
    return list(set(matches))

def getPoints(r):
    gameTable = pd.read_html(r.content)[-1]
    players = list(set(gameTable['Serve'].dropna()))
    away = players[0]
    home = players[1]
    points = {}
    awayServicePoints = 0
    awayReturnPoints = 0
    homeServicePoints = 0
    homeReturnPoints = 0
    for i in range(len(gameTable)):
        r = gameTable.iloc[i]
        if r['Serve'] == away:
            if r['Won'] == away:
                awayServicePoints += 1
            elif r['Won'] == home:
                homeReturnPoints += 1
        elif r['Serve'] == home:
            if r['Won'] == home:
                homeServicePoints += 1
            elif r['Won'] == away:
                awayReturnPoints += 1
        elif r['Event'] == 'end of set':
            points[len(points)] = {away: {'servicePoints': awayServicePoints, 'returnPoints': awayReturnPoints}, 
                                   home: {'servicePoints': homeServicePoints, 'returnPoints': homeReturnPoints}}
            awayServicePoints = 0
            awayReturnPoints = 0
            homeServicePoints = 0
            homeReturnPoints = 0
    return points

def getInfo(r):
    html = BeautifulSoup(r.text, 'html.parser')
    parsePlayers = html.find_all('div', {'class': 'col-4 player'})
    awayName = parsePlayers[0].find_all('a')[0].text
    awayPlayerId = parsePlayers[0].find_all('a')[0]['href'].split('/')[1]
    homeName = parsePlayers[1].find_all('a')[0].text
    homePlayerId = parsePlayers[1].find_all('a')[0]['href'].split('/')[1]
    parseDate = html.find_all('title')
    date = parseDate[0].text.split(' - ')[0]
    convertedDate = datetime.strftime(datetime.strptime(date, '%d %B %Y'), '%Y-%m-%d')
    return {'away': {'name': awayName, 'playerId': awayPlayerId}, 
            'home': {'name': homeName, 'playerId': homePlayerId}, 
            'date': convertedDate}

def setEnded(a, b):
    if (a < 11) and (b < 11):
        return False
    if (a == 11) and (b < 10):
        return True
    if (b == 11) and (a < 10):
        return True
    if (a > 11) and (a - b == 2):
        return True
    if (b > 11) and (b - a == 2):
        return True
    return False

def webscrapeMatch(matchId):
    print(f'Scraping match {matchId}')
    time.sleep(0.1)
    url = f'https://tt.league-pro.com/games/{matchId}'
    r = requests.get(url)
    try:
        info = getInfo(r)
        points = getPoints(r)
        awayName = info['away']['name']
        homeName = info['home']['name']
        awayNameAbb = awayName.split(' ')[0]
        homeNameAbb = homeName.split(' ')[0]
        setsWon = {'away': 0, 'home': 0}
        s = []
        infoRow = {'date': info['date'], 'matchId': matchId, 'awayName': awayName, 'awayId': info['away']['playerId'], 'homeName': homeName, 'homeId': info['home']['playerId']}
        for p in range(len(points)):
            infoRow_ = infoRow.copy()
            infoRow_['setId'] = p
            infoRow_['awayServicePoints'] = points[p][awayNameAbb]['servicePoints']
            infoRow_['awayReturnPoints'] = points[p][awayNameAbb]['returnPoints']
            infoRow_['homeServicePoints'] = points[p][homeNameAbb]['servicePoints']
            infoRow_['homeReturnPoints'] = points[p][homeNameAbb]['returnPoints']
            awayPoints = infoRow_['awayServicePoints'] + infoRow_['awayReturnPoints']
            homePoints = infoRow_['homeServicePoints'] + infoRow_['homeReturnPoints']
            if setEnded(awayPoints, homePoints):
                s.append(infoRow_)
                if awayPoints > homePoints:
                    setsWon['away'] += 1
                else:
                    setsWon['home'] += 1
        if (setsWon['away'] == 3) or (setsWon['home'] == 3):
            return s
        else:
            print(f'Incomplete/invalid score data, skipping {matchId}')
            return 0
    except:
        print(f'Failed to parse data, skipping {matchId}')
        return 0

def getBoxScores(startDate = None, endDate = None):
    if os.path.exists('./Box Scores/czechLigaProTtUnprocessed.pkl'):
        df = pd.read_pickle('./Box Scores/czechLigaProTtUnprocessed.pkl')
        # Get matchIds that have already been scraped so that they don't have to be re-scraped
        scrapedMatchIds = set(df['matchId'])
    else:
        scrapedMatchIds = []
    if startDate == None:
        if os.path.exists('./Box Scores/czechLigaProTtUnprocessed.pkl'):
            startDate = datetime.strftime(datetime.strptime(max(df['date']), '%Y-%m-%d') - timedelta(days = 1), '%Y-%m-%d')
        else:
            # timedelta(hours = 2) syncs the time with the times listed on https://tt.league-pro.com/
            startDate = datetime.strftime(datetime.today().astimezone(timezone(timedelta(hours = 2))), '%Y-%m-%d')
    if endDate == None:
        endDate = datetime.strftime(datetime.today().astimezone(timezone(timedelta(hours = 2))), '%Y-%m-%d')
    webscrapeDt = datetime.strptime(startDate, '%Y-%m-%d')
    webscrapeDate = datetime.strftime(webscrapeDt, '%Y-%m-%d')
    while webscrapeDate <= endDate:
        print(f'Scraping {webscrapeDate}')
        tournaments = getTournaments(webscrapeDate)
        for tournament in tournaments:
            df = []
            print(f'Scraping {tournament[0]}')
            matchIds = getMatchIds(tournament[1])
            for matchId in matchIds:
                if matchId not in scrapedMatchIds:
                    ws = webscrapeMatch(matchId)
                    if ws != 0:
                        df.extend(ws)
            df = pd.DataFrame(df)
            # Box scores file is updated and saved after every tournament
            if os.path.exists('./Box Scores/czechLigaProTtUnprocessed.pkl'):
                Df = pd.read_pickle('./Box Scores/czechLigaProTtUnprocessed.pkl')
                df = pd.concat([Df, df], ignore_index = True)
                df = df.sort_values(by = ['date', 'matchId', 'setId'])
                df = df.drop_duplicates(subset = ['date', 'matchId', 'setId']).reset_index(drop = True)
                df.to_pickle('./Box Scores/czechLigaProTtUnprocessed.pkl')
            else:
                df.to_pickle('./Box Scores/czechLigaProTtUnprocessed.pkl')
            print('\n')
        webscrapeDt += timedelta(days = 1)
        webscrapeDate = datetime.strftime(webscrapeDt, '%Y-%m-%d')
    return df

def getPlayerName(playerId):
    time.sleep(0.1)
    url = f'https://tt.league-pro.com/players/{playerId}'
    r = requests.get(url)
    html = BeautifulSoup(r.text, 'html.parser')
    parseName = html.find_all('title')
    name = parseName[0].text.split(' | ')[0]
    return name

def processBoxScores():
    # This function performs two processing transformations on the unprocessed data, each of which are essentially 
    # a way of detecting and fixing data anomalies. The first transformation involves checking the box score data 
    # for each row (recall that each row in the table represents a set of table tennis) and making sure that the 
    # reported scores are logically valid. Any rows corresponding to logically invalid scores are filtered out 
    # and thus will not appear in the processed box score data.
    print('Processing box scores...')
    df = pd.read_pickle('./Box Scores/czechLigaProTtUnprocessed.pkl')
    df['awayPoints'] = df.apply(lambda x: x['awayServicePoints'] + x['awayReturnPoints'], axis = 1)
    df['homePoints'] = df.apply(lambda x: x['homeServicePoints'] + x['homeReturnPoints'], axis = 1)
    df['awayServes'] = df.apply(lambda x: x['awayServicePoints'] + x['homeReturnPoints'], axis = 1)
    df['homeServes'] = df.apply(lambda x: x['homeServicePoints'] + x['awayReturnPoints'], axis = 1)
    df['setEnded'] = df.apply(lambda x: setEnded(x['awayPoints'], x['homePoints']), axis = 1)
    df['validScore'] = df.apply(lambda x: (np.abs(x['awayServes'] - x['homeServes']) < 3) 
                                and setEnded(x['awayPoints'], x['homePoints']), axis = 1)
    df = df[df['validScore']].copy()
    df = df.drop(['awayPoints', 'homePoints', 'awayServes', 'homeServes', 'validScore'], axis = 1)
    df = df.sort_values(by = ['date', 'matchId', 'setId'])
    df = df.reset_index(drop = True)
    
    # This block of code maintains the player transcode table, which is a security measure to ensure that players with 
    # similar or identical names are properly identified as distinct. For instance, there exist two players both named 
    # Strnad Jaroslav. Luckily, in this particular case, the Czech Liga Pro website already differentiates between the 
    # two players by referring to one player as "Strnad Jaroslav 1961" and referring to the other player as 
    # "Strnad Jaroslav 1964". However, if two distinct players (i.e. two players with different numerical player IDs) 
    # are ever detected to have the exact same name, this block of code will print a flag. However, no error will be 
    # raised. The solution in this case would be to manually edit the transcode table file and assign distinct names 
    # to the two different player IDs. As of the time of writing this, I have never had to do this.
    playerTranscode = pd.read_pickle('czechLigaProTtPlayerTranscode.pkl')
    boxScorePlayerIds = set(list(df['awayId']) + list(df['homeId']))
    newPlayerIds = [playerId for playerId in boxScorePlayerIds if playerId not in playerTranscode.index]
    newPlayerNames = []
    for playerId in newPlayerIds:
        newPlayerNames.append(getPlayerName(playerId))
    newPlayerTranscode = pd.Series(data = newPlayerNames, index = newPlayerIds)
    playerTranscode = pd.concat([playerTranscode, newPlayerTranscode])
    playerTranscode = playerTranscode.sort_index()
    unique, counts = np.unique(playerTranscode.values, return_counts = True)
    duplicates = unique[counts > 1]
    if len(duplicates) > 0:
        print('Duplicate names detected!')
        print(duplicates)
    playerTranscode.to_pickle('czechLigaProTtPlayerTranscode.pkl')
    
    # After the transcode table is updated, the player names are updated according to the transcode table. This is 
    # the final transformation in the processing function, and the processed box score data is saved and ready to be passed 
    # into the ratings pipeline.
    df['awayName'] = df['awayId'].map(playerTranscode)
    df['homeName'] = df['homeId'].map(playerTranscode)
    df.to_pickle('./Box Scores/czechLigaProTtProcessed.pkl')
    return df

def logit(x):
    return np.log(x / (1 - x))

def logistic(x):
    return (1 / (1 + np.exp(-1 * x)))

class Pipeline:
    def __init__(self, additiveSmoothing = 1, ratingsWindowDays = 360, ratingsHalfLifeDays = 60, regularizationAlpha = 0.01):
        self.boxScores = pd.read_pickle('./Box Scores/czechLigaProTtProcessed.pkl')
        self.modelParameters = {'additiveSmoothing': additiveSmoothing, 
                                'ratingsWindowDays': ratingsWindowDays, 
                                'ratingsHalfLifeDays': ratingsHalfLifeDays, 
                                'regularizationAlpha': regularizationAlpha}
    
    def execute(self, date = None):
        if date == None:
            self.dt = datetime.strptime(max(self.boxScores['date']), '%Y-%m-%d') + timedelta(days = 1)
        else:
            self.dt = datetime.strptime(date, '%Y-%m-%d')
        self.date = datetime.strftime(self.dt, '%Y-%m-%d')
        self.getCorpus()
        self.getMuRatings()
        self.getEpsilonRatings()

    def getCorpus(self):
        print('Preparing ratings corpus...')
        ratingsFromDate = datetime.strftime(self.dt - timedelta(days = self.modelParameters['ratingsWindowDays']), '%Y-%m-%d')
        self.corpus = self.boxScores[(self.boxScores['date'] < self.date) & (self.boxScores['date'] >= ratingsFromDate)].copy()
        self.players = sorted(list(set(list(self.corpus['homeName']) + list(self.corpus['awayName']))))
        # Apply exponential time decay
        self.corpus['weight'] = self.corpus.apply(lambda x: (0.5 ** ((self.dt - datetime.strptime(x['date'], '%Y-%m-%d')).days 
                                                                     / self.modelParameters['ratingsHalfLifeDays'])), axis = 1)
        # Serve rating represents the ability for a player to win their service points against an opponent. Additive 
        # smoothing ensures that logit is defined even for the edge cases where a player wins/loses 100% of their service 
        # points.
        c = self.modelParameters['additiveSmoothing']
        self.corpus['homeServe'] = self.corpus.apply(lambda x: logit((x['homeServicePoints'] + c) 
                                                                     / (x['homeServicePoints'] + x['awayReturnPoints'] 
                                                                        + (2 * c))), axis = 1)
        self.corpus['awayServe'] = self.corpus.apply(lambda x: logit((x['awayServicePoints'] + c) 
                                                                     / (x['awayServicePoints'] + x['homeReturnPoints'] 
                                                                        + (2 * c))), axis = 1)
    
    def getMuRatings(self):
        # Each player's mu rating is a pair of numbers, the first of which represents how likely they are to win a point 
        # when they are serving, and the second of which represents how likely they are to win a point when they are 
        # returning. Essentially, this is a 2-dimensional generalization of the standard Elo rating system. 
        # See https://en.wikipedia.org/wiki/Elo_rating_system#Theory for details.
        print('Solving mu ratings...')
        eloComponents = ['serveElo', 'returnElo']
        # Preallocate Numpy arrays
        AElo = np.zeros((2 * len(self.corpus), len(eloComponents) * len(self.players)), dtype = np.float32)
        bElo = np.zeros(2 * len(self.corpus), dtype = np.float32)
        for i in range(len(self.corpus)):
            g = self.corpus.iloc[i]
            # Weight corresponds to the row's contribution to the total squared loss, so for a row to contribute 
            # by a factor of w, we multiply the row by sqrt(w)
            w = np.sqrt(g['weight'])
            home = g['homeName']
            away = g['awayName']
            # Fit home_{serveElo} - away_{returnElo} = homeServe
            AElo[2 * i, self.players.index(home) + (len(self.players) * eloComponents.index('serveElo'))] = w
            AElo[2 * i, self.players.index(away) + (len(self.players) * eloComponents.index('returnElo'))] = -1 * w
            bElo[2 * i] = w * g['homeServe']
            # Fit away_{serveElo} - home_{returnElo} = awayServe
            AElo[(2 * i) + 1, self.players.index(away) + (len(self.players) * eloComponents.index('serveElo'))] = w
            AElo[(2 * i) + 1, self.players.index(home) + (len(self.players) * eloComponents.index('returnElo'))] = -1 * w
            bElo[(2 * i) + 1] = w * g['awayServe']
        # Note that this system is under-determined, so we use ridge regression to ensure the solution is stable
        muRatings = list(sklearn.linear_model.Ridge(alpha = self.modelParameters['regularizationAlpha'], 
                                                    fit_intercept = False).fit(AElo, bElo).coef_)
        # Decompose and organize solution vector into ratings table
        playerMuRatings = []
        for i in range(len(eloComponents)):
            playerMuRatings.append(muRatings[(len(self.players) * i):(len(self.players) * (i + 1))])
        playerMuRatings = pd.DataFrame(np.transpose(np.concatenate([[self.players], playerMuRatings], axis = 0)), 
                                       columns = ['player'] + eloComponents)
        playerMuRatings = playerMuRatings.set_index('player')
        for column in playerMuRatings.columns:
            if column == 'player':
                playerMuRatings[column] = playerMuRatings[column].astype('string')
            else:
                playerMuRatings[column] = playerMuRatings[column].astype('float')
        self.playerMuRatings = playerMuRatings.copy()
        self.playerMuRatings.to_pickle(f'./Trader Parameters/{self.date}czechLigaProTtPlayerMuRatings.pkl')
    
    def getServeBar(self, x, side):
        homeRatings = self.playerMuRatings.loc[x['homeName']]
        awayRatings = self.playerMuRatings.loc[x['awayName']]
        if side == 'home':
            return homeRatings['serveElo'] - awayRatings['returnElo']
        if side == 'away':
            return awayRatings['serveElo'] - homeRatings['returnElo']
    
    def getEpsilonRatings(self):
        # Each player's epsilon rating is essentially a measure of how consistent they are at playing near their 
        # computed mu ratings. The idea is that when we eventually simulate a match between two player's, we can 
        # combine their two respective epsilon matrices to get an estimate of the expected variation from the 
        # projected serve ratings implied by their mu ratings.
        print('Solving epsilon ratings...')
        # serveBar is the mean serve ratings for each row as implied by the calculated mu ratings
        self.corpus['homeServeBar'] = self.corpus.apply(lambda x: self.getServeBar(x, 'home'), axis = 1)
        self.corpus['awayServeBar'] = self.corpus.apply(lambda x: self.getServeBar(x, 'away'), axis = 1)
        # epsilon is the variation between serve bar and the actual serve rating 
        self.corpus['homeServeEpsilon'] = self.corpus.apply(lambda x: x['homeServe'] - x['homeServeBar'], axis = 1)
        self.corpus['awayServeEpsilon'] = self.corpus.apply(lambda x: x['awayServe'] - x['awayServeBar'], axis = 1)
        playerEpsilonRatings = []
        for player in self.players:
            X = []
            Y = []
            weights = []
            homeSub = self.corpus[self.corpus['homeName'] == player]
            # We attribute half of the observed variation to the player (and the other half is attributed to their opponents). 
            # Recall that Var(aX) = a^2 * Var(X). Thus, we need to multiply the data by sqrt(0.5).
            X.extend(list(np.sqrt(0.5) * homeSub['homeServeEpsilon']))
            Y.extend(list(-1 * np.sqrt(0.5) * homeSub['awayServeEpsilon']))
            weights.extend(list(homeSub['weight']))
            awaySub = self.corpus[self.corpus['awayName'] == player]
            X.extend(list(np.sqrt(0.5) * awaySub['awayServeEpsilon']))
            Y.extend(list(-1 * np.sqrt(0.5) * awaySub['homeServeEpsilon']))
            weights.extend(list(awaySub['weight']))
            covarianceMatrix = np.cov(np.array([X, Y]), aweights = weights, ddof = 0)
            playerEpsilonRatings.append([player, covarianceMatrix])
        # Constuct ratings table
        playerEpsilonRatings = pd.DataFrame(playerEpsilonRatings, columns = ['player', 'covarianceMatrix'])
        for column in playerEpsilonRatings.columns:
            if column == 'player':
                playerEpsilonRatings[column] = playerEpsilonRatings[column].astype('string')
            else:
                playerEpsilonRatings[column] = playerEpsilonRatings[column].astype('object')
        playerEpsilonRatings = playerEpsilonRatings.set_index('player')
        self.playerEpsilonRatings = playerEpsilonRatings.copy()
        self.playerEpsilonRatings.to_pickle(f'./Trader Parameters/{self.date}czechLigaProTtPlayerEpsilonRatings.pkl')

def playSet(A, B, playerToServeFirst):
    # A: Probability of Alice winning the point when Alice is serving
    # B: Probability of Bob winning the point when Bob is serving
    # playerToServeFirst: Player who gets to serve first in this set
    
    # Initialize points won scoreboard
    alicePoints, bobPoints = 0, 0
    # Initialize serve counter which will be used to determine which players serves which points
    serveCount = 0
    # Initialize serving player variable which tracks the player currently serving
    servingPlayer = playerToServeFirst
    # Loop until winner of set is determined
    while True:
        # Simulate Alice service point
        if servingPlayer == 'Alice':
            if random.random() < A:
                alicePoints += 1
            else:
                bobPoints += 1
        # Simulate Bob service point
        elif servingPlayer == 'Bob':
            if random.random() < B:
                bobPoints += 1
            else:
                alicePoints += 1
        serveCount += 1
        # Per rules, serve alternates every two points during first 20 points and afterwards alternates after every point
        if (serveCount % 2 == 0) or (alicePoints >= 10 and bobPoints >= 10):
            servingPlayer = 'Bob' if servingPlayer == 'Alice' else 'Alice'
        # Check for winner
        if alicePoints >= 11 and alicePoints - bobPoints >= 2:
            return 'Alice'
        if bobPoints >= 11 and bobPoints - alicePoints >= 2:
            return 'Bob'

def playMatch(a, b, cholesky):
    # a: Mean probability of Alice winning the point when Alice is serving (in logit units)
    # b: Mean probability of Bob winning the point when Bob is serving (in logit units)
    # cholesky: Cholesky decomposition of covariance matrix of the corresponding service point win probabilities (in logit units)

    # Initialize sets won scoreboard
    aliceSets, bobSets = 0, 0
    # Coin toss to decide the player to serve first in the first set
    playerToServeFirst = 'Alice' if random.random() < 0.5 else 'Bob'
    # Check whether match is completed
    while aliceSets < 3 and bobSets < 3:
        # Sample from covariance matrix
        sim = np.dot(cholesky, np.random.normal(size = 2))
        # Compute service point win probabilities
        A = logistic(a + sim[0])
        B = logistic(b + sim[1])
        # Play set with sampled service point win probabilities
        winner = playSet(A, B, playerToServeFirst)
        # Increment scoreboard
        if winner == 'Alice':
            aliceSets += 1
        else:
            bobSets += 1
        # Alternate layer to serve first between each set
        playerToServeFirst = 'Bob' if playerToServeFirst == 'Alice' else 'Alice'
    # Determine winner of the match
    if aliceSets == 3:
        return 'Alice'
    else:
        return 'Bob'

def kelly(p, odds):
    # See https://en.wikipedia.org/wiki/Kelly_criterion for details
    if odds < 0:
        b = -100 / odds
    else:
        b = odds / 100
    return (p - ((1 - p) / b))

def boundedBetSize(w, odds, minBet, maxBet):
    if odds < 0:
        w = min(w, maxBet * odds / -100)
        if w < (minBet * odds / -100):
            return 0
        return int(np.floor(w))
    else:
        w = min(w, maxBet)
        if w < minBet:
            return 0
        return int(np.floor(w))

def getTournamentMatchups(tournament):
    url = f'https://tt.league-pro.com/tours/{tournament}'
    r = requests.get(url)
    html = BeautifulSoup(r.text, 'html.parser')
    players = [p.a['href'].split('players/')[1] for p in html.find_all('tr', attrs = {'id': True})]
    players = [p for p in players if p in playerTranscode.index.values]
    return [(playerTranscode.loc[p], playerTranscode.loc[q]) for p in players for q in players if int(p) < int(q)]

def getPretraderMatchups(date):
    print(f'Fetching tournaments for {date}...')
    tournaments = getTournaments(date)
    matchups = []
    for i in range(len(tournaments)):
        tournament = tournaments[i]
        print(f'Fetching matchups for tournament {tournament[0]}... ({i + 1}/{len(tournaments)})')
        matchups.extend(getTournamentMatchups(tournament[1]))
    return matchups

class Trader:
    def __init__(self, date = None, kellyBasis = 1000, minSize = 10, maxSize = 100):
        self.boxScores = pd.read_pickle('./Box Scores/czechLigaProTtProcessed.pkl')
        if date == None:
            self.dt = datetime.strptime(max(self.boxScores['date']), '%Y-%m-%d') + timedelta(days = 1)
        else:
            self.dt = datetime.strptime(date, '%Y-%m-%d')
        self.date = datetime.strftime(self.dt, '%Y-%m-%d')
        self.traderParameters = {'kellyBasis': kellyBasis, 'minSize': minSize, 'maxSize': maxSize}
        try:
            self.playerMuRatings = pd.read_pickle(f'./Trader Parameters/{self.date}czechLigaProTtPlayerMuRatings.pkl')
            self.playerEpsilonRatings = pd.read_pickle(f'./Trader Parameters/{self.date}czechLigaProTtPlayerEpsilonRatings.pkl')
            self.players = sorted(list(self.playerMuRatings.index), key = lambda s: s.lower())
        except:
            print(f'No ratings files found for {self.date}; either run the pipeline or load old ratings')
    
    def playerSearch(self, search):
        return [player for player in self.players if search.lower() in player.lower()]
    
    def getEpsilon(self, player, opponent):
        playerMatrix = self.playerEpsilonRatings.loc[player]['covarianceMatrix']
        opponentMatrix = self.playerEpsilonRatings.loc[opponent]['covarianceMatrix']
        # Since we are concerned with the difference of two random variables, we multiply by -1
        covariance = -1 * (playerMatrix[0][1] + opponentMatrix[0][1])
        return np.array([[playerMatrix[0][0] + opponentMatrix[1][1], covariance], 
                         [covariance, opponentMatrix[0][0] + playerMatrix[1][1]]])
    
    def simulate(self, player, opponent):
        playerMuRatings = self.playerMuRatings.loc[player]
        opponentMuRatings = self.playerMuRatings.loc[opponent]
        # a is the expected value of the player's serve rating
        # b is the expected value of the opponent's serve rating
        a = playerMuRatings['serveElo'] - opponentMuRatings['returnElo']
        b = opponentMuRatings['serveElo'] - playerMuRatings['returnElo']
        # epsilon matrix represents the expected variation from a and b
        # See https://en.wikipedia.org/wiki/Cholesky_decomposition#Monte_Carlo_simulation for details on usage of 
        # the Cholesky decomposition here
        cholesky = np.linalg.cholesky(self.getEpsilon(player, opponent))
        n = 0
        # N is the number of Monte Carlo simulations and can be increased to increase precision at the expense of 
        # increased runtime
        N = 20000
        for i in range(N):
            winner = playMatch(a, b, cholesky)
            if winner == 'Alice':
                n += 1
        return round(n / N, 4)
        
    def bet(self, player, opponent, playerOdds = None, opponentOdds = None, printOutput = True, writeToBetLog = False):
        winProbability = self.simulate(player, opponent)
        kellyBasis = self.traderParameters['kellyBasis']
        minSize = self.traderParameters['minSize']
        maxSize = self.traderParameters['maxSize']
        if printOutput:
            print(f'{player} price: {max(0, 100 * (winProbability)):.2f}%')
            print(f'{opponent} price: {max(0, 100 * ((1 - winProbability))):.2f}%')
            if (playerOdds != None) and (opponentOdds != None):
                playerBet = boundedBetSize(kellyBasis * kelly(winProbability, playerOdds), playerOdds, minSize, maxSize)
                opponentBet = boundedBetSize(kellyBasis * kelly(1 - winProbability, opponentOdds), opponentOdds, minSize, maxSize)
                print('\n')
                print(f'{player} bet: {playerBet}')
                print(f'{opponent} bet: {opponentBet}')
        if writeToBetLog:
            workbook = openpyxl.load_workbook(filename = './betLog.xlsx')
            betLog = workbook['betLog']
            row = betLog.max_row + 1
            betLog[f'A{row}'] = self.date
            betLog[f'B{row}'] = row - 2
            betLog[f'C{row}'] = player
            betLog[f'D{row}'] = opponent
            betLog[f'F{row}'] = playerOdds
            betLog[f'G{row}'] = opponentOdds
            betLog[f'H{row}'] = f'=IF(OR(F{row}="",F{row}="-"),"-",IF(F{row}<0,(-1*F{row})/(100-F{row}),100/(100+F{row})))'
            betLog[f'I{row}'] = f'=IF(OR(G{row}="",G{row}="-"),"-",IF(G{row}<0,(-1*G{row})/(100-G{row}),100/(100+G{row})))'
            betLog[f'J{row}'] = winProbability
            betLog[f'K{row}'] = f'=1-J{row}'
            betLog[f'L{row}'] = (f'=IF(H{row}="-",0,ROUND(MIN(IF({kellyBasis}*MAX(0,(J{row}-((1-J{row})/((1/H{row})-1))))'
                                 f'<IF(F{row}<0,-{minSize}*F{row}/100,{minSize}),0,{kellyBasis}*MAX(0,(J{row}-((1-J{row})'
                                 f'/((1/H{row})-1))))),IF(F{row}<0,-{maxSize}*F{row}/100,{maxSize}))))')
            betLog[f'M{row}'] = (f'=IF(I{row}="-",0,ROUND(MIN(IF({kellyBasis}*MAX(0,(K{row}-((1-K{row})/((1/I{row})-1))))'
                                 f'<IF(G{row}<0,-{minSize}*G{row}/100,{minSize}),0,{kellyBasis}*MAX(0,(K{row}-((1-K{row})'
                                 f'/((1/I{row})-1))))),IF(G{row}<0,-{maxSize}*G{row}/100,{maxSize}))))')
            betLog[f'N{row}'] = f'=MAX(L{row},M{row})'
            betLog[f'O{row}'] = f'=IF(N{row}=0,"nobet",IF(E{row}="","pending",IF(P{row}>0,"win",IF(P{row}<0,"loss","push"))))'
            betLog[f'P{row}'] = (f'=ROUND(IF(OR(F{row}="",G{row}=""),0,IF(E{row}="player",(L{row}*((1/H{row})-1))-M{row},'
                                 f'IF(E{row}="opponent",(M{row}*((1/J{row})-1))-L{row},0))),2)')
            betLog[f'Q{row}'] = f'=Q{row - 1}+P{row}'
            workbook.save(filename = './betLog.xlsx')
        
    def pretrade(self):
        matchups = getPretraderMatchups(self.date)
        for i in range(len(matchups)):
            matchup = sorted(matchups[i])
            print(f'Simulating {matchup[0]} vs. {matchup[1]}... ({i + 1}/{len(matchups)})')
            try:
                self.bet(matchup[0], matchup[1], printOutput = False, writeToBetLog = True)
            except:
                print('Player(s) not found, skipping...')