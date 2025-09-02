# tabletennistrader

Everything you need to start algorithmically trading [Czech Liga Pro table tennis](https://tt.league-pro.com/) odds. Implemented in Python. Includes a webscraper for getting the latest scores, an implementation of a player ratings algorithm, and an implementation of a Monte Carlo simulation algorithm to estimate win probabilities. The player ratings algorithm has several parameters that can be tuned to improve performance.



Daily matchups can be automatically simulated and printed to a spreadsheet, so all you have to do is go to your favorite sports betting website and enter the odds into the spreadsheet, and the spreadsheet will tell you how much you should bet based on the simulations.



This is provided mainly for entertainment purposes. You will probably lose money if you use this software. Please gamble responsibly.

## 

## Installation and usage

Just clone the repo. After installing, I recommend running the Usage Jupyter notebook from start to end.



To use betLog.xlsx, just input the American betting odds for each respective player under bookPlayerOdds (for the player under the player column) and bookOpponentOdds (for the player under the opponent column). The recommended bet for each player will be automatically calculated under playerWager and opponentWager. It is possible that both of these values will be 0, which means that the simulated win probability is close to the win probability implied by the betting odds. When the match is over, enter 'player' or 'opponent' under the winner column depending on who won (you can enter 'push' if the match was cancelled) and the win/loss amount will be calculated under the delta column.

## 

## Dependencies

BeautifulSoup, NumPy, openpyxl, Pandas, sklearn

## 

## Known issues

* Webscraper is unable to process box score data for matches between two players with the same last name and will skip over these matches

## 

## License

This work is released under the MIT License.

