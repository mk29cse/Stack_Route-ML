# Stack_Route-ML-1-IPL DATA Analysis


The problem statement is attached along with the repo - "Problem_Statement.pdf"

The repo consists of 2 files namely Assign1_1-final.ipynb and IPl_Story-final.ipynb


Assign1_1-final.ipynb 
--------------------------

The file contains the solution to the problem specified in the problem statement. A brief description of the problem solved is discussed below.
	-> Plot a histogram for "venue" TODO: Group by Seasons
	   
	   Data from file matches.csv and deliveries.csv are loaded. Then a crosstab between season and venue is created.
	   The crosstab contains data groupby season and contains count of number of matches played at each stadium in 
           that season

	-> Filter the Matches by id for the period match id number 500 - 550, and plot for "venue"
	   
	   We select the matches with match id between 500 and 550 and then dataFrame is grouped by venue and number of 
	   matches played at each venue is counted. Next the required graph is generated.

	-> Plot the aggregated Number of Runs per Match, first by Venue followed by Innings
	   
	   For this we first merged the data from matches and deliveries on match id. Then we seperately created dataFrame consisting of 	    inning 1 and inning 2. Next we generate a crosstab between venue and total runs. Later we calulated the runs scored in each 		   innings and plotted the graph.



IPl_Story-final.ipynb
--------------------------

This file tries to extract meaningful information from the data provided. This is done in order to tell a story via graphs, regarding the data available.
The code included in the contains 2 major functions whose sole job is to do the following:
	->batsman overview
	  This contains 4 functions. These are used to generate graph for top score scored by batsman under different conditions.
	  It generates graph for 
		-> top score by batsman for a specific team in a specific season,
                -> top score by batsman for a specific team across all season
		-> top score by batsman across all team in a specific season,
                -> top score by batsman across all team across all season
	->bowler overview
	  This contains 4 functions. These are used to generate graph for top wicket taking bowler under different conditions.
	  It generates graph for 
		-> top wicket taking bowler for a specific team in a specific season,
                -> top wicket taking bowler for a specific team across all season
		-> top wicket taking bowler across all team in a specific season,
                -> top wicket taking bowler across all team across all season

