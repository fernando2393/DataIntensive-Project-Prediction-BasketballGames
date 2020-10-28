# Prediction Basketball Games

Final project of the KTH ID2221 - Data-Intensive Computing course.

The aim of this project can be split into the following points:
* Data preprocessing in order to develop a dataset of NBA rosters from
1990 to 2018.
* Test different ML-based approaches in order to predict the final results
of NBA matches along a season.

The first bulletpoint has been performed using Scala and Spark, using
the API [Ball Don't Lie](https://www.balldontlie.io). By means of the obtained historical data we have built our own dataset of rosters in base to
the seasonal averages of the team players in each recorded metric.

The second point has been developed using Python, specifically some already
provided methods of the library [SciKit](https://scikit-learn.org/stable/).

The results obtained with our approach are similar to other state-of-the-art methods.

In order to get more information about the method and its implementation, please review
[NBA Outcomes Predictor](NBA_Outcomes_Predictor.pdf).
