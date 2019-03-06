# Linear-Regression-Example
Understanding of Linear Regression

Why we use Regression

Regression is an statistical concept used to determine the strength of the relationship between a dependent variable and one or more independent variables.
     
Short Intro to Linear regression

     It is an linear approach to modelling the relationship between variables.
     
Equation of Linear Regression

     Y = mx + c
     
Y = Dependent Var

x = Independent Var

m = Slope

c = Intercept

GOAL:-

we are going to find slope and intercept result that has least errors error means difference between actual value and predicted value.we will try to find best line possible.

Dataset:-
We are going to use Kaggle Automobile Dataset to understand the concept of Linear Regression.(Link below)

     https://www.kaggle.com/toramky/automobile-dataset/version/2

After that we will perform some basic step of understanding the data, convrting to int, removing missing data.

First we have to find the correlation in dataset. so, we can findout which columns has strong correlation.I have used pearson corrleation coeficient. I showed correlation by using both library scipy and pandas.

Next, I used Bokeh to plotting the data in scatter plot.

After That i imported sklearn to use Linear Regression Algorithm and fit both variable in my model and find the slope and interception.

Finally i plotted the best line in my scatter plott.
