import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

data = pd.read_csv("Automobile_data.csv")

# understanding data
print(data.head())
print(data.info())
print(data.describe())

# converting columns object to  numeric
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')

# removing missing data
data.dropna(subset=['price', 'horsepower'], inplace=True)

# finding pearson correlation
print(pearsonr(data['horsepower'], data['price']))

# correlation without scipy
print(data.corr(method='pearson'))

from bokeh.io import output_notebook
import bokeh.plotting as bp

output_notebook()

source = bp.ColumnDataSource(data=dict(x=data['horsepower'], y=data['price'], make=data['make']))
tooltips = [('make', '@make'), ('horsepower', '$x'), ('price', '$y{$0}')]

plot = bp.figure(plot_width=600, plot_height=400, tooltips=tooltips)
plot.xaxis.axis_label = 'Horsepower'
plot.yaxis.axis_label = 'Price'

plot.circle('x', 'y', source=source, color='blue', size=8, alpha=0.5)

bp.show(plot)

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.25)

from sklearn import linear_model
model = linear_model.LinearRegression()

#reshape rows are unknown and column is 1

training_x = np.array(train['horsepower']).reshape(-1, 1)
training_y = np.array(train['price'])

test_x = np.array(test['horsepower']).reshape(-1, 1)
test_y = np.array(test['price'])

model.fit(training_x, training_y)

slop = np.asscalar(np.squeeze(model.coef_))
intercept = model.intercept_

print(slop, intercept)

from bokeh.models import Slope
best_line = Slope(gradient=slop, y_intercept=intercept, line_color='red', line_width=3)
plot.add_layout(best_line)

# to show output in direct on html file
bp.output_file('columnDataSource.html', title = 'ColumnDataSource')
bp.show(plot)

# Comparing the Actual Value with Predicted Value Most Important part it will say your model can predict well or not.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def predict_matrics(lr, x, y):
    pred = lr.predict(x)
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    return mae, mse, r2


training_mae, training_mse, training_r2 = predict_matrics(model, training_x, training_y)

testing_mae, testing_mse, testing_r2 = predict_matrics(model, test_x, test_y)

print('Training MAE', training_mae, 'Training MSE', training_mse, 'Training R2', training_r2)

print('Testing MAE', testing_mae, 'Testing MSE', testing_mse, 'Testing R2', testing_r2)