import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

def minmax(x):
    u=(x-min(x))/(max(x)-min(x))
    return u
''''
# Wrangle data:
df = pd.read_csv("wrangling/data/sharks.csv") 
df.head()

y = df['target'] # Set out outcome/target
ctrl_list = [ var_1, var_2, ] # List of control variables
x = df.loc[:, ctrl_list] # Set our covariates/features
u = x.apply(MinMaxScaler) # Scale our variables



k = 5 
# Create a fitted model instance:
model = KNeighborsClassifier(n_neighbors = k) # Create a model instance
model = model.fit(u,y) # Fit the model
# Make predictions:
y_hat = model.predict(u) # Hard prediction


this_plot = sns.scatterplot(x=x1,y=x2,
                hue = y,
                style=y_hat)
sns.move_legend(this_plot, "upper left", bbox_to_anchor=
'''