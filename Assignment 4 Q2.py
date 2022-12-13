# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:49:07 2022

@author: SANJUSHA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Salary_Data.csv")
df.head()
df.shape
df.isnull().sum() 
# There are no null values
df.describe()

# EDA (boxplot, scatterplot, histogram)
df.boxplot("Salary", vert=False)
Q1=np.percentile(df["Salary"],25)
Q3=np.percentile(df["Salary"],75)
IQR=Q3-Q1 # 43824

df.boxplot("YearsExperience", vert=False)
Q1=np.percentile(df["YearsExperience"],25)
Q3=np.percentile(df["YearsExperience"],75)
IQR=Q3-Q1 # 4.5
# There are no outliers and both the graphs are positively skewed

df.plot.scatter(x="Salary", y="YearsExperience")
# Here, if YearsExperience increases then Salary also increases

df["Salary"].hist()
df["YearsExperience"].hist()
# Both the graphs are not bell shaped and the Salary graph has a gap but not the YearsExperience

df.corr()
# Both variables are strong positively correlated and the correlation b/w the variables is 0.978242


# Splitting the variables
Y = df[["Salary"]]
X = df[["YearsExperience"]]


# Model fitting
# MODEL 1
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
y1 = LR.predict(X)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 31270951.7222
 
rmse = np.sqrt(mse).round(4)
# RMSE = 5592.0436

r2_score(Y,y1)
# r2score = 0.95695 (95%)



# Transformations
# MODEL 2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.log(X),Y)
y1=LR.predict(np.log(X))

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 106149618.7218
 
rmse = np.sqrt(mse).round(4)
# RMSE = 10302.8937

r2_score(Y,y1)
# r2score =  0.85388 (85%)

#create log-transformed data
df_log = np.log(df)
#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)
#create histograms
axs[0].hist(df, edgecolor='black')
axs[1].hist(df_log, edgecolor='black')
#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Log-Transformed Data')



# MODEL 3
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.sqrt(X),Y)
y1=LR.predict(np.sqrt(X))

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 50127755.6165
 
rmse = np.sqrt(mse).round(4)
# RMSE = 7080.0957

r2_score(Y,y1)
# r2score = 0.93100 (93%)

#create sqrt-transformed data
df_sqrt = np.sqrt(df)
#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)
#create histograms
axs[0].hist(df, edgecolor='black')
axs[1].hist(df_sqrt, edgecolor='black')
#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Square Root Transformed Data')



# MODEL 4
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**2,Y)
y1=LR.predict(X**2)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 61520040.4605
 
rmse = np.sqrt(mse).round(4)
# RMSE = 7843.4712

r2_score(Y,y1)
# r2score =  0.91531 (91%)

#create cbrt-transformed data
df_cbrt = np.cbrt(df)
#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)
#create histograms
axs[0].hist(df, edgecolor='black')
axs[1].hist(df_cbrt, edgecolor='black')
#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Cube Root Transformed Data')

# Inference : A prediction model is built and the best model selected is model 1 since its r2score is 95%

