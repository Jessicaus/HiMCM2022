#!/usr/bin/env python
# coding: utf-8

#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import statsmodels.api as sm

# csv file name
filename = "co2.csv "
 
# initializing the titles and rows list
fields = []
rows = []
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
 
    # get total number of rows
    line_numbers = (csvreader.line_num)
    print(line_numbers)

years = []
co2conc = []
for item in rows:
    years.append(float(item[0])-1959)
for item in rows:
    co2conc.append(float(item[1]))
bar_y=sum(co2conc)/len(co2conc)
print(bar_y)
for i in range(len(co2conc)):
    co2conc[i]=co2conc[i]-bar_y

slope, intercept, r, p, std_err = stats.linregress(years, co2conc)


def helplinear(years):
  return slope * years + intercept

def linear_regression():
    mymodel = list(map(helplinear, years))
    return mymodel
    
    
def poly_regression():
    mymodel = np.poly1d(np.polyfit(years, co2conc, 3))

    myline = np.linspace(1, 100, 100)
    print(mymodel)
    model=mymodel(years)
    return model

def drawplot(model):
    plt.scatter(years, co2conc)
    plt.plot(years, model)
    plt.show()

def residual(f_model):
    resid = []
    i=0
    for point in f_model:
        residvalue = co2conc[i]-point
        resid.append(residvalue)
        i+=1
    return resid


polymodel = []
polymodel=poly_regression()
linearmodel = linear_regression()
print(polymodel)
# drawplot(polymodel)
# drawplot(linearmodel)
residuals=residual(linearmodel)
print(residuals)
plt.scatter(years, residuals)
plt.show()

years=sm.add_constant(years)
results = sm.OLS(co2conc, years).fit()
print(results.summary())