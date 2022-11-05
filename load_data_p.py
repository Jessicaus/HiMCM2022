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
 def exp(x,a,b,c):
    return a * np.exp(-b * x) + c

#returns regression model of float array
def linear_regression():
    mymodel = list(map(helplinear, years))
    return mymodel
    

#returns regression model of float array
def poly_regression():
    mymodel = np.poly1d(np.polyfit(years, co2conc, 3))

    myline = np.linspace(1, 100, 100)
    print(mymodel)
    model=mymodel(years)
    return model
   
#returns regression model of float array
def exp_regression():
    popt, pcov = curve_fit(exp, years, co2conc,p0=(1, 1e-6, 1))
    model= np.empty(len(years), dtype=object)
    for i in range(len(years)):
        model[i]=exp(years[i],*popt)
    return model

#draws plot from model in form of float array
def drawplot(model):
    plt.scatter(years, co2conc)
    plt.plot(years, model)
    plt.show()

#returns residuals from model in form of float array
def residual(f_model):
    resid = []
    i=0
    for point in f_model:
        residvalue = co2conc[i]-point
        resid.append(residvalue)
        i+=1
    return resid

def res_draw(model):
    residuals=residual(model)
    plt.scatter(years, residuals)
    plt.show()

linearmodel = linear_regression()
polymodel=poly_regression()
expmodel=exp_regression()
print(polymodel)
drawplot(linearmodel,"r-")
drawplot(polymodel,"r-")
drawplot(expmodel,"r-")

res_draw(linearmodel)
res_draw(polymodel)
res_draw(expmodel)

years=sm.add_constant(years)
results = sm.OLS(co2conc, years).fit()
print(results.summary())
