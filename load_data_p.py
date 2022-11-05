#!/usr/bin/env python
# coding: utf-8

#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import curve_fit

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

yearspre = []
co2concpre = []
for item in rows:
    yearspre.append(float(item[0])-1959)
for item in rows:
    co2concpre.append(float(item[1]))
bar_y=sum(co2concpre)/len(co2concpre)
print(bar_y)
for i in range(len(co2concpre)):
    co2concpre[i]=co2concpre[i]-bar_y
    
years=[]
co2conc=[]
yearstest=[]
co2conctest=[]

i=0
help10 = int(len(yearspre)/10)
while i<len(yearspre):
    print(i)
    if i%help10==0:
        yearstest.append(yearspre[i])
        co2conctest.append(co2concpre[i])
    else:
        years.append(yearspre[i])
        co2conc.append(co2concpre[i])
    i+=1
print(years)
print(co2conc)
slope, intercept, r, p, std_err = stats.linregress(years, co2conc)


def helplinear(years):
  return slope * years + intercept
  
def exp(x,a,b,c):
    return a * np.exp(-b * x) + c

def log(x,a,b,c):
    return a+b*np.log(x) + c

#returns regression model of float array
def linear_regression():
    mymodel = list(map(helplinear, years))
    return mymodel
    
def helplog(model):
    out=[]
    for year in years:
        out.append(model[1]-model[0]*np.log(year))
    return out


#returns regression model of float array
def exp_regression():
    popt, pcov = curve_fit(exp, years, co2conc,p0=(1, 1e-6, 1))
    model= np.empty(len(years), dtype=object)
    for i in range(len(years)):
        model[i]=exp(years[i],*popt)
    return model

#returns regression model of float array
def log_regression():
    popt, pcov = curve_fit(log, years, co2conc,p0=(1, 1e-6, 1))
    model= np.empty(len(years), dtype=object)
    for i in range(len(years)):
        model[i]=log(years[i],*popt)
    return model

#returns regression model of float array
def poly_regression():
    mymodel = np.poly1d(np.polyfit(years, co2conc, 3))
    model=mymodel(years)
    return model

#draws plot from model in form of float array
def drawplot(model):
    plt.scatter(years, co2conc)
    plt.plot(years, model)
    plt.scatter(yearstest, co2conctest, c='r')
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

# plt.scatter(years, co2conc)
# plt.plot(years, model)
# plt.scatter(yearstest, co2conctest, c='r')
# plt.show()

polymodel=poly_regression()
linearmodel = linear_regression()
logmodel = log_regression()
print(polymodel)
# drawplot(polymodel)
# drawplot(linearmodel)
drawplot(logmodel)

# residuals=residual(polymodel)
# print(residuals)
# plt.scatter(years, residuals)
# plt.show()

years=sm.add_constant(years)
results = sm.OLS(co2conc, years).fit()
print(results.summary())