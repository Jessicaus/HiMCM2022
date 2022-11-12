#!/usr/bin/env python
# coding: utf-8

#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import scipy
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind
from scipy.optimize import curve_fit
import pandas as pd
from varname import nameof
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX


csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"

# csv file name
filename = "co2.csv "
filename1= "temp.csv"
# initializing the titles and rows list
fields = []
rows = []
fields1 = []
rows1 = []
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

with open(filename1, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields1 = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows1.append(row)
 
    # get total number of rows
    line_numbers = (csvreader.line_num)
yearspre = []
temperaturepre = []

for item in rows1:
    # yearspre.append(float(item[0])-1959)
    yearspre.append(float(item[1]))
for item in rows:
    temperaturepre.append(float(item[1]))
bar_y=sum(temperaturepre)/len(temperaturepre)
print(bar_y)
for i in range(len(temperaturepre)):
    temperaturepre[i]=temperaturepre[i]-bar_y
for i in range(len(yearspre)):
    yearspre[i]=yearspre[i]*100
yearspre=yearspre[1:]


yearsall=[]
for i in range(-100,200):
    yearsall.append(i)

i=0
help10 = int(len(yearspre)/10)

years=[]
temperature=[]
yearstest=[]
temperaturetest=[]
yearslast=yearspre[46:]
temperaturelast=temperaturepre[46:]
years=yearspre[0:46]
temperature=temperaturepre[0:46]
i=0
slope, intercept, r, p, std_err = stats.linregress(years, temperature)

temperaturealldict={}

i=0
for year in yearspre:
    temperaturealldict[year]=temperaturepre[i]
    i+=1
    
def helplinear(years):
    print("abcabcbcabcabcabca111111111111mmmmmmmmmm"+str(slope)+" "+str(years)+" "+str(intercept))
    return slope * years + intercept
  
def exp(x,a,b,c):
    print("abcabcbcabcabcabca444444444mmmmmm"+str(a)+" "+str(b)+" "+str(c))
    return a * np.exp(-b * x) + c

# def log(x,a,b,c):
    # print("abcabcbcabcabcabca33333333333333mmmmmmm"+str(a)+" "+str(b)+" "+str(c))
    return a+b*np.log(x) + c
def log(x, a, b, c, d):
    print("abcabcbcabcabcabca33333333333333mmmmmmm"+str(a)+" "+str(b)+" "+str(c)+""+str(d))
    return a * np.exp(-b * (x - c)) + d
#returns regression model of float array
def linear_regression():
    mymodel = list(map(helplinear, yearsall))
    return mymodel

#returns regression model of float array
def exp_regression():
    # popt, pcov = curve_fit(exp, years, temperature)
    popt, pcov = curve_fit(exp, years, temperature,p0=(1, 0.01, 1))

    model= np.empty(len(yearsall), dtype=object)
    for i in range(len(yearsall)):
        model[i]=exp(yearsall[i],*popt)
    return model

#returns regression model of float array
def log_regression():
    popt, pcov = curve_fit(log, years, temperature,p0=(50, 0, 90, 60), bounds=([0, 0, 90, 0], [1000, 0.1, 200, 200]))
    model= np.empty(len(yearsall), dtype=object)
    for i in range(len(yearsall)):
        model[i]=log(yearsall[i],*popt)
    return model

#returns regression model of float array
def poly_regression():
    mymodel = np.poly1d(np.polyfit(years, temperature, 3))
    print((mymodel))
    model=mymodel(yearsall)
    return model
   
def drawplot(model,x,y):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid()
    ax.scatter(years, temperature, label='Training Data')
    ax.plot(yearspre, model,c='k')
    ax.scatter(yearslast, temperaturelast, c='r',label='Testing Data')
    ax.set_xlabel("t")
    ax.set_ylabel(r'$c-\bar{c}$')
    ax.legend()
    ax.title.set_text('{} Regression'.format(names[x*2+y]))

def drawplotgrid(model,x,y):
    ax[x,y].grid()
    ax[x,y].scatter(years, temperature, label='Training Data')
    print(len(yearsall))
    print(len(model))
    ax[x,y].plot(yearsall, model,c='k')
    print(len(yearslast))
    print(len(temperaturelast))
    ax[x,y].scatter(yearslast, temperaturelast, c='r',label='Testing Data')
    ax[x,y].set_xlabel("t")
    ax[x,y].set_ylabel(r'$c-\bar{c}$')
    ax[x,y].legend()
    ax[x,y].title.set_text('{} Regression'.format(names[x*2+y]))
plt.scatter(yearspre,temperaturepre)
plt.show()
def residualassess(model):
    resid=0
    i=0
    f_model = {}
    for year in yearspre:
        f_model[year]=model[i]
        i+=1
    i=0
    for year in yearslast:
        residvalue = pow(temperaturealldict[year]-f_model[year],2)
        resid=resid+residvalue
        i+=1
    return resid

#returns residualtrains from model in form of float array
def residualtrain(model):
    resid = []
    i=0
    f_model = {}
    for year in yearspre:
        f_model[year]=model[i]
        i+=1
    i=0
    for year in years:
        residvalue = temperaturedict[year]-f_model[year]
        resid.append(residvalue)
        i+=1
    return resid
    
def residualtest(model):
    resid = []
    i=0
    f_model = {}
    for year in yearspre:
        f_model[year]=model[i]
        i+=1
    i=0
    for year in yearslast:
        residvalue = temperaturedict[year]-f_model[year]
        resid.append(residvalue)
        i+=1
    return resid

def drawresid(resids,name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid()
    ax.scatter(years, resids)
    fig.savefig("11residualstrainingdata"+str(name)+".png")

def drawresidtest(resids,name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid()
    ax.scatter(yearslast, resids)
    ax.set_xlabel("t")
    ax.set_ylabel("Deviations")
    ax.title.set_text('Residual Plot')
    fig.savefig("11residualstestdata"+str(name)+".png")

def t_test(data_group1, data_group2):
    a=np.array(data_group1)
    b=np.array(data_group2)
    out=str(ttest_ind(data_group1, data_group2))
    return (out.split(" ")[1])#changed

def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    residualstestaa=np.array(group1)
    residualstrainaa=np.array(group2)
    nun = residualstestaa.size-1
    dun = residualstrainaa.size-1
    p_value = 1-scipy.stats.f.cdf(f, nun, dun)
    return p_value#changed

def shapiro_onlyp(data):
    new=data.split(" ")[1]
    new1=new.split("=")[1]
    new2=new1.split(")")[0]
    return new2#changed

polymodel=poly_regression()
linearmodel = linear_regression()
logmodel = log_regression()
expmodel = exp_regression()
names=["Linear","Polynomial","Logarithmic","Exponential"]
models=[linearmodel,polymodel,logmodel,expmodel]

fig, ax = plt.subplots(2,2,figsize=(12, 7))
for i in range(2):
    for j in range(2):
        drawplotgrid(models[i*2+j],i,j)
plt.show()

for i in range(2):
    for j in range(2):
        drawplot(models[i*2+j],i,j)
#########################################################

models=[]
models.append(linearmodel)
models.append(polymodel)
models.append(logmodel)
models.append(expmodel)
    
i=0

outputdata=[]

for model in models:
    residualstrain=residualtrain(model)
    residualstest=residualtest(model)
    modeldata=[]
    modeldata.append(residualassess(model))
    modeldata.append(t_test(residualstest,residualstrain))
    modeldata.append(shapiro_onlyp(str((shapiro(residualstest)))))
    modeldata.append(shapiro_onlyp(str((shapiro(residualstrain)))))
    modeldata.append(f_test(residualstest,residualstrain))
    outputdata.append(modeldata)
    i+=1



new_array=np.array(outputdata)
file = open("p_values.txt", "w+")
Saving the array in a text file
content = str(new_array)
file.write(content)
file.close()

########################################################

years=sm.add_constant(years)
results = sm.OLS(temperature, years).fit()
print(results.summary())


# stepwise_fit = auto_arima(temperaturepre, start_p = 1, start_q = 1,
                          # max_p = 3, max_q = 3, m = 12,
                          # start_P = 0, seasonal = True,
                          # d = None, D = 1, trace = True,
                          # error_action ='ignore',   # we don't want to know if an order does not work
                          # suppress_warnings = True,  # we don't want convergence warnings
                          # stepwise = True) 
# stepwise_fit.summary()
# model = SARIMAX(temperature, 
                # order = (1, 1, 0), 
                # seasonal_order =(0, 1, 1, 12))
# result = model.fit()
# result.summary()
# start=0
# end=len(yearspre)-1
# predictions = result.predict(start, end,
                             # typ = 'levels').rename("Predictions")
# predictions.plot(legend = True)
# test['# Passengers'].plot(legend = True)


for model in models:
    print(residualassess(model))