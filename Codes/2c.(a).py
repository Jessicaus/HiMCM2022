#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from pmdarima.arima.utils import ndiffs  
# Import the library
import pmdarima as pm
from pmdarima import auto_arima
# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.svm import SVR
from collections import OrderedDict


# In[2]:


csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
# csv file name
filename = "co2.csv"
filename2 = "temp.csv"


# <h1 style="color:Green;">Getting Data</h1> 

# In[3]:


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


# In[4]:


years=[]
co2conc=[]
for item in rows:
    years.append(float(item[0])-1959)
for item in rows:
    co2conc.append(float(item[1]))
years=np.array(years)
co2conc=np.array(co2conc)
print(len(years))


# In[5]:


# initializing the titles and rows list
fields = []
rows = []
 
# reading csv file
with open(filename2, 'r') as csvfile:
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


# In[6]:


temps=[]
for item in rows:
    temps.append(float(item[1]))
temps=temps[1:]
temps=np.array(temps)


# In[7]:


#years_predict=[63,64,65,66,67] #start from 63


# <h1 style="color:Green;">1c Modeling!</h1> 

# In[8]:


def exp(x,a,b,c):
    return a * np.exp(-b * x) + c
def exp_regression(years_predict):
    #plt.scatter(years_train,co2conc_train)
    #plt.show()
    popt, pcov = curve_fit(exp, years, co2conc,p0=(20, 1e-6, -41))
    model= np.empty(len(years_predict), dtype=object)
    for i in range(len(years_predict)):
        model[i]=exp(years_predict[i],*popt)
    return model
def predict1a(years_predict):
    expmodel = exp_regression(years_predict)
    return expmodel


# <h1 style="color:Green;">2a Modeling!</h1> 

# In[9]:


def residualtest(model,emp):
    resid = []; residsum=0
    for i in range(len(emp)):
        r=abs(emp[i]-model[i])
        resid.append(r)
        residsum+=r*r
    return resid, residsum
model=sm.OLS(temps, sm.add_constant(years))
p = model.fit().params

def helplinear(x):
    return p[1] * x + p[0]
def linear_regression(xlist):
    mymodel = list(map(helplinear, xlist))
    return mymodel
mmodel=linear_regression(years)
resid,residsum=residualtest(mmodel,temps)
model = pm.auto_arima(resid, 
                        m=12, seasonal=True,
                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)


# In[10]:


def predict2a1(years_predict):
    linearmodel=linear_regression(years_predict)
    return linearmodel
def predict2a2(years_predict):
    prediction=model.predict(len(years_predict))
    return prediction
def predict2a(years_predict):
    test=[]
    lmodel=predict2a1(years_predict)
    rmodel=predict2a2(years_predict)
    for i in range(len(lmodel)):
        test.append(lmodel[i]+rmodel[i])
    return test


# In[22]:


yearstest=[0]
co2conctest=list(co2conc)
tempstest=list(temps)
corrcoef=[0.96131825]
for i in range(len(years),100+len(years),5):
    #print(i)
    yearstest.append(i-len(years)+5)
    years_predict=np.arange(i,i+5)
    co2conctest.extend(predict1a(years_predict))
    #print(co2conctest)
    tempstest.extend(predict2a(years_predict))
    #print(tempstest)
    corrcoef.append(np.corrcoef(np.array(co2conctest),np.array(tempstest))[0][1])
    #print(np.corrcoef(np.array(co2conctest),np.array(tempstest)))
print(corrcoef)
print(len(corrcoef))
print(len(yearstest))


# In[23]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.scatter(yearstest,corrcoef)
ax.plot(yearstest,corrcoef)
ax.set_xlabel("Years")
ax.set_ylabel("Correlation")
ax.title.set_text('Correlation v Years')
#ax.legend()
plt.show()

