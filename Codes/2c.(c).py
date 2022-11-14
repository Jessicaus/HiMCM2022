#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[16]:


csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
# csv file name
filename = "co2.csv"
filename2 = "temp.csv"


# <h1 style="color:Green;">Getting Data</h1> 

# In[17]:


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


# In[18]:


years=[]
co2conc=[]
for item in rows:
    years.append(float(item[0])-1959)
for item in rows:
    co2conc.append(float(item[1]))
years=np.array(years)
co2conc=np.array(co2conc)


# In[19]:


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


# In[20]:


temps=[]
for item in rows:
    temps.append(float(item[1]))
temps=temps[1:]
temps=np.array(temps)


# <h1 style="color:Green;">1c Modeling!</h1> 

# In[21]:


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

# In[22]:


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
arimamodel = pm.auto_arima(resid, 
                        m=12, seasonal=True,
                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)


# In[23]:


def predict2a1(years_predict):
    linearmodel=linear_regression(years_predict)
    return linearmodel
def predict2a2(years_predict):
    prediction=arimamodel.predict(len(years_predict))
    return prediction
def predict2a(years_predict):
    test=[]
    lmodel=predict2a1(years_predict)
    rmodel=predict2a2(years_predict)
    for i in range(len(lmodel)):
        test.append(lmodel[i]+rmodel[i])
    return test


# <h1 style="color:Green;">2b Modeling!</h1> 

# In[30]:


yearss=np.square(years)
dfyy=[]
for i in range(len(years)):
    dfyy.append([years[i],yearss[i],temps[i]])
xmodel=sm.OLS(co2conc, sm.add_constant(dfyy))
p2 = xmodel.fit().params
def func(x,y,z):
    return p2[3] * z + p2[2] * y + p2[1] * x + p2[0]
def predict2b(years,temps):
    xmodel=list(map(func,years,np.square(years),temps))
    return xmodel


# In[ ]:





# In[34]:


years_predict=[63,64,65,66,67] #start from 63
co2concpredict2b=predict2b(years_predict,predict2a(years_predict))
print(predict2b(years_predict,predict2a(years_predict)))


# In[35]:


co2conc1a=predict1a(years_predict)
print(co2conc1a)


# In[36]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.scatter(years, co2conc, label='Training Data')
ax.scatter(years_predict,co2conc1a,label='Testing Data')
ax.plot(years_predict, co2concpredict2b,c='k')
#ax.plot(testc_test, model_test,color='gray')
#ax.scatter(testc_test, co2conc_test, c='r',label='Testing Data')
ax.set_xlabel("Test Case")
ax.set_ylabel("Predicted CO2 Concentration")
#ax[x,y].title.set_text(""{list}"'Regression')
ax.legend()
#if ppp==0:
#   fig.legend()
ax.title.set_text('predict graph')
#plt.savefig('{}.png'.format(names[x]), bbox_inches='tight',dpi=1200)
#fig.savefig("11regression"+name+".png")
plt.plot()

