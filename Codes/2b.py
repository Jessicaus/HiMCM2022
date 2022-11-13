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


# In[2]:


csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
# csv file name
filename = "co2.csv"
filename2 = "temp.csv"


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


fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.scatter(temps,co2conc)
ax.set_xlabel("Land-Ocean Temperature ("+r'$^{\circ}$'+" C)")
ax.set_ylabel("CO2 Concentration (PPM)")
ax.title.set_text('CO2 Concentration v. Land-Ocean Temperature')
#ax.legend()
plt.show()


# In[8]:


print(np.corrcoef(co2conc,temps))


# In[9]:


msk=int(len(temps)*0.75)
years_train=years[:msk]
years_test=years[msk:]
temps_train=temps[:msk]
temps_test=temps[msk:]
co2conc_train=co2conc[:msk]
co2conc_test=co2conc[msk:]
print(temps_train)
print(years_train)


# In[10]:


model=sm.OLS(co2conc_train, sm.add_constant(temps_train))
p1 = model.fit().params
print(p1)
print(model.fit().summary())


# In[11]:


def helplinear(x):
    return p1[1] * x + p1[0]
def linear_regression(xlist):
    mymodel = list(map(helplinear, xlist))
    return mymodel


# In[12]:


mmodel=linear_regression(temps)


# In[13]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.scatter(temps_train, co2conc_train, label='Training Data')
ax.plot(temps,mmodel,c='k')
ax.scatter(temps_test,co2conc_test, c='r',label='Testing Data')
ax.set_xlabel("Land-Ocean Temperature ("+r'$^{\circ}$'+" C)")
ax.set_ylabel("CO2 Concentration (PPM)")
#ax[x,y].title.set_text(""{list}"'Regression')
ax.legend()
#if ppp==0:
#fig.legend()
ax.title.set_text('Correlation between CO2 Concentration and Land-Ocean Temperature')


# In[14]:


def residualtest(model,co2conc):
    resid = []; residsum=0
    for i in range(len(co2conc)):
        r=abs(co2conc[i]-model[i])
        resid.append(r)
        residsum+=r*r
    return resid, residsum


# In[15]:


resid,residsum=residualtest(mmodel[msk:],co2conc_test)
print(residsum)


# In[16]:


resid,residsum=residualtest(mmodel[:msk],co2conc_train)
print(resid)
print(len(resid))
print(len(co2conc_train))
print(msk)
print(len(mmodel))


# In[17]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.scatter(temps_train, resid, label='Training Data')
#ax.plot(temps,mmodel,c='k')
#ax.scatter(temps_test,co2conc_test, c='r',label='Testing Data')
ax.set_xlabel("Land-Ocean Temperature ("+r'$^{\circ}$'+" C)")
ax.set_ylabel("Residual")
#ax[x,y].title.set_text(""{list}"'Regression')
ax.legend()
#if ppp==0:
#fig.legend()
ax.title.set_text('Residual between CO2 Concentration and Land-Ocean Temperature')


# In[18]:


dfy=[]
for i in range(len(years_train)):
    dfy.append([years_train[i],temps_train[i]])
model=sm.OLS(co2conc_train, sm.add_constant(dfy))
p2 = model.fit().params
print(p2)
print(model.fit().summary())


# In[19]:


yearss=np.square(years)
dfyy=[]
for i in range(len(years_train)):
    dfyy.append([years_train[i],yearss[i],temps_train[i]])
model=sm.OLS(co2conc_train, sm.add_constant(dfyy))
p3 = model.fit().params
print(p3)
print(model.fit().summary())


# In[20]:


# graph all of these
testc=np.arange(1,len(years))
testc_train=years[:msk]
testc_test=years[msk:]


# In[21]:


def func1(x):
    return p1[1] * x + p1[0]
def func2(x,y):
    return p2[2] * y + p2[1] * x + p2[0]
def func3(x,y,z):
    return p3[3] * z + p3[2] * y + p3[1] * x + p3[0]


# In[22]:


model1=list(map(func1,temps))
model2=list(map(func2,years,temps))
model3=list(map(func3,years,yearss,temps))
print(model1)


# In[23]:


model1_train=model1[:msk]
model1_test=model1[msk:]
model2_train=model2[:msk]
model2_test=model2[msk:]
model3_train=model3[:msk]
model3_test=model3[msk:]


# In[33]:


def drawplotgrid(model_train,model_test,x):
    ax[x].grid()
    ax[x].scatter(testc_train, co2conc_train, label='Training Data')
    ax[x].plot(testc_train, model_train,c='k')
    ax[x].plot(testc_test, model_test,color='gray')
    ax[x].scatter(testc_test, co2conc_test, c='r',label='Testing Data')
    ax[x].set_xlabel("Test Case")
    ax[x].set_ylabel("Predicted CO2 Concentration")
    #ax[x,y].title.set_text(""{list}"'Regression')
    ax[x].legend()
    #if ppp==0:
     #   fig.legend()
    ax[x].title.set_text('{} Results'.format(names[x]))
    #fig.savefig("11regression"+name+".png")


# In[42]:


def drawplot(model_train,model_test,x):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid()
    ax.scatter(testc_train, co2conc_train, label='Training Data')
    ax.plot(testc_train, model_train,c='k')
    ax.plot(testc_test, model_test,color='gray')
    ax.scatter(testc_test, co2conc_test, c='r',label='Testing Data')
    ax.set_xlabel("Test Case")
    ax.set_ylabel("Predicted CO2 Concentration")
    #ax[x,y].title.set_text(""{list}"'Regression')
    ax.legend()
    #if ppp==0:
     #   fig.legend()
    ax.title.set_text('{} Results'.format(names[x]))
    #fig.savefig("11regression"+name+".png")


# In[43]:


names=["Model 1", "Model 2", "Model 3"]
models=[[model1_train,model1_test],[model2_train,model2_test],[model3_train,model3_test]]

fig, ax = plt.subplots(3,1,figsize=(9,18))
for i in range(3):
    drawplotgrid(models[i][0],models[i][1],i)
for i in range(3):
    drawplot(models[i][0],models[i][1],i)


# In[49]:


def residualtest(model):
    resid = []; residsum=0
    for i in range(len(co2conc_test)):
        r=abs(co2conc_test[i]-model[i])
        resid.append(r)
        residsum+=r*r
    return resid, residsum


# In[56]:


resid1,residsum1=residualtest(model1_test)
print(residsum1)


# In[57]:


resid2,residsum2=residualtest(model2_test)
print(residsum2)


# In[58]:


resid2,residsum2=residualtest(model3_test)
print(residsum2)

