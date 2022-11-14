#!/usr/bin/env python
# coding: utf-8

# In[85]:


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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from pmdarima.arima.utils import ndiffs  
from scipy.optimize import curve_fit
import pandas as pd
#from varname import nameof
import pmdarima as pm
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
from collections import OrderedDict
import math


# In[35]:


csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"

# csv file name
filename1= "temp.csv"
# initializing the titles and rows list
fields1 = []
rows1 = []
# reading csv file

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
years=[]
temps = []

for item in rows1:
    years.append(float(item[0]))
    temps.append(float(item[1]))
#bar_y=sum(temps)/len(temps)
#print(bar_y)
#for i in range(len(temperaturepre)):
    #temps[i]=temps[i]-bar_y
years=years[1:]
temps=temps[1:]
print(len(years))
print(len(temps))


# In[36]:


msk=int(len(temps)*0.75)
years_train=years[:msk]
years_test=years[msk:]
temps_train=temps[:msk]
temps_test=temps[msk:]
print(temps_train)
print(years_train)


# In[54]:


def residualtest(model,temps):
    resid = []; residsum=0
    for i in range(len(temps)):
        r=abs(temps[i]-model[i])
        resid.append(r)
        residsum+=r*r
    return resid, residsum


# In[52]:


yearss=np.square(years)
dfyy=[]
for i in range(len(years_train)):
    dfyy.append([years_train[i],yearss[i]])
model=sm.OLS(temps_train, sm.add_constant(dfyy))
p2 = model.fit().params
print(p2)
print(model.fit().summary())


# In[59]:


def func2(x,y):
    return p2[2] * y + p2[1] * x + p2[0]
model2=list(map(func2,years,yearss))
print(model2)
model2_train=model2[:msk]
model2_test=model2[msk:]


# In[60]:


resid,residsum=residualtest(model2_test,temps_test)
print(residsum)


# In[63]:


resid,residsum=residualtest(model2,temps)


# In[64]:


resid_train=resid[0:msk]
resid_test=resid[msk:]
resid_train=np.array(resid_train)
resid_test=np.array(resid_test)
print(resid_train)
print(resid_test)


# In[65]:


model = pm.auto_arima(resid_train, 
                        m=12, seasonal=True,
                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)


# In[66]:


prediction=model.predict(len(resid_test))
print(type(prediction))


# In[69]:


test=[]
for i in range(len(model2[msk:])):
    test.append(model2[msk:][i]+prediction[i])
print(len(test))
print(test)


# In[70]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(years_train,temps_train,label="Training")
ax.scatter(years_test,temps_test,label="Testing")
ax.plot(years_test,test,color='black',label="Predicted")
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Land-Ocean Temperature (Degrees Celsius)")
ax.title.set_text('Prediction of Land-Ocean Temperature Using ARIMA Model')
ax.legend()
plt.show()
print(type(model.predict(len(temps_test))))


# In[71]:


resid,residsum=residualtest(test,temps_test)
print(residsum)
print(len(resid))


# In[61]:


############################# Linear


# In[37]:


model=sm.OLS(temps_train, sm.add_constant(years_train))
p = model.fit().params
print(p)
print(model.fit().summary())


# In[38]:


def helplinear(x):
    return p[1] * x + p[0]
def linear_regression(xlist):
    mymodel = list(map(helplinear, xlist))
    return mymodel


# In[39]:


mmodel=linear_regression(years)
print(len(mmodel))


# In[117]:


resid,residsum=residualtest(mmodel,temps)
print(residsum)
print(len(resid))


# In[19]:


######################################### ARIMA Model


# In[115]:


#resid=np.log(resid)
#print(len(resid))


# In[118]:


resid_train=resid[0:msk]
resid_test=resid[msk:]
resid_train=np.array(resid_train)
resid_test=np.array(resid_test)
print(resid_train)
print(resid_test)


# In[119]:


decompose_result_mult = seasonal_decompose(resid_train, model="additive",period=1)

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

decompose_result_mult.plot();


# In[120]:


model = pm.auto_arima(resid_train, 
                        m=12, seasonal=True,
                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)


# In[121]:


print(model.summary())


# In[125]:


prediction=model.predict(len(resid_test))
print(prediction)


# In[126]:


#tmpre=[]
#for i in range(len(prediction)):
#    tmpre.append(math.exp(prediction[i]))
#prediction=tmpre


# In[127]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(years_train,resid_train,label="Training")
ax.scatter(years_test,resid_test,label="Testing")
ax.plot(years_test,prediction,color='black',label="Predicted")
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Land-Ocean Temperature (Degrees Celsius)")
ax.title.set_text('Prediction of Land-Ocean Temperature Using ARIMA Model')
ax.legend()
plt.show()
print(type(model.predict(len(temps_test))))


# In[128]:


test=[]
for i in range(len(mmodel[msk:])):
    test.append(mmodel[msk:][i]+prediction[i])
print(len(test))
print(test)


# In[129]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(years_train,temps_train,label="Training")
ax.scatter(years_test,temps_test,label="Testing")
ax.plot(years_test,test,color='black',label="Predicted")
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Land-Ocean Temperature (Degrees Celsius)")
ax.title.set_text('Prediction of Land-Ocean Temperature Using ARIMA Model')
ax.legend()
plt.show()
print(type(model.predict(len(temps_test))))


# In[130]:


resid,residsum=residualtest(test,temps_test)
print(residsum)
print(len(resid))

