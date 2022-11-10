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
filename = "temp.csv"
 
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


# In[3]:


years=[]
temps=[]
for item in rows:
    years.append(float(item[0]))
for item in rows:
    temps.append(float(item[1]))
years=np.array(years)
temps=np.array(temps)


# In[4]:


msk=int(len(temps)*0.75)
years_train=years[:msk]
years_test=years[msk:]
temps_train=temps[:msk]
temps_test=temps[msk:]


# In[5]:


diff1=np.diff(temps_train)
diff2=np.diff(diff1)


# In[6]:


# The Genuine Series  
fig, axes = plt.subplots(3, 2, sharex = True,figsize=(15,10))  
axes[0, 0].plot(temps_train); axes[0, 0].set_title('The Genuine Series')  
plot_acf(temps_train, ax = axes[0, 1])  
  
# Order of Differencing: First  
axes[1, 0].plot(diff1); axes[1, 0].set_title('Order of Differencing: First')  
plot_acf(diff1, ax = axes[1, 1])  
  
# Order of Differencing: Second  
axes[2, 0].plot(diff2); axes[2, 0].set_title('Order of Differencing: Second')  
plot_acf(diff2, ax = axes[2, 1])  
  
plt.show()  


# In[7]:


adf_test = adfuller(temps_train)
print(f'p-value: {adf_test[1]}')


# In[8]:


adf_test = adfuller(diff1)
print(f'p-value: {adf_test[1]}')


# In[9]:


fig, axes = plt.subplots(1, 2, sharex = True,figsize=(15,5))  
axes[0].plot(diff1); axes[0].set_title('Order of Differencing: First')  
axes[1].set(ylim = (0,5))  
plot_pacf(diff1, ax = axes[1],method='ywm')  
  
plt.show()  


# In[10]:


fig, axes = plt.subplots(1, 2, sharex = True,figsize=(15,5))  
axes[0].plot(diff1); axes[0].set_title('Order of Differencing: First')  
axes[1].set(ylim = (0, 1.2))  
plot_acf(diff1, ax = axes[1])  
  
plt.show()  


# In[11]:


#decompose_result_mult = seasonal_decompose(temps_train, model="additive",period=20)

#trend = decompose_result_mult.trend
#seasonal = decompose_result_mult.seasonal
#residual = decompose_result_mult.resid

#decompose_result_mult.plot();


# In[12]:


model = pm.auto_arima(temps_train, 
                        m=12, seasonal=True,
                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)


# In[18]:


print(model.summary())


# In[13]:


prediction=model.predict(len(temps_test))


# In[21]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(years_train,temps_train,label="Training")
ax.scatter(years_test,temps_test,label="Testing")
ax.plot(years_test,prediction,color='black',label="Predicted")
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Land-Ocean Temperature (Degrees Celsius)")
ax.title.set_text('Prediction of Land-Ocean Temperature Using ARIMA Model')
ax.legend()
plt.show()
print(type(model.predict(len(temps_test))))


# In[15]:


def residualtest(model):
    resid = []; residsum=0
    for i in range(len(years_test)):
        r=abs(temps_test[i]-model[i])
        resid.append(r)
        residsum+=r*r
    return resid, residsum


# In[16]:


resid,residsum=residualtest(prediction)
print(residsum)

