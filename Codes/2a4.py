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
from sklearn.preprocessing import StandardScaler


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


#sc_X = StandardScaler()
#sc_y = StandardScaler()
#yearssc = sc_X.fit_transform(years[:, np.newaxis])
#tempssc = sc_y.fit_transform(temps[:, np.newaxis])
#yearssc=yearssc.flatten()
#tempssc=tempssc.flatten()
#print(yearssc)
#print(tempssc)


# In[8]:


yearsy=[]
for i in range(len(years)):
    yearsy.append([years[i]])


# In[9]:


msk=int(len(temps)*0.75)
years_train=yearsy[:msk]
years_test=yearsy[msk:]
temps_train=temps[:msk]
temps_test=temps[msk:]
#co2conc_train=co2conc[:msk]
#co2conc_test=co2conc[msk:]
print(temps_train)
print(years_train)


# In[10]:


from sklearn.svm import SVR
regressor = SVR(kernel="poly").fit(years_train,temps_train)
prediction=regressor.predict(yearsy)

dct=dict(zip(years,prediction))
dct=OrderedDict(sorted(dct.items()))


# In[11]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(years_train,temps_train)
ax.plot(dct.keys(),dct.values(),color="green")
ax.scatter(years_test,temps_test,label="Testing")
#ax.plot(years_test,prediction,color='black',label="Predicted")
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Land-Ocean Temperature (Degrees Celsius)")
ax.title.set_text('Scatterplot of Land-Ocean Temperature')
#ax.legend()
plt.show()
#print(type(model.predict(len(temps_test))))


# In[12]:


regressor2 =SVR(kernel="rbf").fit(years_train,temps_train)
prediction2=regressor2.predict(yearsy)

dct2=dict(zip(years,prediction2))
dct2=OrderedDict(sorted(dct2.items()))


# In[85]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(years_train,temps_train)
ax.plot(dct2.keys(),dct2.values(),color="green")
ax.scatter(years_test,temps_test,label="Testing")
#ax.plot(years_test,prediction,color='black',label="Predicted")
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Land-Ocean Temperature (Degrees Celsius)")
ax.title.set_text('Scatterplot of Land-Ocean Temperature')
#ax.legend()
plt.show()
#print(type(model.predict(len(temps_test))))


# In[13]:


def residualtest(model,temperature):
    resid = []; residsum=0
    for i in range(len(temperature)):
        r=abs(temperature[i]-model[i])
        resid.append(r)
        residsum+=r*r
    return resid, residsum


# In[14]:


#SSM for SVR Poly Model
resid,residsum=residualtest(prediction[46:],temps_test)
print(residsum)


# In[86]:

#SSM for SVR RBF Model
resid,residsum=residualtest(prediction2[46:],temps_test)
print(residsum)


# In[15]:


model=sm.OLS(temps_train, sm.add_constant(years_train))
p= model.fit().params
print(p)
print(model.fit().summary())


# In[16]:


def func(x):
    return p[1] * x + p[0]
model=list(map(func,years))
model_train=model[:msk]
model_test=model[msk:]


# In[ ]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.scatter(years_train,temps_train, label='Training Data')
ax.plot(years_train, model_train,c='k')
ax.plot(years_test, model_test,color='gray')
ax.scatter(years_test,temps_test, c='r',label='Testing Data')
ax.set_xlabel("Test Case")
ax.set_ylabel("Predicted CO2 Concentration")
ax.legend()
ax.title.set_text("Linear Results")
plt.plot()


# In[ ]:


resid,residsum=residualtest(model_test,temps_test)
print(residsum)


# In[ ]:




