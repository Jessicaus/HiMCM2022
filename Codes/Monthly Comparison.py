#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import curve_fit


# In[2]:


# csv file name
filename = "co2.csv"
 
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
    years.append(float(item[0]))
for item in rows:
    co2conc.append(float(item[1]))
sc=years.index(2004)
years=np.array(years)
co2conc=np.array(co2conc)


# In[5]:


left=[]
right=[]
for i in range(0,sc-10):
    left.append(co2conc[i+9]-co2conc[i])
for i in range(sc-10,len(co2conc)-9):
    right.append(co2conc[i+9]-co2conc[i])
print(left)
print(right)


# In[10]:

csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.bar(years[:len(left)],left,label='Before 2004')
ax.bar(years[len(left):len(left)+len(right)],right,label='Including or After 2004')
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Average Changes in CO2 Concentration over 10 years")
ax.title.set_text('Average Changes in CO2 Concentration of 2004 compared to Others')
ax.legend()
plt.show()

# added 2022/11/08
p_sum=np.array([None]*(sc+2))
p_sum[0]=0
p_sum[1]=co2conc[0]
for i in range(0,sc):
    p_sum[i+2]=p_sum[i+1]+co2conc[i+1]
print(len(co2conc))
print(len(p_sum))

sm=0
avg=np.array([None]*(len(p_sum)-10))
for i in range(10,len(p_sum)):
    avg[i-10]=p_sum[i]-p_sum[i-10]
    avg[i-10]/=10
    sm+=avg[i-10]
sm/=len(avg)
print(avg)
print(len(avg))
print(sm)

fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.scatter(years[6:len(avg)+6],avg)
#ax.bar(years[:len(left)],left,label='Before 2004')
#ax.bar(years[len(left):len(left)+len(right)],right,label='Including or After 2004')
#ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Average Changes in CO2 Concentration over 10 years")
ax.title.set_text('Average Changes in CO2 Concentration of 2004 compared to Others')
# specifying horizontal line type
ax.axhline(avg[len(avg)-1], color = 'r', linestyle = '-',label='Maximum')
ax.axhline(avg[0], color = 'g', linestyle = '-',label='Minimum')
ax.axhline(sm, color = 'orange', linestyle='dashed',label='Mean')
ax.plot(2004, co2conc[sc], marker="o", markersize=6, markerfacecolor="black",label="2004 Value")
ax.legend()
plt.show()
