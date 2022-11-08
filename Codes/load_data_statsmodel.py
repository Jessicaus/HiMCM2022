#!/usr/bin/env python
# coding: utf-8

# In[38]:


#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# In[39]:


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


# In[40]:


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
    print(co2conc[i])



# for item in years:
    # print(item)
    # input()

slope, intercept, r, p, std_err = stats.linregress(years, co2conc)
print(slope,intercept)


# In[41]:


def helplinear(years):
  return slope * years + intercept

def linear_regression():
    mymodel = list(map(helplinear, years))
    print(mymodel)
    plt.scatter(years, co2conc)
    plt.plot(years, mymodel)
    plt.show()
linear_regression()



def readFile(file, readto, readto1):
    fin = open(file,"r", encoding="utf-8")
    for line in fin:
        line = line.strip().split(",")
        readto.append(line[0])
        readto1.append(line[1])
    fin.close()
# temperature=np.genfromtxt("temp.csv", skip_header=1, replace_space="")
# co2=np.genfromtext('co2', skip_header=1)
co2x = []
co2y = []
#readFile("e:\HiMCM\actual\co2_nohead.csv", co2x, co2y)


# In[42]:





# In[6]:


plt.scatter(co2x,co2y)

