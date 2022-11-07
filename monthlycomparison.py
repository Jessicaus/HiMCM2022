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


# In[4]:


years = []
co2conc = []
for item in rows:
    years.append(float(item[0]))
years.pop(0)
for item in rows:
    co2conc.append(float(item[1]))
years=np.array(years)
co2conc=np.array(co2conc)


# In[5]:


co2conc=np.diff(co2conc)
print(co2conc)


# In[10]:


result = np.where(years == 2004)
print(result)


fig, ax = plt.subplots(figsize=(7, 5))
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Change in CO2 Concentration (PPM)")
ax.bar(years, co2conc)
ax.bar(2004,co2conc[result],label="Average Change annually in 2004",color="red")
ax.title.set_text('Average Change of CO2 Concentration in 2004 compared to Other Years')
ax.legend()
plt.show()