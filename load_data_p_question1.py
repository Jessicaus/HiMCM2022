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
yearspre = []
co2concpre = []
for item in rows:
    yearspre.append(float(item[0])-1959)
for item in rows:
    co2concpre.append(float(item[1]))
bar_y=sum(co2concpre)/len(co2concpre)
print(bar_y)
for i in range(len(co2concpre)):
    co2concpre[i]=co2concpre[i]-bar_y
    
years=[]
co2conc=[]
yearstest=[]
co2conctest=[]
yearslast=yearspre[46:]
co2conclast=co2concpre[46:]
years=yearspre[0:46]
co2conc=co2concpre[0:46]
i=0
help10 = int(len(yearspre)/10)
# while i<len(yearspre):
    # if i%help10==0:
        # yearstest.append(yearspre[i])
        # co2conctest.append(co2concpre[i])
    # else:
        # years.append(yearspre[i])
        # co2conc.append(co2concpre[i])
    # i+=1
slope, intercept, r, p, std_err = stats.linregress(years, co2conc)
co2concdict={}
co2conctestdict={}
co2concalldict={}
i=0
for year in years:
    co2concdict[year]=co2conc[i]
    i+=1
i=0
for year in yearstest:
    co2conctestdict[year]=co2conctest[i]
    i+=1
i=0
for year in yearspre:
    co2concalldict[year]=co2concpre[i]
    i+=1
    
def helplinear(years):
    # print("abcabcbcabcabcabca111111111111mmmmmmmmmm"+str(slope)+" "+str(years)+" "+str(intercept))
    return slope * years + intercept
  
def exp(x,a,b,c):
    # print("abcabcbcabcabcabca444444444mmmmmm"+str(a)+" "+str(b)+" "+str(c))
    return a * np.exp(-b * x) + c
def log(x,a,b,c):
    # print("abcabcbcabcabcabca33333333333333mmmmmmm"+str(a)+" "+str(b)+" "+str(c))
    return a-b*np.log(x) + c
#returns regression model of float array
def linear_regression():
    mymodel = list(map(helplinear, yearspre))
    return mymodel
    
def helplog(model):
    out=[]
    for year in years:
        out.append(model[1]-model[0]*np.log(year))
    return out
#returns regression model of float array
def exp_regression():
    print(str(years)+"  "+str(co2conc))
    plt.scatter(years,co2conc)
    plt.show()
    popt, pcov = curve_fit(exp, years, co2conc,p0=(20, 1e-6, -41))
    model= np.empty(len(yearspre), dtype=object)
    for i in range(len(yearspre)):
        model[i]=exp(yearspre[i],*popt)
    return model
#returns regression model of float array
def log_regression():
    popt, pcov = curve_fit(log, years[1:], co2conc[1:],p0=(20, 1e-6, -41), maxfev=2500)
    model= np.empty(len(yearspre), dtype=object)
    for i in range(len(yearspre)):
        model[i]=log(yearspre[i],*popt)
    return model
#returns regression model of float array
def poly_regression():
    mymodel = np.poly1d(np.polyfit(years, co2conc, 3))
    # print((mymodel))
    model=mymodel(yearspre)
    return model
#draws plot from model in form of float array
# def drawplot(model,name):
    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax.grid()
    # ax.scatter(years, co2conc)
    # ax.plot(yearspre, model,c='k')
    # ax.scatter(yearstest, co2conctest, c='r')
    # ax.set_xlabel("t")
    # ax.set_ylabel(r'$y-\bar{y}$')
    # ax.title.set_text('Regression')
    # fig.savefig("11regression"+name+".png")
    # plt.show()
    
#def drawplot(model,x,y):
    # fig, ax = plt.subplots(figsize=(7, 5))
    #ax[x,y].grid()
    #ax[x,y].scatter(years, co2conc, label='Training Data')
    #ax[x,y].plot(yearspre, model,c='k')
    #ax[x,y].scatter(yearslast, co2conclast, c='r',label='Testing Data')
    #ax[x,y].set_xlabel("t")
    #ax[x,y].set_ylabel(r'$y-\bar{y}$')
    #ax[x,y].title.set_text(""{list}"'Regression')
    #if ppp==0:
    #    fig.legend()
    #ax[x,y].title.set_text('{} Regression'.format(names[x*2+y]))
    #fig.savefig("11regression"+name+".png")
   
def drawplot(model,x,y):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid()
    ax.scatter(years, co2conc, label='Training Data')
    ax.plot(yearspre, model,c='k')
    ax.scatter(yearslast, co2conclast, c='r',label='Testing Data')
    ax.set_xlabel("t")
    ax.set_ylabel(r'$c-\bar{c}$')
    #ax[x,y].title.set_text(""{list}"'Regression')
    ax.legend()
    #if ppp==0:
     #   fig.legend()
    ax.title.set_text('{} Regression'.format(names[x*2+y]))
    #fig.savefig("11regression"+name+".png")
def drawplotgrid(model,x,y):
    # fig, ax = plt.subplots(figsize=(7, 5))
    ax[x,y].grid()
    ax[x,y].scatter(years, co2conc, label='Training Data')
    ax[x,y].plot(yearspre, model,c='k')
    ax[x,y].scatter(yearslast, co2conclast, c='r',label='Testing Data')
    ax[x,y].set_xlabel("t")
    ax[x,y].set_ylabel(r'$c-\bar{c}$')
    #ax[x,y].title.set_text(""{list}"'Regression')
    ax[x,y].legend()
    #if ppp==0:
     #   fig.legend()
    ax[x,y].title.set_text('{} Regression'.format(names[x*2+y]))
    #fig.savefig("11regression"+name+".png")
 
def residualassess(model):
    resid=0
    i=0
    f_model = {}
    for year in yearspre:
        f_model[year]=model[i]
        i+=1
    i=0
    for year in yearslast:
        residvalue = pow(co2concalldict[year]-f_model[year],2)
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
        residvalue = co2concdict[year]-f_model[year]
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
        residvalue = co2concdict[year]-f_model[year]
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
    ax.scatter(yearstest, resids)
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
# plt.scatter(years, co2conc)
# plt.plot(years, model)
# plt.scatter(yearstest, co2conctest, c='r')
# plt.show()
polymodel=poly_regression()
linearmodel = linear_regression()
logmodel = log_regression()
expmodel = exp_regression()
print(polymodel)
# drawplot(linearmodel,nameof(linearmodel))
# drawplot(polymodel,nameof(polymodel))
# drawplot(logmodel,nameof(logmodel))
# drawplot(expmodel,nameof(expmodel))
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
# models=[]
# models.append(linearmodel)
# models.append(polymodel)
# models.append(logmodel)
# models.append(expmodel)
# for model in models:
    
# i=0
# outputdata=[]
# for model in models:
    # residualstrain=residualtrain(model)
    # residualstest=residualtest(model)
    # drawresid(residualstrain,i)
    # drawresidtest(residualstest,i)
    # modeldata=[]
    # modeldata.append(t_test(residualstest,residualstrain))
    # modeldata.append(shapiro_onlyp(str((shapiro(residualstest)))))
    # modeldata.append(shapiro_onlyp(str((shapiro(residualstrain)))))
    # modeldata.append(f_test(residualstest,residualstrain))
    # outputdata.append(modeldata)
    # i+=1
# dataframe = pd.DataFrame(outputdata)
# dataframe.to_csv(r"e:\HiMCM\actual\p_values.csv")
# outputdata.tofile('p_value.csv', sep = ',')
# npoutputdata=np.array(outputdata)
# np.savetxt("p_value.txt", npoutputdata, delimiter = ",")
# new_array=np.array(outputdata)
# file = open("p_values.txt", "w+")
# Saving the array in a text file
# content = str(new_array)
# file.write(content)
# file.close()
########################################################
#years=sm.add_constant(years)
#results = sm.OLS(co2conc, years).fit()
#print(results.summary())
# stepwise_fit = auto_arima(co2concpre, start_p = 1, start_q = 1,
                          # max_p = 3, max_q = 3, m = 12,
                          # start_P = 0, seasonal = True,
                          # d = None, D = 1, trace = True,
                          # error_action ='ignore',   # we don't want to know if an order does not work
                          # suppress_warnings = True,  # we don't want convergence warnings
                          # stepwise = True) 
# stepwise_fit.summary()
# model = SARIMAX(co2conc, 
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
                           
yearsori = []
co2concori = []
for item in rows:
    yearsori.append(float(item[0]))
for item in rows:
    co2concori.append(float(item[1]))
yearsori=yearsori[0:46]
co2concori=co2concori[0:46]

#p-value evaluation  
years1=sm.add_constant(years)
results = sm.OLS(co2conc, years1).fit()
print(results.summary())
print(results.pvalues)

yearss=np.square(years)
yearsc=np.power(years,3)
dfy=[]
for i in range(len(years)):
    dfy.append([years[i],yearss[i],yearsc[i]])
dfy=np.array(dfy)
print(len(dfy))
print(type(dfy))
dfy1=sm.add_constant(dfy)
results=sm.OLS(co2conc,dfy1).fit()
print(results.summary())
print(results.pvalues)

years2=sm.add_constant(yearsori)
results = sm.OLS(np.log(co2concori), years2).fit()
print(results.summary())
print(results.pvalues)

years3=np.log(yearsori)
years3=sm.add_constant(years3)
results = sm.OLS(co2concori, years3).fit()
print(results.summary())
print(results.pvalues)
