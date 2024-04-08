# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:56:36 2020

@author: Ammara
"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\iowa\US-Br3")


##########################################################################################

os.chdir(r"C:\ammara_MD\flux_twer_data\potato")
#df=pd.read_csv("wisp_corr.csv")
df=pd.read_csv("data.csv")
os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")

## drop first row of pandas data frame 

mask = (df['TIMESTAMP'] >'2019-04-30 23:45:00') & (df['TIMESTAMP'] <= '2020-10-31')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
s=df.groupby(['month','year']).sum()
df=s
df["prec+irri"]=df["prec_inch"]+df["irrig_inches"]
df=df.reset_index()

df = df.assign(TIMESTAMP=pd.to_datetime(df[['year', 'month']].assign(day=1)))
import numpy as np
import matplotlib.pyplot as plt

df=df.sort_values(by='TIMESTAMP')
plt.rcParams['figure.figsize'] = (12, 7)
fig, ax = plt.subplots()
ax.set_ylabel('Monthly water (inches)', color='black')
N = len(df["TIMESTAMP"])
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars
yvals = df["Pine_ET_inches"]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["Potato_ET_inches"]
rects2 = ax.bar(ind+width, zvals, width, color='orange')

kvals = df["prec+irri"]
rects3 = ax.bar(ind+width+0.27, kvals, width, color='blue',fill=False,edgecolor='blue',linewidth=3.5)

ax.set_xticks(ind+width)
ax.set_xticklabels( ('May2019','Jun2019','July2019','Aug2019','Sep2019','Oct2018','Nov2018','Dec2018','Jan2019','Feb2019','Mar2019','Apr2019','May2019','Jun2019','July2019','Aug2019','Sep2019','Oct2019','Nov2019','Dec2019','Jan2020','Feb2020','Mar2020','Apr2020'
) )
ax.legend( (rects1[0], rects2[0],rects3[0]), ('Pines','Potatoes','Precip+Irrigation') ,loc='upper left', fontsize = 'large')

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#################################################################################################
df["prec+irri"]=df["irrig_inches"]+df["prec_inch"]
mask = (df['TIMESTAMP'] >'2018-07-01 23:45:00') & (df['TIMESTAMP'] <= '2020-04-15')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))

s=df.groupby(['month','year']).sum()

df=s
df=df.reset_index()
df = df.assign(TIMESTAMP=pd.to_datetime(df[['year', 'month']].assign(day=1)))
import numpy as np
import matplotlib.pyplot as plt

df=df.sort_values(by='TIMESTAMP')
plt.rcParams['figure.figsize'] = (12, 7)
fig, ax = plt.subplots()
ax.set_ylabel('Monthly ET (inches)', color='black')
N = len(df["TIMESTAMP"])
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars
yvals = df["Pine_ET_inches"]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["Potato_ET_inches"]
rects2 = ax.bar(ind+width, zvals, width, color='orange')

#kvals = df["prec+irri"]
#rects3 = ax.bar(ind+width+0.27, kvals, width, color='blue',fill=False,edgecolor='blue',linewidth=3.5)

ax.set_xticks(ind+width)
ax.set_xticklabels( ('May2019','Jun2019','July2019','Aug2019','Sep2019'
) )
#ax.legend( (rects1[0], rects2[0],rects3[0]), ('Pines','Potatoes','Precip+Irrigation') ,loc='upper left', fontsize = 'large')
ax.legend( (rects1[0], rects2[0],rects3[0]), ('Pines','Potatoes') ,loc='upper left', fontsize = 'large')
ax.set_ylim(0,8)

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
########################################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
#df=pd.read_csv("wisp_corr.csv")
df=pd.read_csv("data.csv")

df["prec+irri"]=df["irrig_inches"]+df["prec_inch"]
mask = (df['TIMESTAMP'] >'2019-04-30 23:45:00') & (df['TIMESTAMP'] <= '2019-11-04')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))

s=df.groupby(['month','year']).sum()

df=s
df=df.reset_index()
df = df.assign(TIMESTAMP=pd.to_datetime(df[['year', 'month']].assign(day=1)))
import numpy as np
import matplotlib.pyplot as plt

df=df.sort_values(by='TIMESTAMP')
plt.rcParams['figure.figsize'] = (12, 7)
fig, ax = plt.subplots()
ax.set_ylabel('Monthly ET (inches)', color='black')
N = len(df["TIMESTAMP"])
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars
yvals = df["Pine_ET_inches"]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["Potato_ET_inches"]
rects2 = ax.bar(ind+width, zvals, width, color='orange')

#kvals = df["prec+irri"]
#rects3 = ax.bar(ind+width+0.27, kvals, width, color='blue',fill=False,edgecolor='blue',linewidth=3.5)

ax.set_xticks(ind+width)
ax.set_xticklabels( ('May2019','Jun2019','July2019','Aug2019','Sep2019'
) )
#ax.legend( (rects1[0], rects2[0],rects3[0]), ('Pines','Potatoes','Precip+Irrigation') ,loc='upper left', fontsize = 'large')
ax.legend( (rects1[0], rects2[0],rects3[0]), ('Pines','Potatoes') ,loc='upper left', fontsize = 'large')
ax.set_ylim(0,8)

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
########################################################################################################

#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\Wisconsin")

#################################################################################################

### graphs3
### frito lay 2018

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
#df=pd.read_csv("wisp_corr.csv")
df=pd.read_csv("data.csv")

mask = (df['TIMESTAMP'] >'2018-04-01 23:45:00') & (df['TIMESTAMP'] <= '2018-10-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["Potato_ET_inches"],linestyle="-",label="Observed ET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2018-08-29'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2018-08-31'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2018-09-07'), color='pink', linestyle='--', lw=2,label="Vine Killed")
ax1.axvline(pd.to_datetime('2018-09-29'), color='blue', linestyle='--', lw=2,label="Harvested")

ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,0.5)
ax1.set_title("Potatoes Variety #1 (AET)",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation(inches)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,2.0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
#################################################################################################

### lamoka 2019

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
#df=pd.read_csv("wisp_corr.csv")
df=pd.read_csv("data.csv")

mask = (df['TIMESTAMP'] >'2019-04-01 23:45:00') & (df['TIMESTAMP'] <= '2019-10-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["Potato_ET_inches"],linestyle="-",label="Observed ET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2019-04-25'), color='orange', linestyle='--', lw=2,label="Crop Planted")
ax1.axvline(pd.to_datetime('2019-08-18'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2019-08-19'), color='pink', linestyle='--', lw=2,label="Vine Killed")
ax1.axvline(pd.to_datetime('2019-09-07'), color='blue', linestyle='--', lw=2,label="Harvested")

ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,0.5)
ax1.set_title("Potatoes Variety#2 2019 (AET)",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation(inches)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,2.0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
####################################################################################################

### lamoka 2019

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=data
mask = (df['TIMESTAMP'] >'2020-04-01 23:45:00') & (df['TIMESTAMP'] <= '2020-10-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["Potato_ET_inches"],linestyle="-",label="Observed ET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2020-04-11'), color='orange', linestyle='--', lw=2,label="Crop Planted")
ax1.axvline(pd.to_datetime('2020-08-27'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2020-09-04'), color='pink', linestyle='--', lw=2,label="Vine Killed")

#ax1.axvline(pd.to_datetime('2020-06-03'), color='black', linestyle='--', lw=2,label="Full Emergence")
#ax1.axvline(pd.to_datetime('2020-06-23'), color='red', linestyle='--', lw=2,label="100% canopy")
ax1.axvline(pd.to_datetime('2020-09-23'), color='blue', linestyle='--', lw=2,label="Harvested")


ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,0.5)
ax1.set_title("Potatoes Variety#2 2020 (AET)",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation(inches)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,2.0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
####################################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=data
mask = (df['TIMESTAMP'] >'2021-04-01 23:45:00') & (df['TIMESTAMP'] <= '2021-10-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["Potato_ET_inches"],linestyle="-",label="Observed ET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2020-04-11'), color='orange', linestyle='--', lw=2,label="Crop Planted")
ax1.axvline(pd.to_datetime('2020-08-27'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2020-09-04'), color='pink', linestyle='--', lw=2,label="Vine Killed")

#ax1.axvline(pd.to_datetime('2020-06-03'), color='black', linestyle='--', lw=2,label="Full Emergence")
#ax1.axvline(pd.to_datetime('2020-06-23'), color='red', linestyle='--', lw=2,label="100% canopy")
ax1.axvline(pd.to_datetime('2020-09-23'), color='blue', linestyle='--', lw=2,label="Harvested")


ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,0.5)
ax1.set_title("Potatoes Variety#2 2020 (AET)",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation(inches)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,2.0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
















#####################################################################################3
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")

x1=df["Potato_ET_inches"]
y1=df["wisp_AET_canopy"]  ## R2, 0.97, slope, 0.99
plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET

#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=1.43, pbias=-41.6, RMSE=0.065",fontsize=14)

#ax.plot(x1,a*x1,'r-', label='Slope=1.11,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model AET (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed AET (inches)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,0.35)
ax.set_xlim(0,0.35)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
########################################################################################

# corrected idso

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")

x1=df["Potato_ET_inches"]
y1=df["wisp_AET_idso_corr"]  ## R2, 0.97, slope, 0.99
plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET

#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=1.43, pbias=-41.6, RMSE=0.065",fontsize=14)

#ax.plot(x1,a*x1,'r-', label='Slope=1.11,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model AET (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed AET (inches)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,0.35)
ax.set_xlim(0,0.35)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
















#### check new coeffificent 


##############################################################################################################3

###WISP crop coefficient corrected ET versus Actual ET
os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")
x1=df["Potato_ET_inches"]
y1=df["AET_corr_f"]  ## R2, 0.97, slope, 0.99
plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.72')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=0.99, pbias=-5, RMSE=0.025",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=1.11,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Corrected AET (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed AET (inches)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,0.35)
ax.set_xlim(0,0.35)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()


#################################################################################################


os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
data=pd.read_csv('growing_season.csv')

## fix SW in

x1=data["SWIn_pot"]  

y1=data["inso_rad"]*11.574 # R2 0.948, slope 0.9655
plt.scatter(x1,y1)

plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="SWin")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.95')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=0.97, pbias=2.09, RMSE=20.5",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model SWin ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed SWin ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,400)
ax.set_xlim(0,400)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

############################################################################################
data=df
### fix temp
#x1=(data["airtempF_pot"]-32 )*(5/9)
x1=(data["airtempF_pot"])
  
y1=(data["inso_temp_c"].astype(float)* 9/5) + 32  ## R2, 0.97, slope, 0.99
plt.scatter(x1,y1)

plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="Air Temperature")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.97')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=0.99, pbias=0.51, RMSE=1.51",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Air Temperature ($\mathregular{F^{0}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed Air Temperature ($\mathregular{F^{0}}$)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,90)
ax.set_xlim(0,90)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#################################################################################################
### vapor pressure 
x1=(data["AVP_pot"])
y1=data["inso_vp_kpa"].astype(float)  ## R2, 0.98, slope, 0.97
plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="Vapor Pressure")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.98')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=0.97, pbias=3.4, RMSE=0.092",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model VP (kpa)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed VP (kpa)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,3.5)
ax.set_xlim(0,3.5)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#################################################################################################
def lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat):
    return angstrom(avg_v_press,avg_temp)*lwu(avg_temp)*clr_ratio(d_to_sol,day_of_year,lat)

lw_net_wisp=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)*11.574
 
### fix lw net
x1=(data["netLW_pot"])
y1=-lw_net_wisp  ## R2, 0.77, slope, 0.345
plt.scatter(x1,y1)


pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="LWnet")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.77')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=0.35, pbias=67, RMSE=44",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model LWnet ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed LWnet ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(10,-160)
ax.set_xlim(10,-160)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#################################################################################################
#fix net radiation
def rnet(avg_temp,avg_v_press,d_to_sol,day_of_year,lat):
    ## calculate lwnet
    lwnnet=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)       
   ## calculate R_n 
    return (1-ALBEDO)*d_to_sol-lwnnet

rnet_wisp=rnet(avg_temp,avg_v_press,d_to_sol,day_of_year,lat)*11.574

x1=(data["pot_Rnet"])
y1=rnet_wisp  ## R2, 0.87, slope, 1.19
plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="Net Radiations")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.87')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=1.2, pbias=-22, RMSE=29.5",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Rnet ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed Rnet ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,300)
ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
#################################################################################################


### compare correct LW with observation

def lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat):
    return angstrom(avg_v_press,avg_temp)*lwu(avg_temp)*clr_ratio(d_to_sol,day_of_year,lat)

lw_net_wisp=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)*11.574
 

### fix temp
x1=(data["netLW_pot"])
y1=-lw_net_wisp  ## R2, 0.77, slope, 0.345
plt.scatter(x1,y1)

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
pred_lwnet=(y1/reg.coef_)
x1=(data["netLW_pot"])
y1=pred_lwnet
y1=y1.iloc[:,0]
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
plt.scatter(x1,y1)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="LWnet")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.77')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=1, pbias=3.5, RMSE=17.8",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Corrected LWnet ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed LWnet ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(10,-160)
ax.set_xlim(10,-160)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
###############################################################################################

### effect of LWnet correction on ET
os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")


x1=df["Potato_ET_inches"]
y1=df["wisp_AET_canp_LW"]  ## R2, 0.97, slope, 0.99
plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.68')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=1.09, pbias=-6.5, RMSE=0.042",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=1.11,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Corrected AET (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed AET (inches)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,0.35)
ax.set_xlim(0,0.35)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

##############################################################################################

###############################################################################################

### effect of Rnet correction on ET
os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")
x1=df["Potato_ET_inches"]
y1=df["wisp_AET_canp_rnet"]  ## R2, 0.97, slope, 0.99
plt.scatter(x1,y1)
pbias=(np.sum(x1-y1)/np.sum(x1))*100
pbias
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)
sqrt(mean_squared_error(x1, y1))
plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.72')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("Slope=1.09, pbias=-6.5, RMSE=0.042",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=1.11,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Corrected AET (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed AET (inches)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,0.35)
ax.set_xlim(0,0.35)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

###########################################################################################


### graphs3
### frito lay 2018

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")
mask = (df['TIMESTAMP'] >'2018-04-30 23:45:00') & (df['TIMESTAMP'] <= '2018-11-04')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["Potato_ET_inches"],linestyle="-",label="Observed ET",color='green')
plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
plt.plot(df["TIMESTAMP"],df["AET_corr_f"],linestyle="--",label="Direct Correction WISP",color='purple')
plt.plot(df["TIMESTAMP"],df["wisp_AET_canp_rnet"],linestyle="--",label="Indirect Correction WISP",color='black')
#ax1.axvline(pd.to_datetime('2018-09-29'), color='blue', linestyle='--', lw=2,label="Harvested")
#ax1.axvline(pd.to_datetime('2018-08-29'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2018-08-31'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2018-09-07'), color='pink', linestyle='--', lw=2,label="Vine Killed")

ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,0.5)
ax1.set_title("Potatoes Variety FL2137 2018 (AET)",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('frito_wisp.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
#################################################################################################

### lamoka 2019

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
df=pd.read_csv("wisp_corr.csv")
mask = (df['TIMESTAMP'] >'2019-04-30 23:45:00') & (df['TIMESTAMP'] <= '2019-11-04')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["Potato_ET_inches"],linestyle="-",label="Observed ET",color='green')
plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
plt.plot(df["TIMESTAMP"],df["AET_corr_f"],linestyle="--",label="Direct Correction WISP",color='purple')
plt.plot(df["TIMESTAMP"],df["wisp_AET_canp_rnet"],linestyle="--",label="Indirect Correction WISP",color='black')

#ax1.axvline(pd.to_datetime('2019-09-07'), color='blue', linestyle='--', lw=2,label="Harvested")
#ax1.axvline(pd.to_datetime('2019-08-18'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2019-08-19'), color='pink', linestyle='--', lw=2,label="Vine Killed")
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,0.5)
ax1.set_title("Potatoes Variety FL2053  2019 (AET)",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('lamoka_wisp_corr.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()





def long (df):
    if df['FID']==0:
        return -88.9
    
    elif df['FID']==1:
        return -89
    
    elif df['FID']==2:
        return -89.1

    elif df['FID']==3:
        return -89.2

    elif df['FID']==4:
        return -89.3
    
    elif df['FID']==5:
        return -89.4
    
    elif df['FID']==6:
        return -89.5
    
    elif df['FID']==7:
        return -89.6
    
    elif df['FID']==8:
        return -88.9

    elif df['FID']==9:
        return -89.0
    
    elif df['FID']==10:
        return -89.1
    
    elif df['FID']==11:
        return -89.2
    
    elif df['FID']==12:
        return -89.3
    
    elif df['FID']==13:
        return -89.4

    elif df['FID']==14:
        return -89.5
    
    elif df['FID']==15:
        return -89.6
    
    elif df['FID']==16:
        return -89.9
    
    elif df['FID']==17:
        return -89.0
    
    elif df['FID']==18:
        return -89.1

    elif df['FID']==19:
        return -89.2
    
    elif df['FID']==20:
        return -89.3
    
    elif df['FID']==21:
        return -89.4
    
    elif df['FID']==22:
        return -89.5
    
    else:
        return -89.6

df["long"]=df.apply(long, axis = 1)

































































