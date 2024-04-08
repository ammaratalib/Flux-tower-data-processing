# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:39:40 2021

@author: Ammara
"""

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import numpy as np
from math import sqrt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
##################################################################################

import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
from scipy.stats.stats import pearsonr

def NSE(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
#    s,o = filter_nan(s,o)
    return 1 - ((sum((s-o)**2))/(sum((o-np.mean(o))**2)))


def will(s,o):

#    s,o = filter_nan(s,o)
    return  1-(sum((o - s)**2))/(sum((abs(s-np.mean(o)) +abs(o-np.mean(o)))**2))


def ubRMSE(o,s):
    MD=(sum(s-o))/o.count()
    RMSD= (sum((s-o)**2))/o.count()
    return sqrt (((RMSD)**2)-((MD)**2))

##############################################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

data=pd.read_csv('paper2_wisp.csv')
back=data
df=back

df["TIMESTAMP"]= pd.to_datetime(df['TIMESTAMP'])

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2020-08-31 23:45:00')

df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

training_2020=df
# check null values before doing data analysis# do it separately on columns such as LW and ET
df.isnull().values.any()

df=df.iloc[:,1:99]
df=df.interpolate()
df.isnull().values.any()

######################################################################################################

## training data 
##################################################################################################33

df["TIMESTAMP"]=training_2020["TIMESTAMP"]

df["lw_net"]=df["LW_in"]-df["LW_out"]

df_time_all=df

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021\paper_2_fig\paper2\results_2022")

df_time_all.to_csv('training_2020_potat.csv', index=False, header=True)



#df=df.rename_axis('TIMESTAMP').reset_index()


lw=pd.concat((df["LW_in"],df["LW_out"],df["lwnet_idso"],df["lwnet_idso_corr"]),axis=1)

lw.isnull().values.any()

#lw=lw.dropna(0)

lw["lw_net"]=lw["LW_in"]-lw["LW_out"]

df=lw

plt.plot(df.lw_net,'r-', label='obs_lw_net')
plt.plot(df.lwnet_idso,'b-', label='lw_net_idso')
plt.plot(df.lwnet_idso_corr,'g-',label='lw_net_idso_corr')

plt.legend()
plt.show()

 
obs=df.lw_net
#pre=df.lwnet_idso
pre=df.lwnet_idso_corr

#x1=pd.DataFrame(x1)*25.4
#y1=pd.DataFrame(y1)*25.4

x1=pd.DataFrame(obs)
y1=pd.DataFrame(pre)

reg = LinearRegression(fit_intercept=True).fit(x1, y1)

reg.score(x1,y1)



NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

#ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE

#RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias


####################################################################################################
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=training_2020

et=pd.concat((df["pot_ET_inches"],df["wisp_ET_adj"],df["wisp_AET_idso_corr"]),axis=1)
df=et

# check null values before doing data analysis# do it separately on columns such as LW and ET
df.isnull().values.any()

x1=df["pot_ET_inches"]
#y1=df["wisp_ET_adj"]  ## R2, 0.97, slope, 0.99

y1=df["wisp_AET_idso_corr"]

x1=pd.DataFrame(x1)*25.4
y1=pd.DataFrame(y1)*25.4


reg = LinearRegression(fit_intercept=True).fit(x1, y1)

reg.score(x1,y1)


obs=df["pot_ET_inches"]*25.4
#pre=df["wisp_ET_adj"]*25.4
pre=df["wisp_AET_idso_corr"]*25.4

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

#ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE

#RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

#####################################################################################################

## this part for testing data 
########################################################################################################33
# start 

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

data=pd.read_csv('paper2_wisp.csv')
back=data
df=back

df["TIMESTAMP"]= pd.to_datetime(df['TIMESTAMP'])

mask = (df['TIMESTAMP'] >'2021-05-01 23:45:00') & (df['TIMESTAMP'] <= '2022-08-31 23:45:00')

df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

testing_2022=df
# check null values before doing data analysis# do it separately on columns such as LW and ET
df.isnull().values.any()

df=df.drop(['pot_ET_inches'], axis=1) # drop ET the drop LW with missing data
df=df.dropna(0)
df["lw_net"]=df["LW_in"]-df["LW_out"]

testing_LW_pot=df

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021\paper_2_fig\paper2\results_2022")

testing_LW_pot.to_csv('testing_LW_pot.csv', index=False, header=True)

lw=df

plt.plot(df.lw_net,'r-', label='obs_lw_net')
plt.plot(df.lwnet_idso,'b-', label='lw_net_idso')
plt.plot(df.lwnet_idso_corr,'g-',label='lw_net_idso_corr')

plt.legend()
plt.show()
 
obs=df.lw_net
#pre=df.lwnet_idso
pre=df.lwnet_idso_corr

#x1=pd.DataFrame(x1)*25.4
#y1=pd.DataFrame(y1)*25.4

x1=pd.DataFrame(obs)
y1=pd.DataFrame(pre)

reg = LinearRegression(fit_intercept=True).fit(x1, y1)

reg.score(x1,y1)

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

#ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE

#RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

########################################################################################################
# next ET testing data

df=testing_2022
df.isnull().values.any()

df=df.drop(['LW_in','LW_out'],axis=1) # drop ET the drop LW with missing data
df=df.dropna(0)

testing_ET_pot=df

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021\paper_2_fig\paper2\results_2022")

testing_ET_pot.to_csv('testing_ET_pot.csv', index=False, header=True)


x1=df["pot_ET_inches"]
y1=df["wisp_ET_adj"]  ## R2, 0.97, slope, 0.99
#y1=df["wisp_AET_idso_corr"]

x1=pd.DataFrame(x1)*25.4
y1=pd.DataFrame(y1)*25.4
reg = LinearRegression(fit_intercept=True).fit(x1, y1)
reg.score(x1,y1)


obs=df["pot_ET_inches"]*25.4
#pre=df["wisp_ET_adj"]*25.4
pre=df["wisp_AET_idso_corr"]*25.4

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

#ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE

#RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

############################################################################################################
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")  # potatoes site

#data=pd.read_csv('paper2_wisp.csv')    # use this for potatoes site


data=pd.read_csv('US-CC1_ET_all.csv')    # use this for US-CC1

back=data
df=back

et=pd.concat((df["pot_ET_inches"],df["wisp_ET_adj"],df["wisp_AET_idso_corr"]),axis=1)

et=et.dropna(0)

df=et

x1=df["pot_ET_inches"]
y1=df["wisp_ET_adj"]  ## R2, 0.97, slope, 0.99


plt.scatter(x1,y1)



x1=pd.DataFrame(x1)*25.4
y1=pd.DataFrame(y1)*25.4

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
#reg.coef_
reg.score(x1, y1)

#r2_score(x1,y1)


obs=df["pot_ET_inches"]*25.4
pre=df["wisp_ET_adj"]*25.4

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

#ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE

#RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias


plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="ET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.57')#ET before corr

ax.set_title("MAE=1.39, RMSE=1.63, pbias=31.3",fontsize=14)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,8)
ax.set_xlim(0,8)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

###########################################################################################
###idso corrected AET

x1=df["pot_ET_inches"]
#y1=df["wisp_AET_idso"]  ## R2, 0.97, slope, 0.99
y1=df["wisp_AET_idso_corr"]  ## R2, 

plt.scatter(x1,y1)
pbias=(np.sum(y1-x1)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)*25.4
y1=pd.DataFrame(y1)*25.4

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
#reg.coef_
reg.score(x1, y1)  ###R2
#r2_score(x1,y1)

obs=df["pot_ET_inches"]*25.4
pre=df["wisp_AET_idso_corr"]*25.4

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

#ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE

#RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="ET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.65')#ET

ax.set_title("MAE=0.68, RMSE=0.89, pbias=-3.2",fontsize=14)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model ET_corr (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,8)
ax.set_xlim(0,8)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

###########################################################################################
data1=back
data2=pd.read_csv('growing_season_no_shoulder.csv')

pot_temp_c=(data2["Tair"]-32 )*(5/9)  # observed temp
#df=data
avg_temp=data2["inso_temp_c"].astype(float)
avg_v_press=data2["inso_vp_kpa"].astype(float)
d_to_sol=data2["inso_rad"].astype(float)



### fix temp
y1=(data["Tair"].astype(float))
#y1=(data["airtempF_pot"])
x1=(data["inso_temp_c"].astype(float))  

plt.scatter(x1,y1)

plt.scatter(x1,y1)
pbias=(np.sum(y1-x1)/np.sum(x1))*100
pbias


x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_

reg.score(x1, y1)  ###R2

r2_score(x1,y1)

sqrt(mean_squared_error(x1, y1))  # rmse  0.8

mean_absolute_error(x1,y1)

plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="Air Temperature")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.92')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("MAE=0.64, RMSE=0.80, pbias=-0.21",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Air Temperature ($\mathregular{C^{0}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed Air Temperature ($\mathregular{C^{0}}$)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,30)
ax.set_xlim(0,30)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

####################################################################################

### vapor pressure 
data=df

x1=(data["AVP_pot"])
y1=data["inso_vp_kpa"].astype(float)  ## R2, 0.98, slope, 0.97

plt.scatter(x1,y1)

pbias=(np.sum(y1-x1)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)

sqrt(mean_squared_error(x1, y1))
mean_absolute_error(x1,y1)


plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="Vapor Pressure")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.96')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("MAE=0.08, RMSE=0.1, pbias=-3.2",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model Vapor Pressure (kpa)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed Vapor Pressure (kpa)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,4)
ax.set_xlim(0,4)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout() 
##########################################################################################



### 
data=df
df['LW_net'] = df['LW_net'].replace(np.nan, 0.0001)

x1=(df["LW_net"])
y1=df.lwnet_idso  ## R2, 0.77, slope, 0.345
plt.scatter(x1,y1)

## only do it here because for one of the data set when lw out-lwin is equal to zero. 

pbias=(np.sum(y1-x1)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)

mean_absolute_error(x1,y1)
sqrt(mean_squared_error(x1, y1))

plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="LWnet")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.76')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.76')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("MAE=44.4, RMSE=46.8, pbias=-69.3",fontsize=14)
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
ax.set_ylim(0,-150)
ax.set_xlim(0,-150)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

############################################################################################
# corrected idso

data=df
df['LW_net'] = df['LW_net'].replace(np.nan, 0.0001)

x1=(df["LW_net"])
y1=df.lwnet_idso_corr ## R2, 0.77, slope, 0.345
plt.scatter(x1,y1)

pbias=(np.sum(y1-x1)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.coef_
reg.score(x1, y1)

mean_absolute_error(x1,y1)
sqrt(mean_squared_error(x1, y1))

plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="LWnet")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.83')#ET
#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')
ax.set_title("MAE=8.1, RMSE=11.1, pbias=-8.0",fontsize=14)
#ax.plot(x1,a*x1,'r-', label='Slope=0.99,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Model LWnet_corr ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed LWnet ($\mathregular{Wm^{2}}$)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,-150)
ax.set_xlim(0,-150)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
###########################################################################################################

##############################################################################################
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP")
data=pd.read_csv('corr_ET.csv')

df=data
df=df.iloc[:,2:99]  # don't worry about end column. not so important
df.isnull().values.any()
df=df.fillna(df.rolling(2,1).mean())
df.isnull().values.any()

df["TIMESTAMP"]=data["TIMESTAMP"]
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))

back=df
#######################################################################################
df=back
df=df
df["prec+irri"]=df["prec_inch"]+df["irrig_inches"]
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.drop('TIMESTAMP', axis=1)  
df=df.rename_axis('TIMESTAMP').reset_index() 
df['year'] = df['TIMESTAMP'].dt.year
df['month']= df['TIMESTAMP'].dt.month
df = df.iloc[1: , :]
df.index = np.arange(0, len(df))
s=df.groupby(['month','year']).sum()
df=s
df["prec+irri"]=df["prec_inch"]+df["irrig_inches"]
df=df.reset_index()
df = df.assign(TIMESTAMP=pd.to_datetime(df[['year', 'month']].assign(day=1)))
df=df.sort_values(by='TIMESTAMP')

plt.rcParams['figure.figsize'] = (12, 7)
fig, ax = plt.subplots()
ax.set_ylabel('Total Monthly water (mm)', color='black')
N = len(df["TIMESTAMP"])
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars
yvals = df["pot_ET_inches"]*25.4
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["prec_inch"]*25.4
rects2 = ax.bar(ind+width, zvals, width, color='orange')
kvals = df["irrig_inches"]*25.4
rects3 = ax.bar(ind+width+0.27, kvals, width, color='blue',fill=False,edgecolor='blue',linewidth=3.5)


ax.set_xticks(ind+width)
ax.set_xticklabels( ('July2018','Aug2018','Jun2019','July2019','Aug2019','Jun2020','July2020','Aug2020'))
ax.legend( (rects1[0], rects2[0],rects3[0]), ('Potatoes AET','Precipitation','Irrigation') ,loc='upper right', prop={'size': 14})

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 20)
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
ax.set_title("Location:US-CS1, US-CS3, US-CS4, Water use in WI ",fontsize=18)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#########################################################################################
data=pd.read_csv('corr_ET.csv')
df=data
mask = (df['TIMESTAMP'] >'2018-04-01 23:45:00') & (df['TIMESTAMP'] <= '2018-10-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])


plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily AET (mm)', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"]*25.4,linestyle="-",label="Observed AET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2018-08-29'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2018-08-31'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2018-09-07'), color='pink', linestyle='--', lw=2,label="Vine Killed")
ax1.axvline(pd.to_datetime('2018-09-29'), color='grey', linestyle='--', lw=2,label="Harvested")
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,8)
ax1.set_title("US-CS1, Potatoes Variety FL2137 in 2018",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation (mm)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches*25.4, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch*25.4, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,80)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

########################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP")
#df=pd.read_csv("wisp_corr.csv")
#df=pd.read_csv("data.csv")

data=pd.read_csv('corr_ET.csv')
df=data

mask = (df['TIMESTAMP'] >'2019-04-01 23:45:00') & (df['TIMESTAMP'] <= '2019-10-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily AET (mm)', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"]*25.4,linestyle="-",label="Observed AET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2019-04-25'), color='orange', linestyle='--', lw=2,label="Crop Planted")
ax1.axvline(pd.to_datetime('2019-08-18'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2019-08-19'), color='pink', linestyle='--', lw=2,label="Vine Killed")
ax1.axvline(pd.to_datetime('2019-09-07'), color='grey', linestyle='--', lw=2,label="Harvested")

ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,8)
ax1.set_title("US-CS3, Potatoes Variety FL2053 in 2019",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation (mm)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches*25.4, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch*25.4, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,80)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

####################################################################################################
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP")
#df=pd.read_csv("wisp_corr.csv")
#df=pd.read_csv("data.csv")

data=pd.read_csv('corr_ET.csv')
df=data
mask = (df['TIMESTAMP'] >'2020-04-01 23:45:00') & (df['TIMESTAMP'] <= '2020-10-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])


plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily AET (mm)', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"]*25.4,linestyle="-",label="Observed AET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2020-04-11'), color='orange', linestyle='--', lw=2,label="Crop Planted")
ax1.axvline(pd.to_datetime('2020-08-27'), color='pink', linestyle='--', lw=2)
ax1.axvline(pd.to_datetime('2020-09-04'), color='pink', linestyle='--', lw=2,label="Vine Killed")

#ax1.axvline(pd.to_datetime('2020-06-03'), color='black', linestyle='--', lw=2,label="Full Emergence")
#ax1.axvline(pd.to_datetime('2020-06-23'), color='red', linestyle='--', lw=2,label="100% canopy")
ax1.axvline(pd.to_datetime('2020-09-23'), color='grey', linestyle='--', lw=2,label="Harvested")


ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,8)
ax1.set_title("US-CS4, Potatoes Variety FL2053 in 2020",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation (mm)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches*25.4, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch*25.4, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,80)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()



########################################################################################################
## inverse modeling

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP")
data=pd.read_csv('corr_ET.csv')

df=data
df=df.iloc[:,2:99]  # don't worry about end column. not so important
df.isnull().values.any()
df=df.fillna(df.rolling(2,1).mean())
df.isnull().values.any()

df["TIMESTAMP"]=data["TIMESTAMP"]
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df["coeff"]=df["pot_ET_inches"]/df["wisp_PET"]
df["week"]= pd.to_datetime(df['TIMESTAMP']).dt.week

mean_coeff=df.groupby(df['week'])['coeff'].mean()

# values are taken from mean_coeff
def avg_coeff (df):
    if df['week']==22:
        return 0.55
    
    elif df['week']==23:
        return 0.70
    
    elif df['week']==24:
        return 0.91

    elif df['week']==25:
        return 0.78

    elif df['week']==26:
        return 0.66
    
    elif df['week']==27:
        return 0.69
    
    elif df['week']==28:
        return 0.74
    
    elif df['week']==29:
        return 0.75
    
    elif df['week']==30:
        return 0.70

    elif df['week']==31:
        return 0.63
    
    elif df['week']==32:
        return 0.63
    
    elif df['week']==33:
        return 0.65
    
    elif df['week']==34:
        return 0.57
    
    elif df['week']==35:
        return 0.54
    
    else:
        return 0.86

df["avg_coeff"]=df.apply(avg_coeff, axis = 1)

df["AET_inver"]=df["wisp_PET"]*df["avg_coeff"]

back=df

#######################################################################################

## overall compariosn inverse modeling results

x1=df["pot_ET_inches"]
#y1=df["wisp_AET_idso"]  ## R2, 0.97, slope, 0.99
y1=df["AET_inver"]  ## R2, 0.97, slope, 0.99

plt.scatter(x1,y1)
pbias=(np.sum(y1-x1)/np.sum(x1))*100
pbias
# if nan values for bias then calculate it manually 

x1=pd.DataFrame(x1)*25.4
y1=pd.DataFrame(y1)*25.4

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
#reg.coef_
reg.score(x1, y1)  ###R2
r2_score(x1,y1)

sqrt(mean_squared_error(x1, y1))
mean_absolute_error(x1,y1)


plt.rcParams['figure.figsize'] = (7,6)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="green",markersize=5,marker='o',label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=2)
line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle='--',linewidth=2)
#ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.71')#ET
ax.plot(x1,a*x1,'r-', label='$\mathregular{R^{2}}$=0.76')#ET

#ax.plot(x1,pred,'x',color="black",markersize=2, label='Fitted Model R2=0.48')

ax.set_title("MAE=0.46, RMSE=0.57, pbias=1.36",fontsize=14)

#ax.plot(x1,a*x1,'r-', label='Slope=1.11,R2=0.46')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 'large'})
ax.set_ylabel('Inverse Model AET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed AET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 16,direction='in')
ax.set_ylim(0,8)
ax.set_xlim(0,8)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


#####################################################################################

# compare inverse models for 2018
mask = (df['TIMESTAMP'] >'2018-06-01') & (df['TIMESTAMP'] <= '2018-08-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])

plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily AET (mm)', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"]*25.4,linestyle="-",label="Observed AET",color='green')
plt.plot(df["TIMESTAMP"],df["wisp_AET_idso"]*25.4,linestyle="--",label="Model AET",color='black')
plt.plot(df["TIMESTAMP"],df["wisp_AET_idso_corr"]*25.4,linestyle="--",label="Model AET_corr",color='orange')
plt.plot(df["TIMESTAMP"],df["AET_inver"]*25.4,linestyle="--",label="Inverse Model AET",color='purple')

#ax1.axvline(pd.to_datetime('2018-08-31'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2018-09-07'), color='pink', linestyle='--', lw=2,label="Vine Killed")
#ax1.axvline(pd.to_datetime('2018-09-29'), color='grey', linestyle='--', lw=2,label="Harvested")
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper center', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,10)
ax1.set_title("US-CS1, Potatoes Variety FL2137 in 2018",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation (mm)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches*25.4, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch*25.4, color='blue', label="Precipitation")
ax2.axvline(pd.to_datetime('2018-08-29'), color='pink', linestyle='--', lw=2,label="Vine Killed")

ax2.legend()
ax2.legend(loc='upper right', fontsize = 'medium')
ax2.set_ylim(0,80)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

###########################################################################################

# compare inverse models for 2019

df=back
mask = (df['TIMESTAMP'] >'2019-06-01') & (df['TIMESTAMP'] <= '2019-08-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])

plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily AET (mm)', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"]*25.4,linestyle="-",label="Observed AET",color='green')
plt.plot(df["TIMESTAMP"],df["wisp_AET_idso"]*25.4,linestyle="--",label="Model AET",color='black')
plt.plot(df["TIMESTAMP"],df["wisp_AET_idso_corr"]*25.4,linestyle="--",label="Model AET_corr",color='orange')
plt.plot(df["TIMESTAMP"],df["AET_inver"]*25.4,linestyle="--",label="Inverse Model AET",color='purple')

#ax1.axvline(pd.to_datetime('2018-08-31'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2018-09-07'), color='pink', linestyle='--', lw=2,label="Vine Killed")
#ax1.axvline(pd.to_datetime('2018-09-29'), color='grey', linestyle='--', lw=2,label="Harvested")
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'medium')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,10)
ax1.set_title("US-CS3, Potatoes Variety FL2053 in 2019",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation (mm)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches*25.4, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch*25.4, color='blue', label="Precipitation")
ax2.axvline(pd.to_datetime('2019-08-19'), color='pink', linestyle='--', lw=2,label="Vine Killed")

ax2.legend()
ax2.legend(loc='upper right', fontsize = 'medium')
ax2.set_ylim(0,80)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

############################################################################################


# compare inverse models for 2020

df=back
mask = (df['TIMESTAMP'] >'2020-06-01') & (df['TIMESTAMP'] <= '2020-08-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])

plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily AET (mm)', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"]*25.4,linestyle="-",label="Observed AET",color='green')
plt.plot(df["TIMESTAMP"],df["wisp_AET_idso"]*25.4,linestyle="--",label="Model AET",color='black')
plt.plot(df["TIMESTAMP"],df["wisp_AET_idso_corr"]*25.4,linestyle="--",label="Model AET_corr",color='orange')
plt.plot(df["TIMESTAMP"],df["AET_inver"]*25.4,linestyle="--",label="Inverse Model AET",color='purple')

#ax1.axvline(pd.to_datetime('2018-08-31'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2018-09-07'), color='pink', linestyle='--', lw=2,label="Vine Killed")
#ax1.axvline(pd.to_datetime('2018-09-29'), color='grey', linestyle='--', lw=2,label="Harvested")
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'medium')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,10)
ax1.set_title("US-CS4, Potatoes Variety FL2053 in 2020",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation (mm)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches*25.4, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch*25.4, color='blue', label="Precipitation")
ax1.axvline(pd.to_datetime('2020-09-04'), color='pink', linestyle='--', lw=2,label="Vine Killed")


ax2.legend()
ax2.legend(loc='upper right', fontsize = 'medium')
ax2.set_ylim(0,80)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

####################################################################################
## overall statistics for inverse model

df=back
mask = (df['TIMESTAMP'] >'2019-06-01') & (df['TIMESTAMP'] <= '2019-08-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))


obs=df["pot_ET_inches"]*25.4
pre=df["wisp_ET_adj"]*25.4

#pre=df["wisp_AET_idso_corr"]*25.4

#pre=df["AET_inver"]*25.4



r2_score(obs,pre)

#NSE(pre,obs)

#will(pre,obs)

#pearsonr(obs,pre)

mean_absolute_error(obs,pre)

#ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE

#RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias
#####################################################################################33

##statistics by year

























