# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:24:54 2021

@author: Ammara
"""

#potato data

#http://co2.aos.wisc.edu/data/potato/

##### data that prof. used is in local time

# make sure to add timestamp colum manually. You can copy paste from previous file
# do not worry about if time not visible in TIMESTAMP_start column any more


import datetime
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
import matplotlib.dates as mdates

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
import matplotlib.ticker as ticker
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from datetime import datetime as dt


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

df=pd.read_csv("all_data_potatoes_2022.csv")

df.head(10)
df.tail(10)

idx=pd.date_range(start='06/28/2018 00:00:00', periods=96624, freq='30T')  # half hourly. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()

# do not apply mask yet. Do it later 

back=df
#index is correct date.TIMESTAMP date mess up for missing values
# so let remove messed up timestamp column
df=df.iloc[:,1:38]
#df=df.astype(float) 
df.head(10)
df.tail(10)
temp=df

df["LE"]=df.LE_1_1_1+df.SLE_1_1_1
df["H"]=df.H_1_1_1+df.SH_1_1_1
df=df.rename_axis('TIMESTAMP').reset_index()
df["date"] = pd.to_datetime(df.TIMESTAMP)

df["Rg"]=df['SW_IN_1_1_1']

#calculate vapor pressure deficiet
#prepare some variables for eddy proc for gap filling
df["Tair"]= df["TA_1_1_1"] 
df["rH"]=df["RH_1_1_1"]

es=0.6108*np.exp((17.27*df["Tair"])/(df["Tair"]+237.3))
#ea is actual vapoure pressure in millibar and es is saturated vapor pressure 

ea=df["rH"]/100*es
VPD=(es-ea)*10
df['VPD']=VPD   # in kpa
df["AVP"]= ea # in kpa
df.head(10)
df.tail(10)
###########################################################################################################
df['year'] = df['TIMESTAMP'].dt.year
#df['day'] = df['TIMESTAMP'].dt.day
df['day'] = df['TIMESTAMP'].dt.dayofyear

# eddy proc need  these columns in the start
df.insert(1, 'Year',df['year'])
df.insert(2, 'DoY', df['day'])
#add hour column for eddy proc
#try using zero
s=pd.Series(np.arange(0.5,24.5,0.5))
#######################################################################################################
#change this based on data size (number of days of data), 
#([s]*(#number of days)
k=pd.concat([s]*(2013),axis=0)
k.reset_index(drop=True, inplace=True)
#k.index = np.arange(1, len(k)+1)
df.insert(3,'Hour',k)
df=df.iloc[:,1:103]
df.tail(10)
#########################################################################################################

df["LE1"] = np.where(df["LE"]<-200, np.NaN, df["LE"])
df["LE2"] = np.where(df["LE1"]>800, np.NaN, df["LE1"])
df["LE"]=df["LE2"]


df["H1"] = np.where(df["H"]<-200, np.NaN, df["H"])
df["H2"] = np.where(df["H1"]>800, np.NaN, df["H1"])
df["H"]=df["H2"]


df["NEE1"] = np.where(df["NEE_F"]<-50, np.NaN, df["NEE_F"])
df["NEE2"] = np.where(df["NEE1"]>50, np.NaN, df["NEE1"])
df["NEE"]=df["NEE2"]
df["FC"]=df["NEE"]
df["Rnet1"] = np.where(df["NETRAD_1_1_1"]<-500, np.NaN, df["NETRAD_1_1_1"])
df["Rnet2"] = np.where(df["Rnet1"]>1000, np.NaN, df["Rnet1"])
df["Rnet"]=df["Rnet2"]

df["Rg1"] = np.where(df["Rg"]<-50, np.NaN, df["Rg"])
df["Rg2"] = np.where(df["Rg1"]>1200, np.NaN, df["Rg1"])
df["Rg"]=df["Rg2"]

df["Ustar1"] = np.where(df["USTAR_1_1_1"]<0, np.NaN, df["USTAR_1_1_1"])
df["Ustar2"] = np.where(df["Ustar1"]>3, np.NaN, df["Ustar1"])
df["Ustar"]=df["Ustar2"]

df["Tair1"] = np.where(df["Tair"]<-200, np.NaN, df["Tair"])
df["Tair2"] = np.where(df["Tair1"]>200, np.NaN, df["Tair1"])
df["Tair"]=df["Tair2"]


df["rH1"] = np.where(df["rH"]<0, np.NaN, df["rH"])
df["rH2"] = np.where(df["rH1"]>100, np.NaN, df["rH1"])
df["rH"]=df["rH2"]

df["VPD1"] = np.where(df["VPD"]<0, np.NaN, df["VPD"])
df["VPD2"] = np.where(df["VPD1"]>100, np.NaN, df["VPD1"])
df["VPD"]=df["VPD2"]

df["DateTime"]=df["date"]
df["TIMESTAMP"]=df["date"]

df['Tsoil']=np.nan


cols = df.columns.tolist()
back=df

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')


# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))

back_up=df

##################################################################################################################
reframed=pd.concat((df["DateTime"],df[cols[0]],df[cols[1]],df[cols[2]],df["NEE"],df["LE"],df["H"],df['Rg'],df["Tair"],df['Tsoil'],df["rH"],df["VPD"],df["Ustar"]),axis=1)
reframed.to_csv('gaps.csv', index=False, header=True)

##############################################################################################################

potat_back=back_up

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates

import datetime
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

#wis_pot=pd.read_csv("corrected.csv") 

fill=pd.read_csv("filled.csv") 
# Rnet is coming from df from top 
fill["pot_Rnet"]=potat_back["Rnet"]
fill["AVP_pot"]=potat_back["AVP"]
fill["rH_pot"]=potat_back["rH"]
fill["VPD_pot"]=potat_back["VPD"]
fill["Rg"]=potat_back["Rg"]
fill["Tair"]=potat_back["Tair"]
fill["SW_in"]=potat_back["SW_IN_1_1_1"]
fill["SW_out"]=potat_back["SW_OUT_1_1_1"]
fill["LW_in"]=potat_back["LW_IN_1_1_1"]
fill["LW_out"]=potat_back["LW_OUT_1_1_1"]


fill["TIMESTAMP"]=potat_back["DateTime"]

df=fill

df.head(10)
df.tail(10)

#### good so far

df["ET"]=df["LE"]*((1/1000)*(1/(2.5*1000000))*(86400*1000))
df["pot_ET_inches"]=df["ET"]*0.0393701  # convert into inches for farmers
df.isnull().values.any()



mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

fill_pot=df

###################################################################################################

### this part is for emissivity

# time zone change 6 hour different. this part is only for emissivity

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_csv("potsolar_2022.csv") 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
idx = pd.date_range(start='2018/01/01', periods=87648, freq='30T')  # days*24*2

df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
#df = df.iloc[:, :-1]
df=df.rename_axis('TIMESTAMP').reset_index()
df.index = pd.DatetimeIndex(df.TIMESTAMP)

d = pd.to_datetime(df["TIMESTAMP"])
mydatetime = d # or whatever value you want
local = mydatetime - datetime.timedelta(hours=6)
df["local"]=local 
df["TIMESTAMP1"]=local 

df=df.drop('TIMESTAMP', axis=1)  
df=df.reset_index()
df["TIMESTAMP"]=df["TIMESTAMP1"]
df=df.drop('TIMESTAMP1', axis=1)  

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

back_hour=df

mask = (fill_pot['TIMESTAMP'] >'2018-06-29 23:45:00') & (fill_pot['TIMESTAMP'] <= '2022-09-30 23:45:00')

df = fill_pot.loc[mask]
df.index = np.arange(0, len(df))

df["pot_solar"]=back_hour["Solar_W_m-2"]

plt.plot(df["pot_solar"],'g-')
plt.plot(df["SW_in"],'r')

back=df

df["Hr"]=back["TIMESTAMP"].dt.hour

df.isnull().values.any()
df["sol_diff"]=df["pot_solar"]-df["SW_in"]
df=pd.concat((back_hour["TIMESTAMP"],df['Hr'],df),axis=1)

mask = (df['sol_diff'] >=0)
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
plt.plot(df["pot_solar"],'g-')
plt.plot(df["SW_in"],'r')
#df=pd.concat((fill_pot["TIMESTAMP"],fill_pot['Hr'],df),axis=1)

df=df.drop(['Hr'], axis=1)
df["Hr"]=back["TIMESTAMP"].dt.hour
back=df

#### good so far
mask = (df['Hr'] >=10) & (df['Hr'] <= 14)
df = df.loc[mask]
df.index = np.arange(0, len(df))


mask = (df['SW_in'] >=(.9*df['pot_solar']))
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))


plt.plot(df["pot_solar"],'g-')
plt.plot(df["SW_in"],'r')
back=df

#df["es_obs"]=0.96*(df["LW_in"]/df["LW_out"])

df.to_csv('emiss_hour.csv', index=False, header=True)

## remove one of the timestamp by hand

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_csv("emiss_hour.csv")

# hourly emissivity

plt.plot(df["es_obs"])

hourly_emmis=df

df["TIMESTAMP"]= pd.to_datetime(df['TIMESTAMP'])

df=df.resample('D', on='TIMESTAMP').mean()
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()

# daily emissivty 
plt.plot(df["es_obs"])

df["es_obs"]= np.where(df["es_obs"]>1,0.999,df["es_obs"])
emiss=df

plt.plot(df["es_obs"])
emiss.to_csv('emiss.csv', index=False, header=True)

### emissivity part ends
##########################################################################################################

# 

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="new_canopy",header=0)
#df=df.resample('D', on='TIMESTAMP').sum()
#df=df.reset_index()
idx=pd.date_range(start='01/01/2018', periods= 2069, freq='D')  ## days
df.index = pd.DatetimeIndex(df.TIMESTAMP)
#df["can_cov"]=df.mean(axis=1)
back_canop=df
#df=df.interpolate()


df=df.iloc[:,1:99]  # don't worry about end column. not so important
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
#df.resample('D', on='TIMESTAMP').mean()
#df=df.iloc[:,2:99]  # don't worry about end column. not so important

df=df.interpolate()

#df=df.fillna(df.rolling(2,1).mean())
#df=df.fillna(df.rolling(3,1).mean())
#df=df.fillna(df.rolling(9,1).mean())

df.isnull().values.any()
df["canop_cover"]=df["canop"]*100
canop=df

df["can_cov"]=df["canop_cover"]

#df=df.fillna(df.rolling(15,1).mean())
#df=df.fillna(df.rolling(20,1).mean())
#df.isnull().values.any()
#df=df.fillna(df.rolling(30,1).mean())
#df.isnull().values.any()
df=df.rename_axis('TIMESTAMP').reset_index()
canop=df
df=canop
canop=df

df=canop
#mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
#df.loc[mask]
#df = df.loc[mask]
#df.index = np.arange(0, len(df))
canop=df


## no need to apply mask here. Mask is applied next 
#import matplotlib.pyplot  as pyplot
plt.plot(canop["canop_cover"])

########################################################################################################33

### wisp ET
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="wisp_ET",header=0)
# no of days
idx=pd.date_range(start='01/01/2018', periods=2069, freq='D')

df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
df=df.rename_axis('TIMESTAMP').reset_index()
df.isnull().values.any()

df['ET_inches']=df['ET_inches'].astype(object).astype(float)

df=df.resample('D', on='TIMESTAMP').mean()
df=df.reset_index()

wisp=df
wisp["cum_ET"]=wisp["ET_inches"].cumsum()
wisp["canop"]=canop["can_cov"]

#wisp["ET_inches_1"]= np.where(wisp["ET_inches"]<0,0.00001,wisp["ET_inches"])
#wisp["ET_inches"]=wisp["ET_inches_1"]

def adj_AET (wisp):
    if wisp['canop']>=80:
        return wisp['ET_inches']
    else:
        return wisp['ET_inches']*((wisp['canop']/80+0.0833))

wisp["corr"]= wisp.apply(adj_AET, axis = 1)
wisp["wisp_ET_adj"]=wisp["corr"]
wisp["wisp_PET"]=wisp['ET_inches']
wisp["wisp_ETcum"]=wisp["corr"].cumsum()
plt.plot(wisp["ET_inches"])
plt.plot(wisp["wisp_ETcum"],'r')
df=wisp
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
wisp=df

plt.plot(df.wisp_PET,'r--', label="WISP PET")# daily 
#plt.legend()
#plt.show()


plt.plot(df.wisp_ET_adj,'g--', label="WISP AET")# daily 

#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()

# so far good 
#########################################################################################################

### add a precip inches 

#download hancock precip
#https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USC00473405/detail

#noaa standard unit inches and metric units mm
# use mm metric

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="precip",header=0)
df["TIMESTAMP"] = pd.to_datetime(df.TIMESTAMP)
idx=pd.date_range(start='01/01/2018', periods=2069, freq='D')
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
df.isnull().values.any()

df=df.interpolate()


#df=df.fillna(df.rolling(2,1).mean())
#df=df.fillna(df.rolling(3,1).mean())
#df=df.fillna(df.rolling(4,1).mean())

df.isnull().values.any()
df=df.rename_axis('TIMESTAMP').reset_index()

df=df.resample('D', on='TIMESTAMP').sum()
df=df.reset_index()

df["prec_inch"]=df["PRCP_mm"]*0.0393701
prec=df
prec["prec_cum"]=prec["prec_inch"].cumsum()
df=prec
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
prec=df

###########################################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="irrig",header=0)
df=df.resample('D', on='TIMESTAMP').sum()
df=df.reset_index()

idx=pd.date_range(start='01/01/2018', periods= 2069, freq='D')  ## days
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.iloc[:,1:99]  # don't worry about end column. not so important

df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, 0, regex=True) 
df=df.rename_axis('TIMESTAMP').reset_index()
df.index = pd.DatetimeIndex(df.TIMESTAMP)
d = pd.to_datetime(df["TIMESTAMP"])
mydatetime = d # or whatever value you want
df=df.iloc[:,1:99]  # don't worry about end column. not so important
df=df.rename_axis('TIMESTAMP').reset_index()

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
irrig=df
###################################################################################### FC is NEE
#df=daily potato data

df=fill_pot
df=df.resample('D', on='TIMESTAMP').mean()

df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
fill_pote=df

df=fill_pot

#df=df.rename_axis('TIMESTAMP').reset_index()
#df=df.resample('D', on='TIMESTAMP').mean()
#df=df.reset_index()

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
#fill_pot=df


plt.plot(wisp.wisp_PET,'r--', label="WISP PET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
plt.legend()
plt.show()


plt.plot(wisp.wisp_ET_adj,'g--', label="WISP AET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()


#df["pot_ET_inches"]= np.where(df["pot_ET_inches"]<0,0.00001,df["pot_ET_inches"])
#df["pot_ET_EBC_inches"]= np.where(df["pot_ET_EBC_inches"]<0,0.00001,df["pot_ET_EBC_inches"])

##################################################################################################
###bring inso data
#
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="inso",header=0)
idx=pd.date_range(start='01/01/2018', periods=2069, freq='D')
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
#df=df.rename_axis('TIMESTAMP').reset_index()
#df=df.set_index(['TIMESTAMP'])
df.isnull().values.any()
df=df.rename_axis('TIMESTAMP').reset_index()

#df=df.resample('D', on='TIMESTAMP').mean()
#df=df.reset_index()
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
#this line will only work if index is timestamp
inso=df
inso["inso_rad"]=inso["inso_rad_MJ"]

###############################################################################################

df=fill_pot
df=df.resample('D', on='TIMESTAMP').mean()
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
fill_pota=df
####################################################################################################

# next without pine

data=pd.concat((fill_pota,inso["inso_rad"],inso["inso_vp_kpa"].astype(float),inso["inso_temp_c"].astype(float),wisp["canop"],prec["prec_inch"],wisp["wisp_PET"],wisp["wisp_ET_adj"],irrig["irrig_inches"]),axis=1)
df=data

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
growin=df
growin.to_csv('growing_season.csv', index=False, header=True)
data=pd.read_csv('growing_season.csv')
df=data

plt.plot(wisp.wisp_PET,'r--', label="WISP PET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
plt.legend()
plt.show()

plt.plot(wisp.wisp_ET_adj,'g--', label="WISP AET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()

####################################################################################################
#### bring pine part

#######################################################3

data=pd.concat((fill_pota,fill_pine,inso["inso_rad"],inso["inso_vp_kpa"],inso["inso_temp_c"],wisp["canop"],prec["prec_inch"],irrig['irrig_inches'],wisp["wisp_PET"],wisp["wisp_ET_adj"]),axis=1)

#use this
#data=pd.concat((fill_pota,inso["inso_rad"],inso["inso_vp_kpa"],inso["inso_temp_c"],wisp["canop"],prec["prec_inch"],wisp["wisp_PET"],wisp["wisp_ET_adj"]),axis=1)

###make sure to check wisp temp and vapor pressure column is not greyed out 
# dont round off, it will effect wisp lower ET values and will make them zero 
#data=data.round(2)


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data.to_csv('combine_pot_pine.csv', index=False, header=True)

# manually remove one timestamp


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data=pd.read_csv("combine_pot_pine.csv") 


data_back=data
df=data_back

df=df.iloc[:,3:99] 



df=df.fillna(df.rolling(2,1).mean())
df=df.fillna(df.rolling(3,1).mean())
#df=df.fillna(df.rolling(4,1).mean())

df.isnull().values.any()

df["TIMESTAMP"]=data_back["TIMESTAMP"]
df["Year"]=data_back["Year"]

data=df

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data.to_csv('combine_pot_pin_fin.csv', index=False, header=True)


#################################################################################################
back=data
#########################################################################################
#### stop read line below
#####################################################################################
# manualy remove duplicate columns timestmp colum from data
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

 #df= pd.concat((fill_pote,fill_pine),axis=1)

df=pd.read_csv("combine_pot_pin_fin.csv") 


df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=4) & (df['month'] <= 9)
df = df.loc[mask]
df.index = np.arange(0, len(df))
growin=df
growin.to_csv('growing_season.csv', index=False, header=True)
data=pd.read_csv('growing_season.csv')
df=data

plt.plot(wisp.wisp_PET,'r--', label="WISP PET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
plt.legend()
plt.show()

plt.plot(wisp.wisp_ET_adj,'g--', label="WISP AET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()

#####################################################################################
# fix missing data: upto 4 days

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_csv("growing_season.csv") 
back=df
df=df.iloc[:,2:99]  # don't worry about end column. not so important
#df=df.rename_axis('TIMESTAMP').reset_index()
df=df.fillna(df.rolling(2,1).mean())
df=df.fillna(df.rolling(3,1).mean())
df=df.fillna(df.rolling(4,1).mean())
df.isnull().values.any()
df["TIMESTAMP"]=back["TIMESTAMP"]
data=df
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data.to_csv('data.csv', index=False, header=True)

##############################################################################################
# save growinf season data after gap filling
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
 #df= pd.concat((fill_pote,fill_pine),axis=1)

df=pd.read_csv("data.csv") 

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=4) & (df['month'] <= 9)
df = df.loc[mask]
df.index = np.arange(0, len(df))
growin=df
growin.to_csv('growing_season.csv', index=False, header=True)



























#############################################################################################
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year

s=df.groupby(['month','year']).sum()

df=s
df["prec+irri"]=df["prec_inch"]+df["irrig_inches"]
df=df.reset_index()
df = df.assign(TIMESTAMP=pd.to_datetime(df[['year', 'month']].assign(day=1)))
import numpy as np
import matplotlib.pyplot as plt



df=df.sort_values(by='TIMESTAMP')
df = df.iloc[1: , :]
plt.rcParams['figure.figsize'] = (16, 8)
fig, ax = plt.subplots()
ax.set_ylabel('Monthly water (inches)', color='black')
N = len(df["TIMESTAMP"])
ind = np.arange(N)  # the x locations for the groups
width = 0.1     # the width of the bars
yvals = df["pin_ET_inches"]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["pot_ET_inches"]
rects2 = ax.bar(ind+width, zvals, width, color='orange')
kvals = df["prec_inch"]
rects3 = ax.bar(ind+width+0.15, kvals, width, color='purple',edgecolor='purple',linewidth=1)
vvals = df["irrig_inches"]
vrects3 = ax.bar(ind+width+0.15+0.15, vvals, width, color='black',edgecolor='black',linewidth=1)

ax.set_xticks(ind+width+0.27)
#ax.set_xticklabels( ('Jun2018','July2018','Aug2018','Sep2018','May2019','Jun2019','July2019','Aug2019','Sep2019','May2020','Jun2020','July2020','Aug2020','Sep2020','May2021','Jun2021','July2021','Aug2021','Sep2021','May2022','Jun2022','July2022','Aug2022'))
ax.set_xticklabels( ('July2018','Aug2018','Sep2018','May2019','Jun2019','July2019','Aug2019','Sep2019','May2020','Jun2020','July2020','Aug2020','Sep2020','May2021','Jun2021','July2021','Aug2021','Sep2021'))
ax.set_ylim(0,8)

ax.legend( (rects1[0], rects2[0],rects3[0],vrects3[0]), ('Pines','Potatoes','Precip','Irrigation') ,loc='upper right', fontsize = 'large')

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#########################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
data=pd.read_csv("data.csv")

df=data
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.drop(columns='TIMESTAMP')
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()


mask = (df['TIMESTAMP'] >'2021-04-01') & (df['TIMESTAMP'] <= '2021-10-31')

df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"],linestyle="-",label="Observed ET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2021-04-27'), color='orange', linestyle='--', lw=2,label="Crop Planted")
ax1.axvline(pd.to_datetime('2021-08-16'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2021-08-27'), color='pink', linestyle='--', lw=2,label="Vine Killed")

#ax1.axvline(pd.to_datetime('2020-06-03'), color='black', linestyle='--', lw=2,label="Full Emergence")
#ax1.axvline(pd.to_datetime('2020-06-23'), color='red', linestyle='--', lw=2,label="100% canopy")
ax1.axvline(pd.to_datetime('2021-09-15'), color='blue', linestyle='--', lw=2,label="Harvested")


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
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


####################################################################################

##cumulative

df=data
#df["wisp_ET_adj"]= np.where(df["wisp_ET_adj"]<0,0.00001,df["wisp_ET_adj"])
#df["wisp_AETidso_corr"]= np.where(df["wisp_AET_idso_corr"]<0,0.00001,df["wisp_AET_idso_corr"])
df["pot_ET_inches"]= np.where(df["pot_ET_inches"]<0,0.00001,df["pot_ET_inches"])
df["pin_ET_inches"]= np.where(df["pin_ET_inches"]<0,0.00001,df["pin_ET_inches"])
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.drop(columns='TIMESTAMP')
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()


month= pd.to_datetime(df['TIMESTAMP']).dt.month
year= pd.to_datetime(df['TIMESTAMP']).dt.year
#mask = (df['TIMESTAMP'] >'2019-04-30 23:45:00') & (df['TIMESTAMP'] <= '2020-04-15')
mask = (df['TIMESTAMP'] >='2019-04-30 00:00:00') & (df['TIMESTAMP'] <= '2019-09-30 00:00:00')
mask = (df['TIMESTAMP'] >='2020-04-30 00:00:00') & (df['TIMESTAMP'] <= '2020-09-30 00:00:00')
mask = (df['TIMESTAMP'] >='2021-04-30 00:00:00') & (df['TIMESTAMP'] <= '2021-09-30 00:00:00')

df = df.loc[mask]
df.index = np.arange(0, len(df))

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 9)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df_2019=df
df_2020=df
df_2021=df

df_2021["pot_ET_inches"]=df['pot_ET_inches'].fillna(0.1)
df_2021['TIMESTAMP']=pd.to_datetime(df['TIMESTAMP'])

plt.rcParams['figure.figsize'] = (12, 7)
fig, ax = plt.subplots()
df["TIMESTAMP"]=pd.to_datetime(df['TIMESTAMP'])
ax.set_ylabel('ET (inches)', color='black')
ax.set_xlabel('Month', color='black')
#plt.plot(df_2019["TIMESTAMP"],df_2019["pot_ET_inches"].cumsum(),linestyle="-",label="Potatoes year 2019",color='orange')
#plt.plot(df_2019["TIMESTAMP"],df_2019["pin_ET_inches"].cumsum(),linestyle="-",label="Pine year 2019",color='green')

#plt.plot(df_2020["TIMESTAMP"],df_2020["pot_ET_inches"].cumsum(),linestyle="-",label="Potatoes year 2020",color='orange')
#plt.plot(df_2020["TIMESTAMP"],df_2020["pin_ET_inches"].cumsum(),linestyle="-",label="Pine year 2020",color='green')

plt.plot(df_2021["TIMESTAMP"],df_2021["pot_ET_inches"].cumsum(),linestyle="-",label="Potatoes year 2021",color='orange')
plt.plot(df_2021["TIMESTAMP"],df_2021["pin_ET_inches"].cumsum(),linestyle="-",label="Pine year 2021",color='green')

#plt.plot(df["TIMESTAMP"],df["wisp_ET_adj"].cumsum(),linestyle="-",label="Wisp",color='skyblue')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_idso_corr"].cumsum(),linestyle="-",label="Corrected Wisp",color='brown')

#ax.legend()
ax.set_xticklabels( ('May','June','July','Aug','Sep','Oct'))

ax.legend(loc='upper left',prop={'size': 15})
#ax2.legend(loc='upper left', fontsize = 'large',prop={'size': 6})
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))

#lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 15})
ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
ax.set_ylim(0,25)
ax.set_title("Cumulative Growing Season ET (inches)",fontsize=18)
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
###############################################################################################

#####################################################################################
df=data

df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df


s=df.groupby(['year']).sum()

df=s.rename_axis('year').reset_index()
df=df.reset_index()


df=df.sort_values(by='year')
#df = df.iloc[1: , :]
plt.rcParams['figure.figsize'] = (16, 8)
fig, ax = plt.subplots()
ax.set_ylabel('Total May-Aug ET (inches)', color='black')
N = len(df["year"])
ind = np.arange(N)  # the x locations for the groups
width = 0.1     # the width of the bars
yvals = df["pot_ET_inches"]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["wisp_ET_adj"]
rects2 = ax.bar(ind+width, zvals, width, color='orange')
kvals = df["wisp_AETidso_corr"]
rects3 = ax.bar(ind+width+0.1, kvals, width, color='purple',edgecolor='purple',linewidth=1)
#vvals = df["irrig_inches"]
#vrects3 = ax.bar(ind+width+0.15+0.15, vvals, width, color='black',edgecolor='black',linewidth=1)

ax.set_xticks(ind+width+0.15)
#ax.set_xticklabels( ('Jun2018','July2018','Aug2018','Sep2018','May2019','Jun2019','July2019','Aug2019','Sep2019','May2020','Jun2020','July2020','Aug2020','Sep2020','May2021','Jun2021','July2021','Aug2021','Sep2021'))
ax.set_xticklabels( ('2018','2019','2020','2021'))
ax.set_ylim(0,20)

ax.legend( (rects1[0], rects2[0],rects3[0]), ('Tower Observed ET','WISP ET','WISP Corrected ET') ,loc='upper left', fontsize = 'large')

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#########################################################################################
