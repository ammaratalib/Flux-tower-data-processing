# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:24:54 2021

@author: Ammara
"""


##### data that prof. used is in local time
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates


import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from datetime import datetime as dt
    
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\pines_2021")
df=pd.read_csv("all_data_pines_2023.csv")
# make sure to add timestamp colum manually. You can copy paste from previous file
# do not worry about if time not visible in TIMESTAMP_start column any more 
 
df.head(10)
df.tail(10)

idx=pd.date_range(start='06/28/2018 00:00:00', periods=96624, freq='30T')
#idx=pd.date_range(start='10/12/2018 00:00:00', periods=85680, freq='30T')
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

#mask = (df['TIMESTAMP'] >'2018-10-11 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')


mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))

back_up=df

reframed=pd.concat((df["DateTime"],df[cols[0]],df[cols[1]],df[cols[2]],df["NEE"],df["LE"],df["H"],df['Rg'],df["Tair"],df['Tsoil'],df["rH"],df["VPD"],df["Ustar"]),axis=1)
reframed.to_csv('gaps.csv', index=False, header=True)

##############################################################################################################

pine_back=back_up

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


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\pines_2021")

fill=pd.read_csv("filled.csv") 
# Rnet is coming from df from top 
fill["pin_Rnet"]=pine_back["Rnet"]
fill["AVP_pin"]=pine_back["AVP"]
fill["rH_pin"]=pine_back["rH"]
fill["VPD_pin"]=pine_back["VPD"]
fill["Rg"]=pine_back["Rg"]
fill["Tair"]=pine_back["Tair"]
fill["SW_in"]=pine_back["SW_IN_1_1_1"]
fill["SW_out"]=pine_back["SW_OUT_1_1_1"]
fill["LW_in"]=pine_back["LW_IN_1_1_1"]
fill["LW_out"]=pine_back["LW_OUT_1_1_1"]


fill["TIMESTAMP"]=pine_back["DateTime"]

df=fill

df.head(10)
df.tail(10)

#### good
#df.index = pd.DatetimeIndex(df.TIMESTAMP)
#idx = pd.date_range(start='2018/06/30', periods=74592, freq='30T')  # days*24*2

#df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
#df= df.replace(-9999, np.nan, regex=True) 
#df=df.iloc[:,2:99]  # don't worry about end column. not so important
#df = df.iloc[:, :-1]

#df=df.rename_axis('TIMESTAMP').reset_index()
#df.index = pd.DatetimeIndex(df.TIMESTAMP)
#d = pd.to_datetime(df["TIMESTAMP"])
#mydatetime = d # or whatever value you want
#df=df.iloc[:,1:99]  # don't worry about end column. not so important
df["ET"]=df["LE"]*((1/1000)*(1/(2.5*1000000))*(86400*1000))
df["pin_ET_inches"]=df["ET"]*0.0393701  # convert into inches for farmers
df.isnull().values.any()

#df["Potato_ET_inches"]= np.where(df["ET"]<0,0.00001,df["ET"])
#df["Potato_ET_EBC_inches"]= np.where(df["ET"]<0,0.00001,df["ET"])
#df=df.rename_axis('TIMESTAMP').reset_index()


mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
fill_pin=df

################################################################################################

df=fill_pin
df=df.resample('D', on='TIMESTAMP').mean()
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
fill_pine=df

###############################################################################################
