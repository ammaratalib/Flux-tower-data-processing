# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:36:49 2020

@author: Ammara
"""
#https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.49710745317
##########################################################################################################
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

##############################################################################################################

import math
SOLAR_CONSTANT=1367
WATTS_TO_MJ_PER_DAY=0.0864
STEFAN_WATTS=0.0000000567
##  mutiply above two values
STEFAN_MJ_PER_DAY=(WATTS_TO_MJ_PER_DAY*STEFAN_WATTS)
SFCEMISS=0.96  # surface emissivity

ALBEDO=0.25


lat=44.2
#lat=43.3  #DFK
# convert degrees to radian
def degrees_to_rads(degrees):
    return degrees*(np.pi/180)

    
def declin(day_of_year):
    return 0.41*np.cos(2*np.pi*(day_of_year-172)/365)

## sunrise angle will be negative 
    
def sunrise_angle(day_of_year,lat):
    return np.arccos(-1*np.tan(declin(day_of_year))*np.tan(degrees_to_rads(lat)))
  

def sunrise_hour(day_of_year,lat):
    return 12-(12/np.pi)*sunrise_angle(day_of_year,lat)
    
    
def day_hours(day_of_year,lat):
    return 24-2*sunrise_hour(day_of_year,lat)
    

#### find components that will be used in clr_ratio
### clr_ratio is part of calculating what fraction of lw will be used

def av_eir(day_of_year):
    return SOLAR_CONSTANT*(1+0.035*np.cos(2*np.pi*day_of_year/365))
      
def to_eir(day_of_year,lat):
    return (0.0864/np.pi)*av_eir(day_of_year)*\
    (sunrise_angle(day_of_year,lat)*
     np.sin(declin(day_of_year))*
     np.sin(degrees_to_rads(lat))+
     np.cos(declin(day_of_year))*
     np.cos(degrees_to_rads(lat))*
     np.sin(sunrise_angle(day_of_year,lat)))


#s=to_eir(day_of_year,lat)
    
def to_clr(day_of_year,lat):
    return to_eir(day_of_year,lat)*(-0.7+0.86*day_hours(day_of_year,lat))/day_hours(day_of_year,lat)


###############################################################################################################
## longwave upward
##############################################################################################################
## uses surafce emissivity and temperature
def lwu(avg_temp):
    return SFCEMISS*STEFAN_MJ_PER_DAY*(273.15+avg_temp)**4

###############################################################################################################
## slope of saturation vapor curve
##############################################################################################################

def sfactor(avg_temp):
    return 0.398+(0.0171*avg_temp)-(0.000142*avg_temp*avg_temp)

##############################################################################################################
# clear sky emissivity (dimentionless) calculated y using the method of Idso (1981)
#avg_v_press=avg_v_press.loc[1:]
#avg_temp=avg_temp.loc[1:]
## idso emissivity   
def sky_emiss(avg_v_press,avg_temp):
    if(avg_v_press>0.5).any():
        return 0.7+(5.95e-4)*avg_v_press*np.exp(1500/(273+avg_temp))
    else:
        return (1-0.261*np.exp(-0.000777*avg_temp*avg_temp))    
  
#########################################################################################
# after correction       
def sky_emiss(avg_v_press,avg_temp):
        if(avg_v_press>0.5).any():
            return 0.544+(6.4e-04)*avg_v_press*np.exp(1500/(273+avg_temp))
        else:
            return (1-0.261*np.exp(-0.000777*avg_temp*avg_temp))    


#b= 0.706193
#x=   8.016e-05   
 
    
##  0.65
## 5.35 * 10-4
#################################################################################################    
############################################################################################################

## calcultae 1- clear sky emissivity factor for Long wave

def angstrom(avg_v_press,avg_temp):
    return 1-sky_emiss(avg_v_press,avg_temp)/SFCEMISS

#############################################################################################################
# ratio of measured insolation divided by the theoratical value 
# calculated for clear-air conditions
def clr_ratio (d_to_sol,day_of_year,lat):
    tc=to_clr(day_of_year,lat)
    # never return higher than 1
    if (d_to_sol/tc>1).any():
        return 1
    else:
        return d_to_sol/tc
#############################################################################################################
## calculate net thermal infrared influx term (Ln) of the total net radiation consisting of the two directional terms upwelling and downwelling

########
 ###run input data first   
########

def lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat):
    return angstrom(avg_v_press,avg_temp)*lwu(avg_temp)*clr_ratio(d_to_sol,day_of_year,lat)


def et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat):
    ## calculate lwnet
    lwnnet=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)       
   ## calculate R_n 
    net_radiation=(1-ALBEDO)*d_to_sol-lwnnet
    ## formula for evapotranspiration
    ret1=1.26*sfactor(avg_temp)*net_radiation
    # assume 62.3 is the conversion factor but unable to determine 
    return ret1/62.3

######
#emiss["lwout_emis_obs"]=(-long_wave*11.4) 

       
##############################################################################################################

## temperature are in celcius
# avg_v_pressure in kpa (kilopascal)
# d_to_sol is insolation reading in MJ/day (Mega joules/day)
#lat is latitude in fractional degrees
###############################################################################################################
data=pd.read_csv('growing_season.csv')

#data=pd.read_csv('fill_DFK_ET_wisp.csv')



df=data

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
growin=df
growin.to_csv('growing_season_no_shoulder.csv', index=False, header=True)

data=df
pot_temp_c=(data["Tair"]-32 )*(5/9)  # observed temp
#df=data
avg_temp=data["inso_temp_c"].astype(float)
avg_v_press=data["inso_vp_kpa"].astype(float)
d_to_sol=data["inso_rad"].astype(float)
day_of_year=pd.to_datetime(data["TIMESTAMP"]).dt.dayofyear
data["day_of_year"]=day_of_year
## calculate Idso emissivity
#cee["idso_em_corr"]=sky_emiss(avg_v_press,avg_temp) 
#cee=df
#######################################################################################################
# run this part for idso emissivity
long_wave= lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat) 

plt.plot(-long_wave*11.4)

data["lwnet_idso"]=(-long_wave*11.4)
data["idso_emis"]=sky_emiss(avg_v_press,avg_temp) 

data["wisp_PET_idso"]=et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat)

data['canop']=data['canop'] # use that for other sites

def adj_AET (data):
    if data['canop']>=80:
        return data["wisp_PET_idso"]
    else:
        return data["wisp_PET_idso"]*((data['canop']/80+0.0833))

data["wisp_AET_idso"]=data.apply(adj_AET, axis = 1)

####################################################################################################
### now run correct emissivity 

long_wave= lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat) 

data["lwnet_idso_corr"]=(-long_wave*11.4)
data["idso_emis_corr"]=sky_emiss(avg_v_press,avg_temp) 

data["wisp_PET_idso_corr"]=et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat)

def adj_AET (data):
    if data['canop']>=80:
        return data["wisp_PET_idso_corr"]
    else:
        return data["wisp_PET_idso_corr"]*((data['canop']/80+0.0833))

data["wisp_AET_idso_corr"]=data.apply(adj_AET, axis = 1)
########################################################################################################
back=data
df=data

paper_data=pd.concat((df["TIMESTAMP"], df["month"], df["AVP_pot"],df["LW_in"], df["LW_out"], df["Tair"], df ["pot_ET_inches"], df['canop'], df['prec_inch'], df['irrig_inches'],df['wisp_PET'], df['wisp_ET_adj'],
df['lwnet_idso'], df['idso_emis'], df['wisp_PET_idso'], df['wisp_AET_idso'], df['lwnet_idso_corr'], df['idso_emis_corr'],
df['wisp_PET_idso_corr'],df['wisp_AET_idso_corr'],df['pin_ET_inches'],df['prec_inch']),axis=1)

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
paper_data.to_csv('paper2_wisp.csv', index=False, header=True)


##########################################################################
## for paper that submitted

########################################################################
paper_data=pd.concat((df["TIMESTAMP"], df["month"], df["AVP_pot"],df["LW_in"], df["LW_out"], df["Tair"], df ["pot_ET_inches"], df['canop'], df['prec_inch'], df['irrig_inches'],df['wisp_PET'], df['wisp_ET_adj'],
df['lwnet_idso'], df['idso_emis'], df['wisp_PET_idso'], df['wisp_AET_idso'], df['lwnet_idso_corr'], df['idso_emis_corr'],
df['wisp_PET_idso_corr'],df['wisp_AET_idso_corr']),axis=1)

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
paper_data.to_csv('paper2_wisp.csv', index=False, header=True)



data=pd.read_csv('paper2_wisp.csv')

##############################################################################################################

## prior to 2022

df=data
df["pot_ET_inches"]= np.where(df["pot_ET_inches"]<0,0.00001,df["pot_ET_inches"])
df["pot_ET_EBC_inches"]= np.where(df["pot_ET_EBC_inches"]<0,0.00001,df["pot_ET_EBC_inches"])
df["wisp_ET_adj"]= np.where(df["wisp_ET_adj"]<0,0.00001,df["wisp_ET_adj"])
df["wisp_AET_idso"]= np.where(df["wisp_AET_idso"]<0,0.00001,df["wisp_AET_idso"])
df["wisp_AET_idso_corr"]= np.where(df["wisp_AET_idso_corr"]<0,0.00001,df["wisp_AET_idso_corr"])
#df["wisp_AET_idso_corr1"]= np.where(df["wisp_AET_idso_corr1"]<0,0.00001,df["wisp_AET_idso_corr1"])
df["LW_net"]=df["LW_in"]-df["LW_out"]
df.to_csv('wisp_fixed.csv', index=False, header=True)

data=df

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP")
data=pd.read_csv('growing_season.csv')

#os.chdir(r"C:\ammara_MD\flux_tower_data\potato")

df=data
df=pd.concat((df["TIMESTAMP"],df['LW_net'],df['lwnet_idso'],df['lwnet_idso_corr'],
            df["pot_ET_inches"],df["pot_ET_EBC_inches"],df["wisp_ET_adj"],
             df["wisp_AET_idso_corr"]),axis=1)

data.to_csv('corr_ET.csv', index=False, header=True)


df=df.iloc[:,1:99]  # don't worry about end column. not so important
df.isnull().values.any()
df=df.fillna(df.rolling(2,1).mean())
df.isnull().values.any()

df["TIMESTAMP"]=data["TIMESTAMP"]
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))




plt.plot(df["pot_ET_inches"], zorder=1)
plt.plot(df['wisp_AET_idso_corr'], zorder=2)


pearsonr(df["pot_ET_inches"],df['wisp_ET_adj'])
pearsonr(df["pot_ET_inches"],df['wisp_AET_idso_corr'])


pearsonr(df["pot_ET_EBC_inches"],df['wisp_ET_adj'])
pearsonr(df["pot_ET_EBC_inches"],df['wisp_AET_idso_corr'])


NS(df['wisp_ET_adj'],df["pot_ET_inches"])
NS(df['wisp_AET_idso_corr'],df["pot_ET_inches"])


NS(df['wisp_ET_adj'],df["pot_ET_EBC_inches"])
NS(df['wisp_AET_idso_corr'],df["pot_ET_EBC_inches"])


sqrt(mean_squared_error(df["pot_ET_inches"],df['wisp_ET_adj']))
sqrt(mean_squared_error(df["pot_ET_inches"],df['wisp_AET_idso_corr']))


pbias=(np.sum(df['wisp_ET_adj']-df["pot_ET_inches"])/np.sum(df["pot_ET_inches"]))*100
pbias

pbias=(np.sum(df['wisp_AET_idso_corr']-df["pot_ET_inches"])/np.sum(df["pot_ET_inches"]))*100
pbias


pbias=(np.sum(df['wisp_ET_adj']-df["pot_ET_EBC_inches"])/np.sum(df["pot_ET_EBC_inches"]))*100
pbias

pbias=(np.sum(df['wisp_AET_idso_corr']-df["pot_ET_EBC_inches"])/np.sum(df["pot_ET_EBC_inches"]))*100
pbias


scipy.stats.pearsonr(df["pot_ET_inches"],df['wisp_ET_adj'])

r2_score(df["pot_ET_inches"],df['wisp_ET_adj'])
r2_score(df["pot_ET_inches"],df['wisp_AET_idso_corr'])

r2_score(df["pot_ET_EBC_inches"],df['wisp_ET_adj'])
r2_score(df["pot_ET_EBC_inches"],df['wisp_AET_idso_corr'])


plt.plot(df["pot_ET_EBC_inches"], zorder=1)
plt.plot(df['wisp_AET_idso_corr'], zorder=2)


#############################################################################################





def lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat):
    return angstrom(avg_v_press,avg_temp)*lwu(avg_temp)*clr_ratio(d_to_sol,day_of_year,lat)

cee["lw_net_idso_corr"]=-lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)*11.574

cee=df
       
### LAI is in next page
cee["LAI"]=df["LAI"]    
cee["inso_temp_c"]=avg_temp
cee["max_temp_c"]=df["max_temp_c"]
cee["min_temp_c"]=df["min_temp_c"]
cee["LST"]=df["col40_pota"]-273.15
cee["PRCP_7"]=df["PRCP_7"]
cee["PRCP_14"]=df["PRCP_14"]
cee["PRCP_30"]=df["PRCP_30"]
cee["PRCP"]=df["PRCP"]
cee["PRCP_3"]=df["PRCP_3"]
cee["PRCP_5"]=df["PRCP_5"]

plt.plot(cee["PRCP_5"])
plt.plot(cee["PRCP_7"])


import statsmodels.api as sm
X=data["Potato_ET_inches"]
X = sm.add_constant(X)
Y = data["wisp_ET_adj"]+(a*(data["LAI"]*data["LST"])**b)
model = sm.OLS(Y,X)
results = model.fit()
results.params
 #intercept  0.495
 #     

x=data["PRCP_3"] 

x= data["Potato_ET_inches"]  

results.summary()       
       
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


    
def adj_AET (cee):
    if cee['canop']>=80:
        return cee["wisp_PET_idso_corr"]
    else:
        return cee["wisp_PET_idso_corr"]*((cee['canop']/80+0.0833))
    
cee["corrc_idso_AET"]= cee.apply(adj_AET, axis = 1)


def adj_soil (cee):
    if cee['canop']>=80:
        return cee["wisp_PET_idso_corr"]
    else:
        return (((0.2*(cee["LAI"]*cee["LST"])**0.2)*cee["wisp_PET_idso_corr"])+0.03*(cee["PRCP"])+0.035*(cee["PRCP"])+cee["wisp_PET_idso_corr"]*(cee['canop']/100))




cee["corridso_soil_AET"]= cee.apply(adj_soil, axis = 1)

    
#+0.0833
#data["corrc_idso_AET"]= data.apply(adj_AET, axis = 1)


x1=cee["Potato_ET_inches"]*25.4
y1=cee["corridso_soil_AET"]*25.4
y2=cee["corrc_idso_AET"]*25.4


plt.plot(x1,'g-')
plt.plot(y1,'r-')
plt.plot(y2,'b-')

pbias=(np.sum(y1-x1)/np.sum(x1))*100
pbias



rms = sqrt(mean_squared_error(x1, y1)) # rmse 20.5, r2 0.948
rms

rms = sqrt(mean_squared_error(x1, y2)) # rmse 20.5, r2 0.948
rms

pbias=(np.sum(y2-x1)/np.sum(x1))*100
np.mean(pbias)






soil_ET=0.1*(LAI*avg_temp)**0.02

LAI=df["LAI"]

y2=wisp_adj_code


x1=data["Potato_ET_inches"]
y1=data["wisp_ET_adj"]
y2=wisp_adj_code
plt.scatter(x1,y1)
plt.scatter(x1,y2)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.score(x1, y1)

can_x=(data["Potato_ET_inches"]/data["wisp_PET_cod"])
can_x=(data["Potato_ET_inches"]/wisp_AET)

canop_y=data["can_cov"]
y1=wisp_adj

plt.plot(can_x)

##############################################################################################

### apply correction
data["week"]=pd.to_datetime(df['TIMESTAMP']).dt.week

data["year"]=pd.to_datetime(df['TIMESTAMP']).dt.year



def corr_fac (data):
    ### for 2018
    if (data['week']>=26) and (data['week']<=28) and (data['year']==2018):
        return 0.76
    
    elif (data['week']>=29) and (data['week']<=31) and (data['year']==2018):
        return 0.67
    
    elif (data['week']>=32) and (data['week']<=34) and (data['year']==2018):
        return 0.48
    elif (data['week']>=35) and (data['week']<=37) and (data['year']==2018):
        return 0.45
    elif (data['week']>=38) and (data['week']<=40) and (data['year']==2018):
        return 0.85
### 2019    
    elif (data['week']>=18) and (data['week']<=20) and (data['year']==2019):
        return 0.44
    elif (data['week']>=21) and (data['week']<=23) and (data['year']==2019):
        return 0.51
    elif (data['week']>=24) and (data['week']<=25) and (data['year']==2019):
        return 0.67
    elif (data['week']>=26) and (data['week']<=28) and (data['year']==2019):
        return 0.56
    elif (data['week']>=29) and (data['week']<=31) and (data['year']==2019):
        return 0.63
    elif (data['week']>=32) and (data['week']<=34) and (data['year']==2019):
        return 0.60
    elif (data['week']>=35) and (data['week']<=37) and (data['year']==2019):
        return 0.59
    elif (data['week']>=38) and (data['week']<=40) and (data['year']==2019):
        return 0.56    
    else:
        return 0.56
cor_fact=data.apply(corr_fac, axis = 1)
################################################################################################

data["AET_corr_f"]=data["wisp_PET_cod"]*cor_fact

x1=data["Potato_ET_inches"]
y1=data["AET_corr_f"]
plt.scatter(x1,y1)

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)

plt.scatter(x1,y1)

reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_
reg.score(x1, y1)








#########################################################################################################

lw_net_pot=data.netLW_pot
lw_in_pot=data.LWin_pot
lw_out_pot=data.LWout_pot
rnet_pot=data.pot_Rnet

#plt.plot(data.netLW_pot,'r')
plt.plot(lw_in_pot,'r')
plt.plot(lw_out_pot,'b')  # lw out is higher than lw in
plt.plot(lw_net_pot,'black')
plt.plot(lw_in_pot-lw_out_pot,'b')

#####################################################################################################
### run parts by parts
def rnet(avg_temp,avg_v_press,d_to_sol,day_of_year,lat):
    ## calculate lwnet
    lwnnet=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)       
   ## calculate R_n 
    return (1-ALBEDO)*d_to_sol-lwnnet


rnet_wisp=rnet(avg_temp,avg_v_press,d_to_sol,day_of_year,lat)*11.574

lw_net_wisp=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)*11.574
##########################################################################################################
def et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat):

    ## calculate lwnet
    lwnnet=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)
       
   ## calculate R_n 
    net_radiation=(1-ALBEDO)*d_to_sol-lwnnet
    ## formula for evapotranspiration
    ret1=1.28*sfactor(avg_temp)*net_radiation
    # assume 62.3 is the conversion factor but unable to determine 
    return ret1/62.3

wisp_PET=et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat)
data["wisp_PET"]=wisp_PET
#######################################################################################################
########################################################################################################


########################################################################################################
#x1=emiss["Potato_ET_inches"]
#y1=emiss["wisp_PET"]
#y2=emiss["ET_obs_emiss"]
x1=df["netLW_pot"]
y1=df["lw_net_corr_emiss"]  ### this is actually longwave net
y2=df["lw_net_wisp"]

x1=df["netLW_pot"]
y1=df["lw_net_corr_emiss"]  ### this is actually longwave net
y2=df["lw_net_wisp"]


x1=x1
y1=y1
y1=y2


x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_

reg.coef_
#reg.intercept_
reg.score(x1, y1)
pred=(y1/reg.coef_)
########################################################################################################
## PET graphs
import matplotlib.lines as mlines
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="indigo",markersize=2,label="lWnet")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
#ax.plot(x1,a*x1,'r-')
ax.plot(x1,a*x1,'r-', label='Slope=0.94,R2=0.87')
#ax.plot(x1,pred,'x',color="black",markersize=2, label='R2=0.53')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper right',prop={'size': 6})
ax.set_ylabel('wisp LWnet')  # we already handled the x-label with ax1
ax.set_xlabel('Observed LW net (wm-2)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(10,-150)
ax.set_xlim(10,-150)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#### check new coeffificent 
reg = LinearRegression(fit_intercept=False).fit(x1, y1/a)
reg.score(x1, y1/a)
b=reg.coef_
b
###########################################################################################################
x1=data["Potato_ET_inches"]
y1=data["corrc_emiss_ET"]
y2=df["wisp_ET_adj"]


pbias=(np.sum(x1-y2)/np.sum(x1))*100
pbias


x1=x1
y1=y1
y1=y2


x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=False).fit(x1, y1)
a=reg.coef_

reg.coef_
#reg.intercept_
reg.score(x1, y1)
pred=(y1/reg.coef_)

import matplotlib.lines as mlines
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="indigo",markersize=2,label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
#ax.plot(x1,a*x1,'r-')
ax.plot(x1,a*x1,'r-', label='Slope=1.38,R2=0.74')
#ax.plot(x1,pred,'x',color="black",markersize=2, label='R2=0.53')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper right',prop={'size': 6})
ax.set_ylabel('model AET')  # we already handled the x-label with ax1
ax.set_xlabel('obs AET')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,0.3)
ax.set_xlim(0,0.3)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#### check new coeffificent 
reg = LinearRegression(fit_intercept=False).fit(x1, y1/a)
reg.score(x1, y1/a)
b=reg.coef_
b

#############################################################################################
data=pd.read_csv('growing_season.csv')
#df=data
pot_temp_c=(emiss["airtempF_pot"]-32 )*(5/9)
 
AVP_pot=emiss["AVP_pot"]
   
#def sky_emiss_pot(AVP_pot,pot_temp_c):
   # if(AVP_pot>0.5).any():
    #    return 0.7+(5.95e-4)*avg_v_press*np.exp(1500/(273+pot_temp_c))
    #else:
     #   return (1-0.261*np.exp(-0.000777*pot_temp_c*pot_temp_c))    

#emiss["tower_em"]=sky_emiss_pot(AVP_pot,pot_temp_c) 


def sky_emiss(avg_v_press,avg_temp):
    if(avg_v_press>0.5).any():
        return 0.7+(5.95e-4)*avg_v_press*np.exp(1500/(273+avg_temp))
    else:
        return (1-0.261*np.exp(-0.000777*avg_temp*avg_temp))    


emiss["corr_emiss"]=sky_emiss(avg_v_press,avg_temp)
x=es_obs
y=emiss["corr_emiss"]



a=0.7
b=5.95e-4
c=1500
p=a+(b)*avg_v_press*np.exp(c/(273+pot_temp_c)



from scipy.optimize import fsolve
import math

def equations(p):
    a, b,c = p
    return (x+y**2-4, math.exp(x) + x*y - 3)

x, y =  fsolve(equations, (1, 1))

print equations((x, y))
#########################################################################################

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = AVP_pot
xdata = emiss["es_obs"]
ydata = pot_temp_c
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');



from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = xdata
sequence_containing_y_vals = ydata
sequence_containing_z_vals = zdata


ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
pyplot.show()
