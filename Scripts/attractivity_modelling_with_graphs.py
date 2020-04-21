"""
Attractivity modelling 
"""
#Importing packages
#import osmnx as ox
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from sklearn.preprocessing import minmax_scale
import powerlaw
import os


#Importing shape files--------------------------------------------------------
sheff_OShighways = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_OA.shp"))
sheff = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_Network.gdb"))
sheff_lsoa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))
sheff_msoa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_msoa11.shp"))


#Import Population data
sheff_lsoa_pop = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\tables\KS101EW_lsoa11.csv"))
sheff_lsoa_pop['KS101EW0001'] = pd.to_numeric(sheff_lsoa_pop['KS101EW0001']) #Count: All categories:sex

#import income
sheff_lsoa_income = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Individual LSOA Income Estimate\E37000040\spatial\E37000040.shp"))
#Import education
sheff_lsoa_education = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\tables\KS501EW_lsoa11.csv"))


#Remove lsoa's that aren't in census data so income & pop are same size
ids = sheff_lsoa_income['lsoa11cd'].isin(sheff_lsoa_pop['GeographyCode'].values)
ids = np.where(ids==True)
sheff_lsoa_income = sheff_lsoa_income.iloc[ids]


#Indexing 25 evenly spaced from array
#idx = np.round(np.linspace(0, len(sheff_lsoa_income) - 1, 25)).astype(int)

s = 50 #samples 
idx = np.round(np.linspace(0, len(sheff_lsoa_shape) - 1, s)).astype(int) #Indexing s spaced from array

#Creating empirical income distributions -------------------------------------
"""
Counts used are those we have data for in terms of income. The same counts are then used in education sampling. 
"""
bounds = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 150000] #income bounds
bounds = minmax_scale(bounds)

income_dists=[]
counts=[]

b_params = np.zeros((len(idx), 4))
bin_no = 20


#fig0 = plt.figure(figsize=(18,16))
#fig0.suptitle('LSOA income distribution sample', fontsize=16)


for i in range(len(idx)):
    count = sheff_lsoa_pop['KS101EW0001'].values[idx[i]]
    count = count * sheff_lsoa_income.iloc[idx[i]]['rank_1':'rank_9'].values
    count = count.astype(int)
    #ax = fig0.add_subplot(5,5,i+1)
    x = []
    for j in range(len(count)):
        if j == 0:
            x.append(np.random.uniform(low = bounds[j], high = bounds[j], size = count[j]))
        else:
            x.append(np.random.uniform(low = bounds[j], high = bounds[j + 1], size = count[j]))
    x = np.concatenate(x, axis = 0)
    
    #Create list of counts to use later
    counts.append(sum(count))
       
    #Fitting beta functions
    b_params[i, 0], b_params[i, 1], b_params[i, 2], b_params[i, 3] = stats.beta.fit(x, loc = 0, scale = 1)
    #ax.plot(np.linspace(0,1,100),  stats.beta.pdf(np.linspace(0,1,100), b_params[i, 0], b_params[i, 1], loc = b_params[i, 2], scale = b_params[i, 3]) , label = 'CDF')
    
    #Generating Income dists
    income_dists.append(stats.beta.rvs(b_params[i, 0], b_params[i, 1], loc = b_params[i, 2], scale = b_params[i, 3], size=sheff_lsoa_pop['KS101EW0001'].values[idx[i]].astype(int)))
    
    # #Generating plots
    # ax.hist(x, bins = bin_no, density = True, label = 'Uniform generated')
    # ax.hist(income_dists[i], bins = bin_no, density = True, label = 'Beta-drawn')
    # ax.legend(loc='upper right')
    # ax.set_ylabel("Count")
    # ax.set_xlabel("Income")
    # ax.set_xlim(0,1)
    # plt.tight_layout()
    
#fig0.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Temp\LSOA_income_distributions.pdf"), format = 'pdf')    
 


   
#Education Distribution ------------------------------------------------------
"""
Find in the atlas.
"""


education=[]
#Indexing 25 evenly spaced from array
#idx = np.round(np.linspace(0, len(sheff_lsoa_education) - 1, 25)).astype(int)

# fig1 = plt.figure(figsize=(18,16))
# fig1.suptitle('LSOA Education distribution sample', fontsize=16)

for i in range(len(idx)):
    count = sheff_lsoa_pop['KS101EW0001'].values[idx[i]].astype(int)
    edu = sheff_lsoa_education.iloc[idx[i]][[2,3,4,5,6,7]].values.astype(int)
    levels = [edu[0], edu[1]+edu[2], edu[3] + edu[4], edu[5]]
    edu = sum(sheff_lsoa_education.iloc[idx[i]][[2,3,4,5,6,7]].values.astype(int))
    ratios = np.divide(levels,edu)
    
    education.append(np.random.choice(4, size = count, p=ratios)) #where p values are effectively the ratio of people with a given education level you can alternatively use the same method for income as well
    
    # ax = fig1.add_subplot(5,5,i+1)
    # ax.hist(education[i])
    # ax.set_ylabel(r'$count(education)$')
    # ax.set_xlabel(r'$education$')  
    # plt.tight_layout()
#fig1.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Temp\LSOA_education_distributions.pdf"), format = 'pdf')  



    
#Attractivity ----------------------------------------------------------------
attractivity = []


# fig2 = plt.figure(figsize=(18,16))
# fig2.suptitle('LSOA Attractivity distribution sample', fontsize=16)


for i in range(len(idx)):
    attractivity.append(np.power(income_dists[i], -education[i]))
    
    ax = fig2.add_subplot(5,5,i+1)
    # ax.hist(attractivity[i], bins=10000, density=True, histtype='step')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylabel(r'$density(attractivity)$')
    # ax.set_xlabel(r'$attractivity$')
    # plt.tight_layout()
#fig2.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Temp\LSOA_attractivity_distributions.pdf"), format = 'pdf') 


    
#Power law distributions -----------------------------------------------------
# fig3 = plt.figure(figsize=(18,16))
# fig3.suptitle('LSOA power law distribution sample', fontsize=16)


for i in range(len(idx)):
    
    attractivtiy_powerlaw = powerlaw.Fit(attractivity[i])
    attractivtiy_powerlaw_temp = powerlaw.Fit(attractivity[i], xmin=1)
    
    ax = fig3.add_subplot(5,5,i+1)
    
    powerlaw_plot=attractivtiy_powerlaw_temp.plot_ccdf(original_data=True, ax = ax, color='k', marker='.', lw=0, alpha=0.5)
    
    X=attractivtiy_powerlaw_temp.ccdf()
    x=X[0][X[0]>=attractivtiy_powerlaw.xmin]
    y=[np.exp( (-(attractivtiy_powerlaw.alpha-1)*(np.log(k)-np.log(attractivtiy_powerlaw.xmin))+np.log(X[1][X[0]==attractivtiy_powerlaw.xmin])) ) for k in x]
    
   
    ax.plot(x, y, 'r')
    ax.set_ylabel(r'$P(attractivity \geq x)$')
    ax.set_xlabel(r'$attractivity$')
    plt.tight_layout()

#fig3.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Temp\LSOA_powerlaw_distributions.pdf"), format = 'pdf') 