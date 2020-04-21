def attractivity():
    
    """
    Attractivity modelling 
    """
    
    #Importing packages
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    
    from scipy import stats
    from sklearn.preprocessing import minmax_scale
    import os
    
    
    #Importing shape files--------------------------------------------------------
    sheff_lsoa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))
    #sheff_oa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_oa11.shp"))
    
    
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
    
    
    
    
    
    
    
    
    
    
    s = int( 1*  len(sheff_lsoa_shape)) #samples 
    idx = np.round(np.linspace(0, len(sheff_lsoa_shape) - 1, s)).astype(int) #Indexing s spaced from array
    
    #Creating empirical income distributions -------------------------------------
    """
    Counts used are those we have data for in terms of income. The same counts are then used in education sampling. 
    """
    bounds = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 150000] #income bounds
    bounds = minmax_scale(bounds)
    
    income_dists=[]
    counts=[]
    edu_counts = []
    edu_ratios = []
    b_params = np.zeros((len(idx), 4))
    education=[]
    
    
    for i in range(len(idx)):
        
        #Income
        count = sheff_lsoa_pop['KS101EW0001'].values[idx[i]]
        count = count * sheff_lsoa_income.iloc[idx[i]]['rank_1':'rank_9'].values
        count = count.astype(int)
        
        x = []
        for j in range(len(count)):
            if j == 0:
                x.append(np.random.uniform(low = bounds[j], high = bounds[j], size = count[j]))
            else:
                x.append(np.random.uniform(low = bounds[j], high = bounds[j + 1], size = count[j]))
        x = np.concatenate(x, axis = 0)
        counts.append(sum(count)) #Create list of counts to use later
               
        b_params[i, 0], b_params[i, 1], b_params[i, 2], b_params[i, 3] = stats.beta.fit(x, loc = 0, scale = 1) #Fitting beta functions       
        #income_dists.append(stats.beta.rvs(b_params[i, 0], b_params[i, 1], loc = b_params[i, 2], scale = b_params[i, 3], size=sheff_lsoa_pop['KS101EW0001'].values[idx[i]].astype(int)))  #Generating Income dists
        
        
        
        
        #Education ---------------------------------------------------------------
        edu_counts.append(sheff_lsoa_pop['KS101EW0001'].values[idx[i]].astype(int))
        #count = sheff_lsoa_pop['KS101EW0001'].values[idx[i]].astype(int)
        edu = sheff_lsoa_education.iloc[idx[i]][[2,3,4,5,6,7]].values.astype(int)
        levels = [edu[0], edu[1]+edu[2], edu[3] + edu[4], edu[5]]
        edu = sum(sheff_lsoa_education.iloc[idx[i]][[2,3,4,5,6,7]].values.astype(int))
        edu_ratios.append(np.divide(levels,edu))
        
        #education.append(np.random.choice(4, size = count, p=ratios)) #where p values are effectively the ratio of people with a given education level you can alternatively use the same method for income as well
        
    
        #Attractivity ------------------------------------------------------------
        #attractivity.append(np.power(income_dists[i], -education[i]))
    
    return b_params, edu_counts, edu_ratios
  
