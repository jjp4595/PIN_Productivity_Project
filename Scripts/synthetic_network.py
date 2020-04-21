"""
Synthetic Sheffield
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os 
from scipy import stats
import scipy.optimize
from pyproj import CRS
import osmnx as ox
import networkx as nx
import time
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.regression.linear_model import OLS

import attractivity_modelling
import path_querying
import fractal_working

#Importing shape files--------------------------------------------------------
sheff_OShighways = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_OA.shp"))
sheff = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_Network.gdb"))
sheff_lsoa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))
sheff_oa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_oa11.shp"))
sheff_shape = sheff_lsoa_shape



income_params, edu_counts, edu_ratios = attractivity_modelling.attractivity()


def attractivity_sampler(oa):
    """
    Parameters
    ----------
    oa : Integer of oa

    Returns
    -------
    attractivity 

    """
    
    edu = np.random.choice(4, size = 1, p=edu_ratios[oa]) #where p values are effectively the ratio of people with a given education level
    income = stats.beta.rvs(income_params[oa, 0], income_params[oa, 1], loc = income_params[oa, 2], scale = income_params[oa, 3], size=1)
    
    attractivity = np.power(income, -edu)
    
    return attractivity


#Loading Graph ---------------------------------------------------------------
#Getting drivable streets in sheffield
place_name = "Sheffield, UK"
graph = ox.graph_from_place(place_name, network_type = 'drive')
#graph = ox.graph_from_file() # create graph from OSM data in XML file
nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True) #Convert graph int geodataframe
crs = CRS.from_string("epsg:27700")#convert data into same co-ordinate system as lsoa data
graph_proj = ox.project_graph(graph, to_crs=crs)
nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True) #Convert graph into UTM zone 30 format (with m units)







#Setting indices -------------------------------------------------------------
s = 0.1
s = int(s*  len(sheff_lsoa_shape)) 
idx = np.round(np.linspace(0, len(sheff_lsoa_shape) - 1, s)).astype(int) #Indexing s spaced from array


#Sampling coordinates and attractivities --------------------------------------
attractivity1 = np.zeros((s))
attractivity2 = np.zeros((s))
point1s = []
point2s = []
origin=[]
target=[]
for i in range(len(idx)): #Loop across  OAs    
    point1s.append(path_querying.get_random_point_in_polygon(sheff_lsoa_shape['geometry'][idx[i]]))
    attractivity1[i] = attractivity_sampler(i)
    origin.append((point1s[i].x, point1s[i].y))
    
    point2s.append(path_querying.get_random_point_in_polygon(sheff_lsoa_shape['geometry'][idx[i]]))                     
    attractivity2[i] = attractivity_sampler(i)
    target.append((point2s[i].x, point2s[i].y))
all_samples = origin + target
all_samples = pd.DataFrame(all_samples, columns = ['x-coord', 'y-coord'])
all_attractivity = np.concatenate((attractivity1, attractivity1) , axis=0)





#Calculating shortest paths --------------------------------------------------
paths_matrix = np.zeros((s, s))
t = []
for i in range(len(idx)): 
    t.append(time.time())
    for j in range(len(idx)): #Loop across target OAs to create 
        if i==j:
            paths_matrix[i,j] = 0
        else:
            paths_matrix[i,j] = path_querying.shortestpath_oa(point1s[i],point2s[j], graph_proj)  
    t[i] = time.time() - t[i]
    print(t[i])
    
       
#median path distances     
paths = np.concatenate(paths_matrix, axis=0)
paths = paths[paths != 0]
paths = paths[paths != 1e8]
med_paths = sorted(paths)
med_paths = med_paths[int(len(med_paths)/2)]






#Fractal Dimension, D ---------------------------------------------------------
"""
Graph may require some intuition to fit the linear regression through certain points
"""
rangex = all_samples['x-coord'].values.max() - all_samples['x-coord'].values.min()
rangey = all_samples['y-coord'].values.max() - all_samples['y-coord'].values.min()
L = int(max(rangex, rangey)) # max of x or y distance

r = np.array([ L/(2.0**i) for i in range(5,0,-1) ]) #Create array of box lengths
N = [ fractal_working.count_boxes( all_samples, ri, L ) for ri in r ] #Non empty boxes for each array of box lenghts
 
popt, pcov = scipy.optimize.curve_fit( fractal_working.f_temp, np.log( 1./r ), np.log( N ) )
A, Df = popt #A lacunarity, Df fractal dimension


fig, ax = plt.subplots(1,1)
ax.plot(1./r, N, 'b.-')
ax.plot( 1./r, np.exp(A)*1./r**Df, 'g', alpha=1.0 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect(1)
ax.set_xlabel('Box Size')
ax.set_ylabel('Number of boxes')

#Playing around with data points to use
Y = np.log( N )
X = np.log( 1./r )
T = np.vstack((Y,X,np.ones_like(X))).T
 
df = pd.DataFrame( T, columns=['N(r)','Df','A'] )
Y = df['N(r)']
X = df[['Df','A']]
result = OLS( Y, X ).fit()
result.summary()




#-----------------------------------------------------------------------------
m = 4


eps = med_paths
theta = np.exp(np.log(xmin) - (m*np.log(eps)))

connectivity = np.divide(np.multiply(attractivity1, attractivity2), np.power(paths_matrix, m))
connectivity[np.where(np.isinf(connectivity))[0], np.where(np.isinf(connectivity))[1]] = 0 

adjacency = np.zeros_like(connectivity)
adjacency[np.where(connectivity>theta)] = 1

