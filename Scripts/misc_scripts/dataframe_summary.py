# -*- coding: utf-8 -*-
"""
Creating dataframes
"""

import pandas as pd
import geopandas as gpd
import pickle
import scipy.io as sio 
from scipy import stats
import numpy as np

def save_obj(obj, name ):
    with open('summary/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
    
def paths_mat(layout):
    paths = layout
    paths = np.reshape(paths, (len(paths)))
    paths_matrix = np.zeros((345,345))
    lowInds = np.tril_indices(345,k=-1)
    highInds  = np.triu_indices(345, k=1)
    paths_matrix[lowInds] = paths
    paths_matrix = paths_matrix + paths_matrix.T - np.diag(paths_matrix) 
    return paths_matrix




lsoa_data = load_obj("lsoa_data")

normal = load_obj("normal_layout_2000run")
minmax = load_obj("minmax_layout_2000run")
slmin = load_obj("slmin_layout_2000run")
od50 = load_obj("od50_layout_2000run")


centroid_info = load_obj("centroids_beta_params_centroiddists")
centroid_paths_matrix = np.vstack(centroid_info['euclidean_dists'])

mldata = sio.loadmat(r'C:\Users\cip18jjp\Dropbox\PIN\hadi_scripts\optimisedpaths.mat')#import new paths
normal_paths_matrix = load_obj("ave_paths")
minmax_paths_matrix = paths_mat(mldata['minmax'])
slmin_paths_matrix = paths_mat(mldata['slmin'])
od50_paths_matrix = paths_mat(mldata['od50'])    
  
  
opt_data = load_obj("opt_data_2000run_lsoa")
a1a2 = paths_mat(opt_data['a1a2'])
pop = np.asarray(lsoa_data['edu_counts']).reshape((len(lsoa_data['edu_counts']), 1))
pop = np.matmul(pop,pop.T)
#np.fill_diagonal(pop,0)
    
income_means = []
for i,j,k,l in lsoa_data['income_params']:
    income_means.append(stats.beta.mean(i,j,k,l))
income_means = np.asarray(income_means) 
    
education_means = []
for i in range(len(lsoa_data['edu_counts'])):
    education_means.append(np.random.choice(4, size = lsoa_data['edu_counts'][i], p=lsoa_data['edu_ratios'][i]).mean())    
education_means = np.asarray(education_means)     
    

lsoas = lsoa_data['sheff_lsoa_shape']['LSOA11CD'].values.reshape((len(lsoa_data['sheff_lsoa_shape']), 1))    
lsoa1 = np.repeat(lsoas, len(lsoa_data['sheff_lsoa_shape']), axis=1)
lsoa2 = np.repeat(lsoas.T, len(lsoa_data['sheff_lsoa_shape']), axis=0)


LSOAdata = pd.DataFrame(data = {'Code':lsoa_data['sheff_lsoa_shape']['LSOA11CD'],
                                'Mean income':income_means,
                                'Mean education':education_means,
                                'Population':lsoa_data['edu_counts'],
                                'Population density':lsoa_data['edu_counts']/lsoa_data['sheff_lsoa_shape'].area
                                })

def createDF(dataset, opt_paths):
    OPTdata = pd.DataFrame(data = {'LSOA1':lsoa1.flatten(),
                                   'LSOA2':lsoa2.flatten(),
                                   'Frequency m=0.5':dataset['edge_freqs'][0].flatten(),
                                   'Frequency m=1':dataset['edge_freqs'][1].flatten(),
                                   'Frequency m=1.5':dataset['edge_freqs'][2].flatten(),
                                   'Frequency m=2':dataset['edge_freqs'][3].flatten(),                                  
                                   'Original paths':normal_paths_matrix.flatten(),
                                   'Centroid paths':centroid_paths_matrix.flatten(),
                                   'Optimized paths':opt_paths.flatten(),
                                   'Attractivity product':a1a2.flatten(),
                                   'Population product':pop.flatten()
                                    })
    return OPTdata
normal = createDF(normal, normal_paths_matrix)
minmax = createDF(minmax, minmax_paths_matrix)
slmin = createDF(slmin, slmin_paths_matrix)
od50 = createDF(od50, od50_paths_matrix)


data = {'LSOAdata':LSOAdata,
        'normal':normal,
        'minmax':minmax,
        'slmin':slmin,
        'od50':od50}

save_obj(data, "2000_run_monte_carlo")