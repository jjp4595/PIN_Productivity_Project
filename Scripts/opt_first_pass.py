"""
Optimisation attempt
"""


import numpy as np
import pickle
from scipy.optimize import minimize, linprog
import math



def load_obj(name ):
    with open('obj/'+ name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#Data Generation -------------------------------------------------------------
# #1a) Directly run monte carlo
from synthetic_network import sample_attractivities
lsoa_data = load_obj("lsoa_data")
pop = np.asarray(lsoa_data['edu_counts']).reshape((len(lsoa_data['edu_counts']), 1))
pop = pop + pop.T
pop = np.tril(pop) #attractivity product
np.fill_diagonal(pop,0) #removing a1a1 scenario as already assumed connection
pop = np.ndarray.flatten(pop)
pop = pop[pop !=0]


# a1_dist, a2_dist = [], []
# for i in range(2000): #Sample run for attractivities
#     attractivity1, attractivity2 = sample_attractivities(lsoa_data['edu_ratios'], lsoa_data['income_params'])
#     a1_dist.append(attractivity1)
#     a2_dist.append(attractivity2)


# a1 = np.stack(a1_dist, axis=0).mean(0)
# a2 = np.stack(a2_dist, axis=0).mean(0)
# a1 = a1.reshape((len(a1),1))
# a2 = a2.reshape((len(a2),1))
# a1a2 = np.tril(np.multiply(a1, np.transpose(a2))) #attractivity product
# np.fill_diagonal(a1a2,0) #removing a1a1 scenario as already assumed connection
# a1a2 = np.ndarray.flatten(a1a2)
# a1a2 = a1a2[a1a2 !=0]

# opt_data = {'a1a2':a1a2,
#             'pop':pop}

# save_obj(opt_data,"opt_data_2000run_lsoa")
#-----------------------------------------------------------------------------
#2) load premade data
opt_data = load_obj("opt_data_2000run_lsoa")
paths_matrix = load_obj("ave_paths")
paths = np.tril(paths_matrix) #lower half of od matrix
paths = np.ndarray.flatten(paths)
paths = paths[paths !=0]
#-----------------------------------------------------------------------------


#Scipy Minimize ---------------------------------------------------------------
# def f(paths, a1a2, pop, original_path):
#     out = np.divide(a1a2, paths)
#     out = np.multiply(out, pop)
#     return out.sum()*0.5*-1

# def summing_paths(x, a1a2, pop, original_path):   
#     return np.sum(x) - np.sum(original_path)
# def min_path(x, a1a2, pop, original_path):     
#     return np.min(x) - np.min(original_path) 
# def max_path(x, a1a2, pop, original_path):   
#     return (np.max(x) - np.max(original_path)) *-1 #to make constraint non-negative

# n = 250

# arguments = (opt_data['a1a2'][0:n], opt_data['pop'][0:n], paths[0:n])

# con1 = {'type':'eq', 'fun':summing_paths, 'args':arguments}
# con2 = {'type':'ineq', 'fun':min_path, 'args':arguments}
# con3 = {'type':'ineq', 'fun':max_path, 'args':arguments}
# cons = [con1, con2, con3]

# bnds = [(math.floor(np.min(paths[0:n])),math.ceil(np.max(paths[0:n])))]
# bnds = bnds * n

# res = minimize(f, x0=paths[0:n], args=arguments, constraints = cons, bounds = bnds, method = 'trust-constr', options = {'disp':True, 'verbose':2})
# sci_min = res['x']
#------------------------------------------------------------------------------

# n=20
# c = np.multiply(opt_data['a1a2'][0:n],opt_data['pop'][0:n])

# scale = 100
# bnds = [((1*scale)/math.ceil(np.max(paths[0:n])) , 1*(scale)/math.floor(np.min(paths[0:n])))]
# bnds = bnds * n

# res = linprog(-c, A_eq = np.ones((1,len(c))), b_eq = (1/np.sum(paths[0:n])), bounds=bnds)

#-------
def linprog_optmodel(n):
    c = 1/np.multiply(opt_data['a1a2'][0:n],opt_data['pop'][0:n])
    bnds = [(math.floor(np.min(paths[0:n])),math.ceil(np.max(paths[0:n])))]
    bnds = bnds * n
    Aub = np.full((1,n),1/len(paths[0:n]))
    res = linprog(c, A_eq = np.ones((1,len(c))), b_eq = (np.sum(paths[0:n])), A_ub= Aub, b_ub = np.mean(paths[0:n]), bounds=bnds)
    return res['x']
import matplotlib.pyplot as plt


# ns = [100, 500, 1000, 2000, 2500, 3000, 3500, 4000]
# fig, axs = plt.subplots(2,int(len(ns)/2))
# axs = axs.ravel()
# for i in range(len(ns)):
#     new = paths[0:ns[i]]
#     axs[i].hist(new, alpha=0.3)
#     axs[i].hist(linprog_optmodel(ns[i]), alpha = 0.4)
#     axs[i].set_title(str(ns[i]) + 'runs')
# plt.tight_layout()

