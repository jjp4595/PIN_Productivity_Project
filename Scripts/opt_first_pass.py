"""
Optimisation attempt
"""


import numpy as np
import pickle
from scipy.optimize import minimize




def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# #1a) Directly run monte carlo
#from synthetic_network import sample_attractivities
# lsoa_data = load_obj("lsoa_data")
# pop = np.asarray(lsoa_data['edu_counts']).reshape((len(lsoa_data['edu_counts']), 1))
# pop = np.multiply(pop, np.transpose(pop))

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


# pop = np.tril(pop) #attractivity product
# np.fill_diagonal(pop,0) #removing a1a1 scenario as already assumed connection
# pop = np.ndarray.flatten(pop)
# pop = pop[pop !=0]


# opt_data = {'a1a2':a1a2,
#             'pop':pop}

# save_obj(opt_data,"opt_data_2000run_lsoa")

#2) load premade
opt_data = load_obj("opt_data_2000run_lsoa")

#-----------------------------------------------------------------------------



#Load paths ------------------------------------------------------------------

paths_matrix = load_obj("ave_paths")
paths = np.tril(paths_matrix) #lower half of od matrix
paths = np.ndarray.flatten(paths)
paths = paths[paths !=0]


def f(paths, a1a2, pop):
    out = np.divide(a1a2, paths)
    out = np.multiply(out, pop)
    return out.sum()*0.5*-1


def summing_paths(x):   
    return np.sum(x) - 371254.41549999994
def min_path(x):     
    return np.min(x) - 474.30150000000003 
def max_path(x):   
    return (np.max(x) - 10059.118499999999) *-1 #to make constraint non-negative


n = 100
sumn = np.sum(paths[0:n])
sum_min = np.min(paths[0:n])
sum_max = np.max(paths[0:n])

arguments = (opt_data['a1a2'][0:n], opt_data['pop'][0:n])

con1 = {'type':'eq', 'fun':summing_paths}
con2 = {'type':'ineq', 'fun':min_path}
con3 = {'type':'ineq', 'fun':max_path}
cons = [con1, con2, con3]

bnds = [(0,np.inf)]
bnds = bnds * n

res = minimize(f, x0=paths[0:n], args=arguments, constraints = cons, bounds = bnds, method='trust-constr', options = {'disp':True})
opt_edges= res['x']





# paths_splt = np.split(paths[0:n], splits)
# a1a2_splt = np.split(opt_data['a1a2'][0:n], splits)
# pop_splt = np.split(opt_data['pop'][0:n], splits)
# opts = []
# for i in range(len(paths_splt)):
#     res = minimize(f, x0=paths_splt[i], args=(a1a2_splt[i], pop_splt[i]), constraints = cons)
#     opts.append(res['x'])
# opts = np.concatenate(opts)




