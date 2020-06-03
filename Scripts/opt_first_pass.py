"""
Optimisation attempt
"""


import numpy as np
import pickle
from scipy.optimize import linprog, minimize, NonlinearConstraint
import multiprocessing

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



#For sampling attractivities -------------------------------------------------




#1a) Directly run monte carlo
lsoa_data = load_obj("lsoa_data")
s, idx, income_params, edu_counts, edu_ratios = lsoa_data['s'], lsoa_data['idx'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']
pop = np.asarray(edu_counts).reshape((len(edu_counts), 1))
a1, a2 = [], []
for i in range(200): #Sample run for attractivities
    attractivity1, attractivity2 = sample_attractivities(s,idx, edu_ratios, income_params)
    a1.append(attractivity1)
    a2.append(attractivity2)


# a1 = np.stack(a1, axis=0).mean(0)
# a2 = np.stack(a2, axis=0).mean(0)
# a1 = a1.reshape((len(a1),1))
# a2 = a2.reshape((len(a2),1))
# a1a2 = np.tril(np.matmul(a1, a2.transpose())) #attractivity product
# np.fill_diagonal(a1a2,0) #removing a1a1 scenario as already assumed connection
# a1a2 = np.ndarray.flatten(a1a2)
# a1a2 = a1a2[a1a2 !=0]
#save_obj(a1a2,"a1a2_1000run")

#2) load premade
#a1a2 = load_obj("a1a2_1000run")

#-----------------------------------------------------------------------------



# #Load paths ------------------------------------------------------------------
# paths_matrix = load_obj("ave_paths")
# paths = np.tril(paths_matrix) #lower half of od matrix
# paths = np.ndarray.flatten(paths)
# paths = paths[paths !=0]
# #paths[paths==max(paths)] = #in od matrix I have raised the exception for 1e8 when a solution can't be found


# def f(paths, a1a2):
#     out = np.divide(a1a2, paths)
#     return out.sum()*0.5*-1

# def eq_constraint(x):
#     return sum(x)
# def min_constraint(x):
#     return min(x)
# def max_constraint(x):
#     return max(x)


# nonlin_eq = NonlinearConstraint(eq_constraint, paths.sum(), paths.sum())
# nonlin_min = NonlinearConstraint(min_constraint, min(paths), np.inf)
# nonlin_max = NonlinearConstraint(max_constraint, 0, max(paths))
# cons = [nonlin_eq, nonlin_min, nonlin_max]



# #------------------------------------------------------------------------------
# #testing splitting into different sections
# # n = 4000
# # splits = 20
# # res = minimize(f, x0=paths[0:n], args=(a1a2[0:n]), constraints = cons)
# # opt_edges = res['x']



# # paths_splt = np.split(paths[0:n], splits)
# # a1a2_splt = np.split(a1a2[0:n], splits)
# # opts = []
# # for i in range(len(paths_splt)):
# #     res = minimize(f, x0=paths_splt[i], args=(a1a2_splt[i]), constraints = cons)
# #     opts.append(res['x'])
# # opts = np.concatenate(opts)
# # diff = opts-opt_edges


# #-----------------------------------------------------------------------------
# n = 100
# eq = np.asarray([np.sum(paths[0:n]) for i in range(n)])

# boundss = [(np.min(paths), np.max(paths)) for i in range(n)]

# res = linprog(paths[0:n], A_eq = eq, bounds = boundss)


# #save_obj(opt_edges, "optimum_edge")