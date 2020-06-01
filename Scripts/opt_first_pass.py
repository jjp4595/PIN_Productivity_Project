"""
Optimisation attempt
"""


import numpy as np
import pickle
from scipy.optimize import linprog, minimize, NonlinearConstraint

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



#For sampling attractivities -------------------------------------------------
# from attractivity_modelling import attractivity_sampler
# def sample_attractivities(s,idx, edu_ratios, income_params):  
#     attractivity1 = np.zeros((s))
#     attractivity2 = np.zeros((s))
#     for i in range(len(idx)): #Loop across  OAs         
#         attractivity1[i] = attractivity_sampler(i, edu_ratios, income_params)                     
#         attractivity2[i] = attractivity_sampler(i, edu_ratios, income_params)
#     return attractivity1, attractivity2


#1a) Directly run monte carlo
# lsoa_data = load_obj("lsoa_data")
# s, idx, income_params, edu_counts, edu_ratios = lsoa_data['s'], lsoa_data['idx'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']
# a1, a2 = [], []
# for i in range(1000):
#     attractivity1, attractivity2 = sample_attractivities(s,idx, edu_ratios, income_params)
#     a1.append(attractivity1)
#     a2.append(attractivity2)
# a1 = np.stack(a1, axis=0).mean(0)
# a2 = np.stack(a2, axis=0).mean(0)
# a1 = a1.reshape((len(a1),1))
# a2 = a2.reshape((len(a2),1))
# a1a2 = np.tril(np.matmul(a1, a2.transpose())) #attractivity product
# np.fill_diagonal(a1a2,0) #removing a1a1 scenario as already assumed connection
# a1a2 = np.ndarray.flatten(a1a2)
# a1a2 = a1a2[a1a2 !=0]
# save_obj(a1a2,"a1a2_1000run")
#1b) Multiprocess
# if __name__ == '__main__':
#     import multiprocessing
#     lsoa_data = load_obj("lsoa_data")
#     s, idx, income_params, edu_counts, edu_ratios = lsoa_data['s'], lsoa_data['idx'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']
#     def runs():
#         a1, a2 = [], []
#         for i in range(100):
#             attractivity1, attractivity2 = sample_attractivities(s,idx, edu_ratios, income_params)
#             a1.append(attractivity1)
#             a2.append(attractivity2)

        
#     no_scripts = multiprocessing.cpu_count() #when running on personal
#     args = []    
#     for i in range(no_scripts):
#         args.append((s, idx, income_params, edu_counts, edu_ratios))   
#     with multiprocessing.Pool(processes=no_scripts) as pool:
#         all_runs = pool.starmap(runs, args)
#         #all_runs = pd.concat(all_runs)
#         # a1 = np.stack(a1, axis=0).mean(0)
#         # a2 = np.stack(a2, axis=0).mean(0)
#         # a1 = a1.reshape((len(a1),1))
#         # a2 = a2.reshape((len(a2),1))
#     #save_obj(all_paths, 'all_paths')
# a1a2 = np.tril(np.matmul(a1, a2.transpose())) #attractivity product
# np.fill_diagonal(a1a2,0) #removing a1a1 scenario as already assumed connection
# a1a2 = np.ndarray.flatten(a1a2)
# a1a2 = a1a2[a1a2 !=0]
# save_obj(a1a2,"a1a2_1000run")

#2) load premade
a1a2 = load_obj("a1a2_1000run")
#-----------------------------------------------------------------------------



#Load paths ------------------------------------------------------------------
paths_matrix = load_obj("ave_paths")
paths = np.tril(paths_matrix) #lower half of od matrix
paths = np.ndarray.flatten(paths)
paths = paths[paths !=0]
#paths[paths==max(paths)] = #in od matrix I have raised the exception for 1e8 when a solution can't be found



def f(paths, a1a2):
    out = np.divide(a1a2, paths)
    return out.sum()*0.5*-1

def eq_constraint(x):
    return sum(x)
def min_constraint(x):
    return min(x)
def max_constraint(x):
    return max(x)

nonlin_eq = NonlinearConstraint(eq_constraint, paths_matrix.sum(), paths_matrix.sum())
nonlin_min = NonlinearConstraint(min_constraint, min(paths), np.inf)
nonlin_max = NonlinearConstraint(max_constraint, 0, max(paths))
cons = [nonlin_eq, nonlin_min, nonlin_max]


res = minimize(f, x0=paths, args=(a1a2), constraints = cons)
opt_edges = res['x']
save_obj(opt_edges, "optimum_edge")
