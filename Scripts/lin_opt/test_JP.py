import pickle
import numpy as np
#import pandas as pd

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



lsoa_data = load_obj("lsoa_data")
pop = np.asarray(lsoa_data['edu_counts']).reshape((len(lsoa_data['edu_counts']), 1))
pop = pop * pop.T
pop = np.tril(pop) #attractivity product
np.fill_diagonal(pop,0) #removing a1a1 scenario as already assumed connection
pop = np.ndarray.flatten(pop)
pop = pop[pop !=0]

optdata = load_obj("opt_data_2000run_lsoa")
pop = optdata['pop'] * optdata['pop'].T
A=(1/np.multiply(optdata['a1a2'],pop))

#print((10**-np.floor(np.log10(np.max(A)))))
#A=A*(10**-np.floor(np.log10(np.max(A))))

paths_matrix = load_obj("ave_paths")
#Added bit---
centroid_info = load_obj("centroids_beta_params_centroiddists")
centroid_paths = np.vstack(centroid_info['euclidean_dists'])
#------------
paths = np.tril(paths_matrix) #lower half of od matrix
paths = np.ndarray.flatten(paths)
paths = paths[paths!=0]

aub12=np.ones([2, len(paths)])/len(paths)
aub12[1][:]*=-1

aub34=np.ones([2, len(paths)])
aub34[1][:]*=-1

p_A_ub=np.append(aub12, aub34, axis=0)

p_b_ub=[np.mean(paths)*1.2, -np.mean(paths)*0.8, np.sum(paths)*1.01, -np.sum(paths)*0.99]

lb=np.ones([1, len(paths)])*min(paths)
ub=np.ones([1, len(paths)])*max(paths)


p_bounds=(min(paths), max(paths))


np.savetxt('f.txt', A)
np.savetxt('x0.txt', paths)
np.savetxt('A.txt', p_A_ub)
np.savetxt('b.txt', p_b_ub)
np.savetxt('minmax_lb.txt', lb)
np.savetxt('minmax_ub.txt', ub)

lb=paths.T * 0.5
ub=paths.T * 1.5
np.savetxt('OD50_lb.txt', lb)
np.savetxt('OD50_ub.txt', ub)


paths = np.minimum(paths_matrix, centroid_paths)
paths = np.tril(paths_matrix) #lower half of od matrix
paths = np.ndarray.flatten(paths)
paths = paths[paths!=0]
lb=np.ones([1, len(paths)])*min(paths)
ub=np.ones([1, len(paths)])*max(paths)
np.savetxt('straightlineMIN_lb.txt', lb)
np.savetxt('straightlineMIN_ub.txt', ub)
