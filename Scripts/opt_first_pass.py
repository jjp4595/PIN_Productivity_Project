# -*- coding: utf-8 -*-
"""
Optimisation attempt
"""


import numpy as np
import pickle

from synthetic_network import sample_attractivities

from scipy.optimize import linprog


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

lsoa_data = load_obj("lsoa_data")
paths_matrix = load_obj("ave_paths")
paths = np.concatenate(paths_matrix, axis=0)
paths = paths[paths != 0]
med_paths = sorted(paths)
med_paths = int(med_paths[int(len(med_paths)/2)])    
eps = med_paths


s, idx, sheff_shape, income_params, edu_counts, edu_ratios = lsoa_data['s'], lsoa_data['idx'], lsoa_data['sheff_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']
attractivity1, attractivity2, alpha, xmin = sample_attractivities(s,idx, edu_ratios, income_params)


lol = np.divide(np.matmul(attractivity1, attractivity2.transpose()), paths_matrix)

