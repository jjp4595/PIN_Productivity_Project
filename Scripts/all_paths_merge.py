# -*- coding: utf-8 -*-
"""
Script for merging all_paths
"""


import pickle
import pandas as pd
import os
import glob
import numpy as np

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def FileAddressList(fileIN, GaugeData=None):
    #This functions reads the fileIN and then creates the filelist as a list of strings from that recursive search
    #If a second argument is entered, then it will import the gauge data. 
    #In other words, one input for file lists, add a number for gauge data import.
    OutputList=[]
    if GaugeData is None:
        for filename in glob.glob(fileIN, recursive=True):
            OutputList.append(filename)
    else: 
        for filename in glob.glob(fileIN, recursive=True):
            OutputList.append(np.loadtxt(filename))
    return OutputList


paths_list = FileAddressList(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\Scripts\paths\*.pkl"))



all_paths = []
for i in range(len(paths_list)):
    all_paths.append(load_obj(paths_list[i]))
             
    
paths = np.zeros((len(all_paths[0]), len(all_paths[0]), len(paths_list)))
for i in range(len(paths_list)):
    paths[:,:,i] = all_paths[i].to_numpy()
    
ave_paths = paths.mean(2)

save_obj(ave_paths, 'ave_paths')