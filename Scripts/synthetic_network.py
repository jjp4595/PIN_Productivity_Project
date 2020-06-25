"""
Synthetic Sheffield
"""

import numpy as np
import pandas as pd
import os
import geopandas as gpd
from scipy import stats
import scipy.optimize

import time
import powerlaw
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#my scripts
import attractivity_modelling
import fractal_working


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#--------------Loading data ---------------------------------------------------
def sheff_import():
    sheff_lsoa = {}
    sheff_lsoa['sheff_lsoa_shape'] = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))
    sheff_lsoa['sheff_lsoa_pop'] = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\tables\KS101EW_lsoa11.csv"))
    sheff_lsoa['sheff_lsoa_pop']['KS101EW0001'] = pd.to_numeric(sheff_lsoa['sheff_lsoa_pop']['KS101EW0001']) #Count: All categories:sex
    sheff_lsoa['sheff_lsoa_income'] = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Individual LSOA Income Estimate\E37000040\spatial\E37000040.shp"))
    #LSOA Income data includes extra LSOA that are not in Sheffield City region, these should be removed.
    ids = sheff_lsoa['sheff_lsoa_income']['lsoa11cd'].isin(sheff_lsoa['sheff_lsoa_pop']['GeographyCode'].values)
    ids = np.where(ids==True)
    sheff_lsoa['sheff_lsoa_income'] = sheff_lsoa['sheff_lsoa_income'].iloc[ids]
    sheff_lsoa['sheff_lsoa_education'] = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\tables\KS501EW_lsoa11.csv"))
    return sheff_lsoa
#sheff_lsoa = sheff_import()


#Graphing
params = {'font.family':'serif',
        'axes.labelsize':'small',
        'xtick.labelsize':'x-small',
        'ytick.labelsize':'x-small', 
        'axes.linewidth':0.5,
        
        'xtick.major.width':0.5,
        'xtick.minor.width':0.4,
        'ytick.major.width':0.5,
        'ytick.minor.width':0.4,
        'xtick.major.size':3.0,
        'xtick.minor.size':1.5,
        'ytick.major.size':3.0,
        'ytick.minor.size':1.5,
        
        'legend.fontsize':'small',
        'legend.title_fontsize':'small',
        'legend.fancybox': False,
        'legend.framealpha': 1,
        'legend.shadow': False,
        'legend.frameon': True,
        'legend.edgecolor':'black',
        'patch.linewidth':0.5,
        
        'scatter.marker': 's',
        
        'grid.linewidth':'0.5',
        
        'lines.linewidth':'0.5'}
plt.rcParams.update(params)


def income_edu_dists():
    
    #Graphing data
    lsoa_data = load_obj("lsoa_data")
    #Sense check distributions
    
    # fig, axs = plt.subplots(15,23)
    # axs = axs.ravel()
    # for i in range(len(lsoa_data['income_params'])):
    #     i
    #     r = stats.beta.rvs(lsoa_data['income_params'][i,0], lsoa_data['income_params'][i,1], loc =lsoa_data['income_params'][i,2], scale = lsoa_data['income_params'][i,3], size = 1000)
    #     axs[i].plot(np.linspace(0,1,100),  stats.beta.pdf(np.linspace(0,1,100), lsoa_data['income_params'][i, 0], lsoa_data['income_params'][i, 1], loc = lsoa_data['income_params'][i, 2], scale = lsoa_data['income_params'][i, 3]) , label = 'CDF')
    #     axs[i].hist(r, density = True)
    # plt.tight_layout()
    
    # fig, axs = plt.subplots(15,23)
    # axs = axs.ravel()
    # for i in range(len(lsoa_data['income_params'])):
    #     education=np.random.choice(4, size = lsoa_data['edu_counts'][i], p=lsoa_data['edu_ratios'][i]) #where p values are effectively the ratio of people with a given education level you can alternatively use the same method for income as well    
    #     axs[i].hist(education, density = True)
    # plt.tight_layout()
    sns.set_palette("deep")
    sns.set_color_codes()
    maxid, minid = 103, 337 #income means max/min lsoa id


    fig=plt.figure()
    fig.set_size_inches(2,3)
    gs = gridspec.GridSpec(2, 1, height_ratios=[10,1], width_ratios=[1], wspace=0.25, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex = ax0)
    inc1 = stats.beta.rvs(lsoa_data['income_params'][minid,0], lsoa_data['income_params'][minid,1], loc =lsoa_data['income_params'][minid,2], scale = lsoa_data['income_params'][minid,3], size = 1000)
    sns.distplot(inc1, color='b', kde=False, norm_hist=True, ax=ax0)
    ax0.plot(np.linspace(0,1,100),  stats.beta.pdf(np.linspace(0,1,100), lsoa_data['income_params'][minid, 0], lsoa_data['income_params'][minid, 1], loc = lsoa_data['income_params'][minid, 2], scale = lsoa_data['income_params'][minid, 3]) , label = 'CDF')
    ax0.axvline(x=np.median(inc1), ls='--', lw=2)
    sns.pointplot(data = inc1, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.set_xlim(0,1)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Normalised income')
    plt.tight_layout()
    
    fig=plt.figure()
    fig.set_size_inches(2,3)
    gs = gridspec.GridSpec(2, 1, height_ratios=[10,1], width_ratios=[1], wspace=0.25, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex = ax0)
    inc2 = stats.beta.rvs(lsoa_data['income_params'][maxid,0], lsoa_data['income_params'][maxid,1], loc =lsoa_data['income_params'][maxid,2], scale = lsoa_data['income_params'][maxid,3], size = 1000)
    sns.distplot(inc2, color='b', kde=False, norm_hist=True, ax=ax0)
    ax0.plot(np.linspace(0,1,100),  stats.beta.pdf(np.linspace(0,1,100), lsoa_data['income_params'][maxid, 0], lsoa_data['income_params'][maxid, 1], loc = lsoa_data['income_params'][maxid, 2], scale = lsoa_data['income_params'][maxid, 3]) , label = 'CDF')
    ax0.axvline(x=np.median(inc2), ls='--', lw=2)
    sns.pointplot(data = inc2, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.set_xlim(0,1)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Normalised income')
    plt.tight_layout()
   


    
    
    fig, [ax0, ax1] = plt.subplots(1,2)
    fig.set_size_inches(4,2.5)
    edu1=np.random.choice(4, size = 1000, p=lsoa_data['edu_ratios'][minid]) #where p values are effectively the ratio of people with a given education level you can alternatively use the same method for income as well    
    sns.countplot(edu1, color = 'b', ax=ax0)
    ax0.set_ylabel('Count')
    ax0.set_xlabel('Highest education level')
    plt.tight_layout()
    
    edu2=np.random.choice(4, size = 1000, p=lsoa_data['edu_ratios'][maxid])
    sns.countplot(edu2, color='b', ax=ax1)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Highest education level')
    plt.tight_layout()
    
    attractivity1=inc1**(-edu1)   
    attractivtiy1_powerlaw = powerlaw.Fit(attractivity1)
    attractivtiy1_powerlaw_temp = powerlaw.Fit(attractivity1, xmin=1)
    fig = plt.figure()
    fig.set_size_inches(2.5,2.5)
    powerlaw_plot=attractivtiy1_powerlaw_temp.plot_ccdf(original_data=True, color='b', marker='.', ms=2, lw=0, alpha=0.5)
    X=attractivtiy1_powerlaw_temp.ccdf()
    x=X[0][X[0]>=attractivtiy1_powerlaw.xmin]
    y=[np.exp( (-(attractivtiy1_powerlaw.alpha-1)*(np.log(i)-np.log(attractivtiy1_powerlaw.xmin))+np.log(X[1][X[0]==attractivtiy1_powerlaw.xmin])) ) for i in x]
    plt.plot(x, y, 'k')
    plt.ylabel(r'$P(attractivity \geq x)$')
    plt.xlabel(r'$attractivity$')
    plt.tight_layout()
    
    attractivity2=inc2**(-edu2)   
    attractivtiy2_powerlaw = powerlaw.Fit(attractivity2)
    attractivtiy2_powerlaw_temp = powerlaw.Fit(attractivity2, xmin=1)
    fig = plt.figure()
    fig.set_size_inches(2.5,2.5)
    powerlaw_plot=attractivtiy2_powerlaw_temp.plot_ccdf(original_data=True, color='b', marker='.', ms=2, lw=0, alpha=0.5)
    X=attractivtiy2_powerlaw_temp.ccdf()
    x=X[0][X[0]>=attractivtiy2_powerlaw.xmin]
    y=[np.exp( (-(attractivtiy2_powerlaw.alpha-1)*(np.log(i)-np.log(attractivtiy2_powerlaw.xmin))+np.log(X[1][X[0]==attractivtiy2_powerlaw.xmin])) ) for i in x]
    plt.plot(x, y, 'k')
    plt.ylabel(r'$P(attractivity \geq x)$')
    plt.xlabel(r'$attractivity$')
    plt.tight_layout()
    
    paths_matrix = load_obj("ave_paths")
    paths = paths_matrix.flatten()
    centroid_info = load_obj("centroids_beta_params_centroiddists")
    centroid_paths = np.vstack(centroid_info['euclidean_dists']).flatten()
    
    fig=plt.figure()
    fig.set_size_inches(3,3)
    gs = gridspec.GridSpec(2, 1, height_ratios=[10,1], width_ratios=[1], wspace=0.25, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex = ax0)
    sns.distplot(centroid_paths, color='b', kde=False,  ax=ax0)
    ax0.axvline(x=np.median(centroid_paths), ls='--', lw=2)
    sns.pointplot(data = centroid_paths, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.set_xlim(0,30000)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Centroid path distances')
    plt.tight_layout()
    
    fig=plt.figure()
    fig.set_size_inches(3,3)
    gs = gridspec.GridSpec(2, 1, height_ratios=[10,1], width_ratios=[1], wspace=0.25, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex = ax0)
    sns.distplot(paths, color='b', kde=False,  ax=ax0)
    ax0.axvline(x=np.median(paths), ls='--', lw=2)
    sns.pointplot(data = paths, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.set_xlim(0,30000)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Driveable path distance')
    plt.tight_layout()
    
    fig=plt.figure()
    fig.set_size_inches(3,3)
    gs = gridspec.GridSpec(2, 1, height_ratios=[10,1], width_ratios=[1], wspace=0.25, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex = ax0)
    diffs = abs(paths-centroid_paths)
    sns.distplot(diffs, color='b', kde=False,  ax=ax0)
    ax0.axvline(x=np.median(diffs), ls='--', lw=2)
    sns.pointplot(data = diffs, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax0.set_xlim(0,7000)
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Difference in distances')
    plt.tight_layout()
    
def plotgraph():
    import osmnx as ox
    import path_querying
    G1 = path_querying.load_graph()
    ox.plot.plot_graph(G1, dpi = 600, node_color = 'None', fig_width=6, fig_height=6)
    
#------------------------------------------------------------------------------
 
                                

#Sampling attractivities --------------------------------------
def sample_attractivities(edu_ratios, income_params, fit = None):  

    attractivity1 = np.zeros((len(income_params)))
    attractivity2 = np.zeros((len(income_params)))
    for i in range(len(income_params)): #Loop across  OAs            
        attractivity1[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)                     
        attractivity2[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)
        
    if fit != None:
        all_attractivity = np.concatenate((attractivity1, attractivity1) , axis=0)
        attractivity_powerlaw = powerlaw.Fit(all_attractivity)
        alpha = attractivity_powerlaw.alpha
        xmin = attractivity_powerlaw.xmin
        return attractivity1, attractivity2, alpha, xmin
    else:
        return attractivity1, attractivity2

def euclidean_dists_fun(sheff_shape): 
    """
    Dummy distances function. 
    """    
    euclidean_dists = []
    point1s = []
    centroids = sheff_shape.centroid
    for i in range(len(sheff_shape)):
        euclidean_dists.append(centroids.distance(centroids[i]).values)
        point1s.append((centroids.x[i], centroids.y[i]))
    all_coords = pd.DataFrame(point1s, columns = ['x-coord', 'y-coord'])  
    
    #generating path matrix
    paths_matrix = np.column_stack(euclidean_dists)
    
    #median path distances     
    paths = np.concatenate(euclidean_dists, axis=0)
    paths = paths[paths != 0]
    med_paths = sorted(paths)
    med_paths = int(med_paths[int(len(med_paths)/2)])
    
    return euclidean_dists, all_coords, paths_matrix, med_paths
#euclidean_dists, centroids, paths_matrix, med_paths = euclidean_dists_fun()



def fractal_dimension(coords_data):
    """
    Graph may require some intuition to fit the linear regression through certain points
    """
    rangex = coords_data['x-coord'].values.max() - coords_data['x-coord'].values.min()
    rangey = coords_data['y-coord'].values.max() - coords_data['y-coord'].values.min()
    L = int(max(rangex, rangey)) # max of x or y distance
    
    r = np.array([ L/(2.0**i) for i in range(5,0,-1) ]) #Create array of box lengths
    N = [ fractal_working.count_boxes( coords_data, ri, L ) for ri in r ] #Non empty boxes for each array of box lenghts
     
    popt, pcov = scipy.optimize.curve_fit( fractal_working.f_temp, np.log( 1./r ), np.log( N ) )
    A, Df = popt #A lacunarity, Df fractal dimension
    
    
    # fig, ax = plt.subplots(1,1)
    # ax.plot(1./r, N, 'b.-')
    # ax.plot( 1./r, np.exp(A)*1./r**Df, 'g', alpha=1.0 )
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_aspect(1)
    # ax.set_xlabel('Box Size')
    # ax.set_ylabel('Number of boxes')
    
    # #Playing around with data points to use
    # Y = np.log( N )
    # X = np.log( 1./r )
    # T = np.vstack((Y,X,np.ones_like(X))).T
     
    # df = pd.DataFrame( T, columns=['N(r)','Df','A'] )
    # Y = df['N(r)']
    # X = df[['Df','A']]
    # result = OLS( Y, X ).fit()
    # result.summary()
    return Df



#Shuffling data 
def paths_shuffle(shape, income_params):
    """
    Returns the mean of each oa's income distribution'
    """
    means = []
    for oa in range(len(shape)):
        means.append(stats.beta.mean(income_params[oa, 0], income_params[oa, 1], loc = income_params[oa, 2], scale = income_params[oa, 3]))
    return means



#Monte Carlo run throughs ----------------------------------------------------
def monte_carlo_runs(m, n, lsoa_data, paths_matrix, is_shuffled=None):
    startt = time.time()
    time_log = []  
    
    
    sheff_shape, income_params, edu_counts, edu_ratios = lsoa_data['sheff_lsoa_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']
    
    #Constants
    base_m = 1
    
    
    #dummy distances
    euclidean_dists, centroids, centroid_paths_matrix, med_paths = euclidean_dists_fun(sheff_shape)
    eps = med_paths
    
    
    if is_shuffled is None:
        pass 
    else:
        east_inds = centroids["x-coord"].argsort().values  #low to high sort
        
        income_inds = np.argsort(paths_shuffle(sheff_shape, income_params)) #returns indices that would sort the array, low to high
        
    
    
    #fractal dimension
    Df = fractal_dimension(centroids)
    
    #create data structures
    UrbanY = []
    edges = np.zeros((len(sheff_shape), len(sheff_shape), n))
    
    for i in range(n):
        
        
        #Sample attractivities
        attractivity1, attractivity2, alpha, xmin = sample_attractivities(edu_ratios, income_params, 1)
        #alpha = 1.45653 #mean fixed alpha from 1000 runs
        
        theta = np.exp(np.log(xmin**2) - (base_m*np.log(eps)))
        dc = base_m * (alpha - 1)
        
        
        #connectivity matrix
        attractivity1 = attractivity1.reshape((len(attractivity1),1))
        attractivity2 = attractivity2.reshape((len(attractivity2),1))
        
        #population amplification
        pop = np.asarray(edu_counts).reshape((len(edu_counts), 1))
              
        
        if is_shuffled is None:
            pop = np.matmul(pop, pop.transpose()) 
        else:
            attractivity1[east_inds] = attractivity1[income_inds]
            attractivity2[east_inds] = attractivity2[income_inds]

            pop[east_inds] = pop[income_inds]
            pop = np.matmul(pop, pop.transpose())

        
        attractivity_product = np.matmul(attractivity1, attractivity2.transpose())
        
        #ensure 0 on diagonal?   
        connectivity = np.divide(attractivity_product, np.power(paths_matrix, m))
        connectivity[np.where(np.isinf(connectivity))[0], np.where(np.isinf(connectivity))[1]] = 0
        connectivity[np.diag_indices_from(connectivity)] = 0
        
        #adjacency matrix
        adjacency = np.zeros_like(connectivity)
        adjacency[np.where(connectivity>theta)] = 1      
        adjacency = np.multiply(adjacency, pop) #population amplification factor
        edges[:,:,i] = adjacency
        
        
        if Df <= dc:
            eta = ((-5/6) * Df) + dc
        else: 
            eta = (Df/6)
        
        #activity
        paths_matrix_n = (paths_matrix - paths_matrix.min()) / (paths_matrix.max() - paths_matrix.min()) +1
        activity = np.power(paths_matrix_n, eta)
        activity[np.where(np.isinf(activity))[0], np.where(np.isinf(activity))[1]] = 0
        
        UrbanY.append( 0.5 * np.sum(np.multiply(adjacency, activity)) )
        #UrbanY.append( 0.5 * np.sum(adjacency)) 
        
        
    #Creating network data
    edge_freq = np.count_nonzero(edges, axis = 2) / n      
    edge_width = np.sum(edges, axis = 2) / n 
    
    endt = time.time()
    print("Time for this n run through is: "+str(endt-startt))
    
    
    time_log.append(endt-startt)
    total_time = sum(time_log)
    print("Total run time is: " + str(total_time))
    return UrbanY, edge_freq, edge_width





#------------------------------------------------------------------------------
#
#                   Running Monte Carlo
#
#------------------------------------------------------------------------------
import multiprocessing
if __name__ == '__main__':
    #imports
    lsoa_data = load_obj("lsoa_data")
    import scipy.io as sio 
    mldata = sio.loadmat(r'C:\Users\cip18jjp\Dropbox\PIN\hadi_scripts\optimisedpaths.mat')#import new paths
    n = 500

    # 2a)--------------------------------------
    # Normal paths, m = [0.5, 1, 2]
    # -----------------------------------------
    
    
    t1 = time.time()
    no_scripts = multiprocessing.cpu_count()
    
    ms = [0.5, 1, 2]
    
    paths_matrix = load_obj("ave_paths")
    args_normal = []
    
    for i in range(len(ms)):
        args_normal.append((ms[i], n, lsoa_data, paths_matrix))
    
    with multiprocessing.Pool(processes=no_scripts) as pool:
        output = pool.starmap(monte_carlo_runs, args_normal)
           
    UrbanYs, edge_freqs, edge_widths = [], [], []
    for i in range(len(output)):
        UrbanYs.append(output[i][0])
        edge_freqs.append(output[i][1])
        edge_widths.append(output[i][2])
    
    normal = {
        "UrbanYs": UrbanYs,
        "edge_freqs": edge_freqs,
        "edge_widths": edge_widths,
        "m_values": ms
        }    
    
    save_obj(normal, "normal_layout_"+str(n)+"run")
    
    print(time.time()-t1)


    #3a)--------------------------------------
    # #minmax paths, m = [0.5, 1, 2]
    #-----------------------------------------
    
    paths = mldata['minmax']
    paths = np.reshape(paths, (len(paths)))
    paths_matrix = np.zeros((345,345))
    lowInds = np.tril_indices(345,k=-1)
    highInds  = np.triu_indices(345, k=1)
    paths_matrix[lowInds] = paths
    paths_matrix = paths_matrix + paths_matrix.T - np.diag(paths_matrix)     
    
    t1 = time.time()
    no_scripts = multiprocessing.cpu_count()
    
    ms = [0.5, 1, 2]


    args_normal = []
    
    for i in range(len(ms)):
        args_normal.append((ms[i], n, lsoa_data, paths_matrix))
    
    with multiprocessing.Pool(processes=no_scripts) as pool:
        output = pool.starmap(monte_carlo_runs, args_normal)
           
    UrbanYs, edge_freqs, edge_widths = [], [], []
    for i in range(len(output)):
        UrbanYs.append(output[i][0])
        edge_freqs.append(output[i][1])
        edge_widths.append(output[i][2])

    normal = {
        "UrbanYs": UrbanYs,
        "edge_freqs": edge_freqs,
        "edge_widths": edge_widths,
        "m_values": ms
        }    
    save_obj(normal, "minmax_layout_"+str(n)+"run")
    
    print(time.time()-t1)

    #3b)--------------------------------------
    # #slmin straightline paths, m = [0.5, 1, 2]
    #-----------------------------------------
    paths = mldata['slmin']
    paths = np.reshape(paths, (len(paths)))
    
    paths_matrix = np.zeros((345,345))
    lowInds = np.tril_indices(345,k=-1)
    highInds  = np.triu_indices(345, k=1)
    paths_matrix[lowInds] = paths
    paths_matrix = paths_matrix + paths_matrix.T - np.diag(paths_matrix)     
    
    #monte carlo set up
    t1 = time.time()
    no_scripts = multiprocessing.cpu_count()
    
    ms = [0.5, 1, 2]


    args_normal = []
    
    for i in range(len(ms)):
        args_normal.append((ms[i], n, lsoa_data, paths_matrix))
    
    with multiprocessing.Pool(processes=no_scripts) as pool:
        output = pool.starmap(monte_carlo_runs, args_normal)
           
    UrbanYs, edge_freqs, edge_widths = [], [], []
    for i in range(len(output)):
        UrbanYs.append(output[i][0])
        edge_freqs.append(output[i][1])
        edge_widths.append(output[i][2])


    normal = {
        "UrbanYs": UrbanYs,
        "edge_freqs": edge_freqs,
        "edge_widths": edge_widths,
        "m_values": ms
        }    

    save_obj(normal, "slmin_layout_"+str(n)+"run")

    print(time.time()-t1)


    #3c)--------------------------------------
    # #+- 50% paths, m = [0.5, 1, 2]
    #-----------------------------------------
    paths = mldata['od50']
    paths = np.reshape(paths, (len(paths)))
    
    paths_matrix = np.zeros((345,345))
    lowInds = np.tril_indices(345,k=-1)
    highInds  = np.triu_indices(345, k=1)
    paths_matrix[lowInds] = paths
    paths_matrix = paths_matrix + paths_matrix.T - np.diag(paths_matrix)     
    
    #monte carlo set up
    t1 = time.time()
    no_scripts = multiprocessing.cpu_count()
    
    ms = [0.5, 1, 2]

    args_normal = []
    
    for i in range(len(ms)):
        args_normal.append((ms[i], n, lsoa_data, paths_matrix))
    
    with multiprocessing.Pool(processes=no_scripts) as pool:
        output = pool.starmap(monte_carlo_runs, args_normal)
           
    UrbanYs, edge_freqs, edge_widths = [], [], []
    for i in range(len(output)):
        UrbanYs.append(output[i][0])
        edge_freqs.append(output[i][1])
        edge_widths.append(output[i][2])

    normal = {
        "UrbanYs": UrbanYs,
        "edge_freqs": edge_freqs,
        "edge_widths": edge_widths,
        "m_values": ms
        }    
    save_obj(normal, "od50_layout_"+str(n)+"run")

    print(time.time()-t1)
#------------------------------------------------------------------------------

