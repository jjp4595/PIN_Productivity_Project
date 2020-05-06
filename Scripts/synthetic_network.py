"""
Synthetic Sheffield
"""

import numpy as np
import pandas as pd
import os
import geopandas as gpd
from scipy import stats
import scipy.optimize
from pyproj import CRS
import osmnx as ox
import networkx as nx
import time
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.regression.linear_model import OLS
import powerlaw
import pickle
from densratio import densratio

import attractivity_modelling
import path_querying
import fractal_working


#Importing files---------------------------------------------------------------
#Wokshop 2 data
#sheff_OShighways = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_OA.shp"))
#sheff_network = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_Network.gdb"))

#Graph
def load_graph():
    """
    Loading Graph ---------------------------------------------------------------
    """
    #Getting drivable streets in sheffield
    place_name = "Sheffield, UK"
    graph = ox.graph_from_place(place_name, network_type = 'drive')
    #graph = ox.graph_from_file() # create graph from OSM data in XML file
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True) #Convert graph int geodataframe
    crs = CRS.from_string("epsg:27700")#convert data into same co-ordinate system as lsoa data
    graph_proj = ox.project_graph(graph, to_crs=crs)
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True) #Convert graph into UTM zone 30 format (with m units)
    return graph_proj 
#graph_proj = load_graph()



# #Shape
# sheff_lsoa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))
# #sheff_oa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_oa11.shp"))

# #Population
# sheff_lsoa_pop = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\tables\KS101EW_lsoa11.csv"))
# sheff_lsoa_pop['KS101EW0001'] = pd.to_numeric(sheff_lsoa_pop['KS101EW0001']) #Count: All categories:sex
# #sheff_oa_pop = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\tables\KS101EW_oa11.csv"))
# #sheff_oa_pop['KS101EW0001'] = pd.to_numeric(sheff_oa_pop['KS101EW0001']) #Count: All categories:sex

# #Income
# def trim_lsoa():
#     sheff_lsoa_income = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Individual LSOA Income Estimate\E37000040\spatial\E37000040.shp"))
#     #LSOA Income data includes extra LSOA that are not in Sheffield City region, these should be removed.
#     ids = sheff_lsoa_income['lsoa11cd'].isin(sheff_lsoa_pop['GeographyCode'].values)
#     ids = np.where(ids==True)
#     sheff_lsoa_income = sheff_lsoa_income.iloc[ids]
#     return sheff_lsoa_income
# sheff_lsoa_income = trim_lsoa()
# #Education
# sheff_lsoa_education = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\tables\KS501EW_lsoa11.csv"))



#------------------------------------------------------------------------------





                                                                   

##Generate income and education distributions from OA/LSOA
#income_params, edu_counts, edu_ratios = attractivity_modelling.attractivity(sheff_lsoa_shape, sheff_lsoa_pop, sheff_lsoa_income, sheff_lsoa_education, idx)
#Sampling attractivities --------------------------------------
def sample_attractivities(s,idx, edu_ratios, income_params):  


    attractivity1 = np.zeros((s))
    attractivity2 = np.zeros((s))
    for i in range(len(idx)): #Loop across  OAs    
        
        attractivity1[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)                     
        attractivity2[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)
        
    
    all_attractivity = np.concatenate((attractivity1, attractivity1) , axis=0)
    attractivity_powerlaw = powerlaw.Fit(all_attractivity)
    alpha = attractivity_powerlaw.alpha
    xmin = attractivity_powerlaw.xmin
    return attractivity1, attractivity2, alpha, xmin
#attractivity1, attractivity2, alpha, xmin = sample_attractivities(s,idx, edu_ratios, income_params)


#Distance sampling -----------------------------------------------------------
#Dummy distances
def euclidean_dists_fun(sheff_shape):
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


#Real distances
def sample_coords(idx, sheff_shape):
    """
    Sample random coordinates in a polygon to be used for path querying
    all_coords is collected list of origin and target coords
    """
    point1s = []
    point2s = []
    origin=[]
    target=[]
    for i in range(len(idx)): #Loop across  OAs    
        point1s.append(path_querying.get_random_point_in_polygon(sheff_shape['geometry'][idx[i]]))
        origin.append((point1s[i].x, point1s[i].y))
        
        point2s.append(path_querying.get_random_point_in_polygon(sheff_shape['geometry'][idx[i]]))                     
        target.append((point2s[i].x, point2s[i].y))
    
    all_coords = origin + target
    all_coords = pd.DataFrame(all_coords, columns = ['x-coord', 'y-coord'])
    return all_coords, point1s, point2s
#all_coords, point1s, point2s = sample_coords(idx, sheff_shape)


def run_shortest_path(s, idx, point1s, point2s, graph_proj):
    """
    Running shortest path function
    """
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
    med_paths = int(med_paths[int(len(med_paths)/2)])
    return paths_matrix, med_paths
#paths_matrix, med_paths = run_shortest_path(s, idx, point1s, point2s, graph_proj)


#-----------------------------------------------------------------------------
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






#Import script ro create attractivity distributions
def create_attractivity_dists(shape, pop, income, education):
    """
    This is run once and the distributions are saved. 
    """
    #setting indices of OAs/LSOAs
    s = 1
    s = int(s*  len(shape)) 
    idx = np.round(np.linspace(0, len(shape) - 1, s)).astype(int) #Indexing s spaced from array
    
    income_params, edu_counts, edu_ratios = attractivity_modelling.attractivity(shape, pop, income, education, idx)
    return s, idx, income_params, edu_counts, edu_ratios
#s, idx, income_params, edu_counts, edu_ratios = create_attractivity_dists(sheff_lsoa_shape, sheff_lsoa_pop, sheff_lsoa_income, sheff_lsoa_education)



def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#Saving data 
# lsoa_dist = {"sheff_shape":sheff_lsoa_shape,"s":s, "idx":idx, "income_params":income_params, "edu_counts":edu_counts, "edu_ratios":edu_ratios}
# save_obj(lsoa_dist, "lsoa_data")



#Shuffling data 
def paths_shuffle(shape, income_params):
    """
    Shuffle data based on income parameter means
    """
    means = []
    for oa in range(len(shape)):
        means.append(stats.beta.mean(income_params[oa, 0], income_params[oa, 1], loc = income_params[oa, 2], scale = income_params[oa, 3]))
    return means


  
  
    





#Monte Carlo run throughs
def monte_carlo_runs(m, n, s, idx, sheff_shape, income_params, edu_counts, edu_ratios, is_shuffled = None):
    startt = time.time()
    time_log = []  
    
    #Constants

    
    
    #dummy distances
    euclidean_dists, centroids, paths_matrix, med_paths = euclidean_dists_fun(sheff_shape)
    eps = med_paths
    
    if is_shuffled is None:
        pass 
    else:        
        new_inds = np.argsort(paths_shuffle(sheff_shape, income_params))
        paths_matrix = paths_matrix[new_inds, :]
        paths_matrix = paths_matrix[: , new_inds]
    
    
    
    #fractal dimension
    Df = fractal_dimension(centroids)
    
    #create data structures
    UrbanY = []
    edges = np.zeros((len(sheff_shape), len(sheff_shape), n))
    
    for i in range(n):
        
        
        #Sample attractivities
        attractivity1, attractivity2, alpha, xmin = sample_attractivities(s,idx, edu_ratios, income_params)
        
        
        theta = np.exp(np.log(xmin**2) - (m*np.log(eps)))
        dc = m * (alpha - 1)
        
        #connectivity matrix
        attractivity1 = attractivity1.reshape((len(attractivity1),1))
        attractivity2 = attractivity2.reshape((len(attractivity2),1))
        attractivity_product = np.matmul(attractivity1, attractivity2.transpose())
        
        connectivity = np.divide(attractivity_product, np.power(paths_matrix, m))
        connectivity[np.where(np.isinf(connectivity))[0], np.where(np.isinf(connectivity))[1]] = 0        
        
        #adjacency matrix
        adjacency = np.zeros_like(connectivity)
        adjacency[np.where(connectivity>theta)] = 1
        
        pop = np.stack(edu_counts).reshape((len(edu_counts), 1))
        pop = np.matmul(pop, pop.transpose())
        adjacency = np.multiply(adjacency, pop) #population amplification factor
        edges[:,:,i] = adjacency
        
        
        if Df <= dc:
            eta = ((-5/6) * Df) + dc
        else: 
            eta = (Df/6)
        
        #activity
        activity = np.power(paths_matrix, eta)
        activity[np.where(np.isinf(activity))[0], np.where(np.isinf(activity))[1]] = 0
        
        UrbanY.append( 0.5 * np.sum(np.multiply(adjacency, activity)) )
        
    #Creating network data
    edge_freq = np.count_nonzero(edges, axis = 2) / n      
    edge_width = np.sum(edges, axis = 2) / n 
    
    endt = time.time()
    print("Time for this n run through is: "+str(endt-startt))
    
    
    time_log.append(endt-startt)
    total_time = sum(time_log)
    print("Total run time is: " + str(total_time))
    return UrbanY, edge_freq, edge_width





#-----------------------------------------------------------------------------
#Working script


#1a) Load data to run sims
lsoa_data = load_obj("lsoa_data")


# Simulation Runs
# ms = np.logspace(np.log10(1),np.log10(3), num=9)
# ms = np.insert(ms, 0,0)
# ms = np.round(ms, 3)
# #

# UrbanYs = []
# edge_freqs = []
# edge_widths = []
# for m in ms:
#     UrbanY, edge_freq, edge_width = monte_carlo_runs(m, 1000, lsoa_data['s'], lsoa_data['idx'], lsoa_data['sheff_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios'])
#     UrbanYs.append(UrbanY)
#     edge_freqs.append(edge_freq)
#     edge_width = (edge_width - edge_width.min()) / (edge_width.max() - edge_width.min())
#     edge_widths.append(edge_width)

# normal = {
#     "UrbanYs": UrbanYs,
#     "edge_freqs": edge_freqs,
#     "edge_widths": edge_widths,
#     "m_values": ms
#     }
# save_obj(normal, "0_3_res10_1000runs_normal")


# UrbanYs = []
# edge_freqs = []
# edge_widths = []
# for m in ms:
#     UrbanY, edge_freq, edge_width = monte_carlo_runs(m, 1000, lsoa_data['s'], lsoa_data['idx'], lsoa_data['sheff_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios'], is_shuffled = 1)
#     UrbanYs.append(UrbanY)
#     edge_freqs.append(edge_freq)
#     edge_width = (edge_width - edge_width.min()) / (edge_width.max() - edge_width.min())
#     edge_widths.append(edge_width)
# shuffled = {
#     "UrbanYs": UrbanYs,
#     "edge_freqs": edge_freqs,
#     "edge_widths": edge_widths,
#     "m_values": ms
#     }
# save_obj(shuffled, "0_3_res10_1000runs_shuffled")



# #stability of montecarlo
# ms = [1,1,1,1,1]
# UrbanYs = []
# edge_freqs = []
# edge_widths = []
# for m in ms:
#     UrbanY, edge_freq, edge_width = monte_carlo_runs(m, 1000, lsoa_data['s'], lsoa_data['idx'], lsoa_data['sheff_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios'])
#     UrbanYs.append(UrbanY)
#     edge_freqs.append(edge_freq)
#     edge_width = (edge_width - edge_width.min()) / (edge_width.max() - edge_width.min())
#     edge_widths.append(edge_width)

# normal = {
#     "UrbanYs": UrbanYs,
#     "edge_freqs": edge_freqs,
#     "edge_widths": edge_widths,
#     "m_values": ms
#     }
# save_obj(normal, "stability_normal")


# UrbanYs = []
# edge_freqs = []
# edge_widths = []
# for m in ms:
#     UrbanY, edge_freq, edge_width = monte_carlo_runs(m, 1000, lsoa_data['s'], lsoa_data['idx'], lsoa_data['sheff_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios'], is_shuffled = 1)
#     UrbanYs.append(UrbanY)
#     edge_freqs.append(edge_freq)
#     edge_width = (edge_width - edge_width.min()) / (edge_width.max() - edge_width.min())
#     edge_widths.append(edge_width)
# shuffled = {
#     "UrbanYs": UrbanYs,
#     "edge_freqs": edge_freqs,
#     "edge_widths": edge_widths,
#     "m_values": ms
#     }
# save_obj(shuffled, "stability_shuffled")



#------------------------------------------------------------------------------

params = {'font.family':'serif',
        'axes.labelsize':'small',
        'xtick.labelsize':'x-small',
        'ytick.labelsize':'x-small',
#        
        'lines.markersize': 10,
        'scatter.marker': 'o',
#        
        'legend.fontsize':'small',
        'legend.title_fontsize':'small',
        'legend.fancybox': True,
        'legend.framealpha': 0.5,
        'legend.shadow': False,
        'legend.frameon': True,
#        
#
        'grid.linestyle':'--',
        'grid.linewidth':'0.5',
        'lines.linewidth':'0.5'}
plt.rcParams.update(params)

# #3) Data visualisation of results
#stability
def stability_graph():
    #ms = [1,1,1,1,1]
    normal = load_obj("stability_normal")
    shuffled = load_obj("stability_shuffled")
    norms = np.stack(normal['UrbanYs'])
    norms = np.reshape(norms, 5000)
    shuffs = np.stack(shuffled['UrbanYs'])
    shuffs = np.reshape(shuffs, 5000)   
    result = densratio(norms, shuffs)
    density_ratio = result.compute_density_ratio(shuffs)
    plt.plot(shuffs, density_ratio, "o")
    plt.xlabel("normal")
    plt.ylabel("Density Ratio")
    plt.show()
    # fig, ax = plt.subplots(1,1)
    # #ax.plot( )
    # ax.set_ylabel('Normal/Shuffled Geography')
    # ax.set_xlabel('Iteration No.')
    # ax.set_ylim(0.9,1.1)
    # ax.set_title('Stability of m = 1')
    # ax.locator_params(axis = 'both',tight=True, nbins=5)
    return density_ratio
density_ratio = stability_graph()



#normal = load_obj("0_3_res3_1000runs_normal")
#shuffled = load_obj("0_3_res3_1000runs_shuffled")




#4) Creating graphs
def plot_graphs_sheffield(sheff_shape, m_no, edge_freqs, edge_widths):
    
    centroid = sheff_shape.centroid
    nodesx = centroid.x.tolist()
    nodesy = centroid.y.tolist()
    nodes = list(zip(nodesx, nodesy))
    edges = np.nonzero(edge_freqs[m_no])
    edges = list(zip(edges[0], edges[1]))
    
    G = nx.house_graph()
    
    for i in range(len(nodes)):
        G.add_node(i, pos = nodes[i])
    
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1], alpha = edge_freqs[m_no][edges[i]])
    
    base = lsoa_data['sheff_shape'].plot(color='white', edgecolor='black')
    nx.draw_networkx_nodes(G, nodes, node_size = 10, ax = base)
    
    #for i in range(len(edges)):
    #    nx.draw_networkx_edges(G, nodes[edges[i]], width = edge_widths[m_no][edges[i]], alpha = edge_freqs[m_no][edges[i]], ax= base)
        
    nx.draw_networkx_edges(G, nodes, width = [edge_widths[m_no][i] for i in edges], alpha = 0.5, ax= base)
    return

#plot_graphs_sheffield(lsoa_data['sheff_shape'], 0, shuffled['edge_freqs'], shuffled['edge_widths'])