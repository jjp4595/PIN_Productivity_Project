"""
Path Querying
"""
import geopandas as gpd
import os
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, Point
import random
import networkx as nx
from pyproj import CRS
import pandas as pd
import multiprocessing
from time import time
import pickle


#Graph
def get_graph():
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
def save_graph():
    ox.save_load.save_graphml(graph_proj, filename='sheff_driveable.graphml', folder = os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\scripts_misc"))



def get_random_point_in_polygon(poly):
     minx, miny, maxx, maxy = poly.bounds
     while True:
         p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         if poly.contains(p):
             return p


def shortestpath_oa(point1,point2, graph_proj):
    """


    """

    #Getting points to travel between
    orig_xy = (point1.y, point1.x)
    target_xy = (point2.y, point2.x)
    
    
    orig_node = ox.get_nearest_node(graph_proj, orig_xy, method='euclidean')
    target_node = ox.get_nearest_node(graph_proj, target_xy, method='euclidean')
    
    
    #o_closest = nodes_proj.loc[orig_node]
    #t_closest = nodes_proj.loc[target_node] 
    #od_nodes = gpd.GeoDataFrame([o_closest, t_closest], geometry='geometry', crs=nodes.crs)
    
    #calculate the shortest path length
    try:
        return nx.shortest_path_length(G=graph_proj, source=orig_node, target=target_node, weight='length')
    except nx.NetworkXNoPath:
        return 1e8


def run_shortest_path(paths, sheff_shape, graph_proj):
    """
    Running shortest path function
    """
    
    for i in range(paths.shape[0]):         
        for j in range(paths.shape[1]): #Loop across target OAs to create 
            if paths.index[i]==paths.columns[j]:
                pass
            else:
                point1s = get_random_point_in_polygon(sheff_shape['geometry'][paths.index[i]])
                point2s = get_random_point_in_polygon(sheff_shape['geometry'][paths.columns[j]])
                paths.iloc[i,j] = shortestpath_oa(point1s,point2s, graph_proj)       
    return paths



#to run scripts
def load_graph():
    graph_proj = ox.save_load.load_graphml("sheff_driveable.graphml", folder = os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\scripts_misc"))
    return graph_proj


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


#actual scripts
sheff_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))    
#sheff_shape = sheff_shape.iloc[0:10,:]
graph_proj = load_graph()

paths_dataframe = pd.DataFrame(np.zeros((len(sheff_shape), len(sheff_shape))))


no_scripts = multiprocessing.cpu_count() #when running on personal

#no_scripts = 16
paths_splits = np.array_split(paths_dataframe, no_scripts)
                              

#no paralellising
# t1=time()
# all_paths = []
# for i in range(no_scripts):
#     all_paths.append(run_shortest_path(paths_splits[i], sheff_shape, graph_proj))
# all_paths = pd.concat(all_paths)
# t2=time()
# print(t2-t1)


if __name__ == '__main__':
    t1 = time()
    args = []    
    for i in range(no_scripts):
        args.append((paths_splits[i], sheff_shape, graph_proj))
    
    
    with multiprocessing.Pool(processes=no_scripts) as pool:
        all_paths = pool.starmap(run_shortest_path, args)
        all_paths = pd.concat(all_paths)
    t2 = time()
    print(t2-t1)
    save_obj(all_paths, 'all_paths')