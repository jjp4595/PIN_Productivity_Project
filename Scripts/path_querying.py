"""
Path Querying
"""
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

from shapely.geometry import Polygon, Point
import random
import networkx as nx
from pyproj import CRS
import time




def get_random_point_in_polygon(poly):
     minx, miny, maxx, maxy = poly.bounds
     while True:
         p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         if poly.contains(p):
             return p


def shortestpath_oa(point1,point2, graph_proj):
    """
    

    Parameters
    ----------
    point1 : int
        DESCRIPTION.
    point2 : int
        DESCRIPTION.
    graph_proj : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

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


#-----------------------------------------------------------------------------
if __name__ == "__main__":
   
    #Importing shape files
    sheff_OShighways = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_OA.shp"))
    sheff = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\Workshop_2_data\Sheffield_Network.gdb"))
    sheff_lsoa_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))
    
    
    
    #drivable streets in sheffield
    place_name = "Sheffield, UK"
    graph = ox.graph_from_place(place_name, network_type = 'drive')
    fig, ax = ox.plot_graph(graph)
    
    
    #Convert graph int geodataframe 
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    print("Coordinate system:", edges.crs)
    
    #convert data into same co-ordinate system as lsoa data
    crs = CRS.from_string("epsg:27700")
    graph_proj = ox.project_graph(graph, to_crs=crs)
    fig, ax = ox.plot_graph(graph_proj)
    
    #Convert graph into UTM zone 30 format (with m units)
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    print("Coordinate system:", edges_proj.crs)
    
    
    
    # Creating the distributions of shortest paths
    s = 50 #samples 
    paths_matrix = np.zeros((s, s))
    
    idx = np.round(np.linspace(0, len(sheff_lsoa_shape) - 1, s)).astype(int) #Indexing s spaced from array
    
    t = []
    for i in range(len(idx)):
        t.append(time.time())
        for j in range(len(idx)):
            point1 = get_random_point_in_polygon(sheff_lsoa_shape['geometry'][idx[i]])
            point2 = get_random_point_in_polygon(sheff_lsoa_shape['geometry'][idx[j]])
            if i==j:
                paths_matrix[i,j] = 0
            else:
                paths_matrix[i,j] = shortestpath_oa(point1,point2)
        t[i] = time.time() - t[i]
        print(t[i])
        
                
    paths = np.concatenate(paths_matrix, axis=0)
    paths = paths[paths != 0]
    med_paths = sorted(paths[paths!=0])
    med_paths = med_paths[int(len(med_paths)/2)]
    
    fig, ax = plt.subplots(1,1)
    ax.hist(paths, bins = 25)
