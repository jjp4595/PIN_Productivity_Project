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


def run_shortest_path(sheff_shape, graph_proj):
    """
    Running shortest path function
    """
    paths_matrix = np.zeros((len(sheff_shape), len(sheff_shape)))

    for i in range(len(sheff_shape)): 
        
        for j in range(len(sheff_shape)): #Loop across target OAs to create 
            if i==j:
                paths_matrix[i,j] = 0
            else:
                point1s = get_random_point_in_polygon(sheff_shape['geometry'][i])
                point2s = get_random_point_in_polygon(sheff_shape['geometry'][j])
                paths_matrix[i,j] = shortestpath_oa(point1s,point2s, graph_proj)  
      
    return paths_matrix



#to run scripts
def load_graph():
    graph_proj = ox.save_load.load_graphml("sheff_driveable.graphml", folder = os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\scripts_misc"))
    return graph_proj

sheff_shape = gpd.read_file(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\1 Data\1 Data\CDRC\Census Data Pack\Sheffield\shapefiles\Sheffield_lsoa11.shp"))
graph_proj = load_graph()


sheff_shape = sheff_shape.iloc[0:20]

all_paths = []
for i in range(1):
    all_paths.append(run_shortest_path(sheff_shape, graph_proj))