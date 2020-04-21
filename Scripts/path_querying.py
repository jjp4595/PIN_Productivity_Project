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


