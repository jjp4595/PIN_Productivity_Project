# -*- coding: utf-8 -*-
"""
Network analysis
"""
import pickle
import networkx as nx
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os



def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#------------------------------------------------------------------------------
#
#                   Analysis Sections...
#
#------------------------------------------------------------------------------
lsoa_data = load_obj("lsoa_data")
normal = load_obj("normal_ms_0_25_8res_1000run")
shuffled = load_obj("shuffled_ms_0_25_8res_1000run")



#Density ratio estimation
def density_ratio(first, second):
    ratios = [a/b for a in first for b in second]
    sorted_ratios = np.sort(ratios)
    median_50 = sorted_ratios[int((len(ratios) * 0.5))]
    ratio_25 = sorted_ratios[int((len(ratios) * 0.25))]
    ratio_75 = sorted_ratios[int((len(ratios) * 0.75))]
    mean, sigma = np.mean(ratios), np.std(ratios)
    conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma / np.sqrt(len(ratios)))
    #fig, ax = plt.subplots(1,1)
    #ax.hist(ratios, bins = 20 , density=True)
    info = {
        "sorted ratios":sorted_ratios,
        "median": median_50,
        "25th":ratio_25, 
        "75th":ratio_75,
        "mean":mean,
        "sigma":sigma, 
        "conf_int":conf_int
        }
    
    return info

# #-----------------------------------------------------------------------------


params = {'font.family':'serif',
        'axes.labelsize':'small',
        'xtick.labelsize':'x-small',
        'ytick.labelsize':'x-small', 
        
        'legend.fontsize':'small',
        'legend.title_fontsize':'small',
        'legend.fancybox': True,
        'legend.framealpha': 0.5,
        'legend.shadow': False,
        'legend.frameon': True,
        
        'grid.linestyle':'--',
        'grid.linewidth':'0.5',
        'lines.linewidth':'0.5'}
plt.rcParams.update(params)


# fig, [ax0, ax1, ax2] = plt.subplots(1,3)
# fig.set_size_inches(9, 4)
# ax0.boxplot([norm_m0['sorted ratios'], shuff_m0['sorted ratios']], labels = ['Normal', 'Shuffled'], notch=True, showfliers=False, showmeans=True)
# ax0.set_title('m = 0')
# ax0.set_xlabel('Geographical Layout', labelpad = 20)
# ax0.set_ylabel('Density ratios compared to baseline')
# ax1.boxplot([norm_m1['sorted ratios'], shuff_m1['sorted ratios']], labels = ['Normal', 'Shuffled'], notch=True, showfliers=False, showmeans=True)
# ax1.set_title('m = 1')
# ax1.set_xlabel('Geographical Layout', labelpad = 20)
# ax2.boxplot([norm_m2['sorted ratios'], shuff_m2['sorted ratios']], labels = ['Normal', 'Shuffled'], notch=True, showfliers=False, showmeans=True)
# ax2.set_title('m = 2')
# ax2.set_xlabel('Geographical Layout', labelpad = 20)
# plt.tight_layout()
# fig.savefig(os.environ['USERPROFILE'] + r'\Dropbox\PIN\workshop\13-05\boxplots.png', format = 'png')


def productivitygraph():
    #3 x 3 ms 0 - 4
    fig, axs = plt.subplots(2,4)
    fig.set_size_inches(6.39,4)
    axs=axs.ravel()
    baseline = 0 #density ratio for comparison. 
    for i in range(len(normal['m_values'])):
        base = normal['UrbanYs'][baseline]
        norm = density_ratio(normal['UrbanYs'][i], base)
        shuff = density_ratio(shuffled['UrbanYs'][i], base)
        axs[i].boxplot([norm['sorted ratios'], shuff['sorted ratios']], labels = ['Normal', 'Shuffled'], notch=True, showfliers=False, showmeans=True)
        axs[i].set_title('m = ' + str(round(normal['m_values'][i],2)))
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\reports\m_sensitivity.png"), dpi=100, format = 'png')
#productivitygraph()





def layoutgraphs():
    from synthetic_network import paths_shuffle
    centroid = lsoa_data['sheff_shape'].centroid
    nodesx = centroid.x.tolist()
    nodesy = centroid.y.tolist()
    nodesx = np.asarray(nodesx)
    nodesy = np.asarray(nodesy)
    east_inds = np.argsort(nodesx)
    
    
    means = paths_shuffle(lsoa_data['sheff_shape'], lsoa_data['income_params'])
    means = np.asarray(means)
    
    means_norm = np.divide(np.subtract(means, means.min()),
                                np.subtract(means.max(), means.min())) 
    
    income_inds = np.argsort(means_norm) #sorting means 
    
    
    
    #sense check
    plt.scatter(nodesx[east_inds], means_norm[income_inds])
    
    
    cNorm  = colors.Normalize(vmin=0, vmax=np.max(means_norm))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='gray')
    # colorList = []
    # for i in range(len(income_inds)):
    #    colorList.append(scalarMap.to_rgba(means_norm[i]))
    
    
    
    fig, [base2, base] = plt.subplots(1,2)
    fig.set_size_inches(7,2.5)
    lsoa_data['sheff_shape'].plot(color='white', edgecolor='black', ax=base)
    base.scatter(nodesx[east_inds], nodesy[east_inds], c=means_norm[income_inds], s=8., cmap='gray')
    divider = make_axes_locatable(base)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(scalarMap, cax=cax)
    plt.tight_layout()
    lsoa_data['sheff_shape'].plot(color='white', edgecolor='black', ax=base2)
    base2.scatter(nodesx, nodesy, c=means_norm, s=8., cmap='gray')
    divider = make_axes_locatable(base2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(scalarMap, cax=cax)
    #cax.set_ylabel('normalised income')
    base.set_xticks([])
    base.set_yticks([])
    base2.set_xticks([])
    base2.set_yticks([])
    
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\reports\layouts.png"), dpi=100, format = 'png')




#4) Creating network graphs
def networkgraph(layout, m, ax_in):
    
    centroid = lsoa_data['sheff_shape'].centroid
    nodesx = centroid.x.tolist()
    nodesy = centroid.y.tolist()
    nodes = list(zip(nodesx, nodesy))
    edges = np.nonzero(layout['edge_freqs'][m])
    edges = list(zip(edges[0], edges[1]))
    edge_freqs = [layout['edge_freqs'][m][i,j] for i in range(len(layout['edge_freqs'][m])) for j in range(len(layout['edge_freqs'][m]))]
    edge_widths = [layout['edge_widths'][m][i,j] for i in range(len(layout['edge_widths'][m])) for j in range(len(layout['edge_widths'][m]))]
    
    
    
    
    nodes2 = list(zip([i for i in range(len(nodes))], nodesx, nodesy))
    def Convert(tup, di): 
        for a, b, c in tup: 
            #di.setdefault(a, []).append((b,c)) 
            di.setdefault(a, (b,c)) 
        return di 
    di={}
    di = Convert(nodes2, di)
    edge_widths_map = np.divide(np.subtract(np.asarray(edge_widths), np.asarray(edge_widths).min()),
                                np.subtract(np.asarray(edge_widths).max(), np.asarray(edge_widths).min()))
    # Add a color_map for the edges
    cNorm  = colors.Normalize(vmin=0, vmax=np.max(edge_widths_map))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.cm.Greys)
    colorList = []
    for i in range(len(edges)):
       colorList.append(scalarMap.to_rgba(edge_widths_map[i]))
    
    
    
    
    G = nx.house_graph()
    
    for i in range(len(nodes)):
        G.add_node(i, pos = nodes[i])
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1])
    
    
    lsoa_data['sheff_shape'].plot(color='white', edgecolor='black', ax=ax_in)
    # nx.drawing.nx_pylab.draw_networkx(G, di, node_colour='b', node_size=5, width=edge_freqs*100, edge_color=colorList,  with_labels=False, ax=base)
    nodes_dr = nx.draw_networkx_nodes(G, di, node_colour = 'b', node_size=5, with_labels=False, ax = ax_in)
    edges_dr = nx.draw_networkx_edges(G, di, width=edge_freqs*100, edge_color=colorList, ax=ax_in)
    #plt.colorbar(scalarMap, ax=ax_in)
    divider = make_axes_locatable(ax_in)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_ylabel('edge strength')
    plt.colorbar(scalarMap, cax=cax)



fig, [[ax0, ax1], [ax1a, ax1b], [ax2, ax3]] = plt.subplots(3,2)
fig.set_size_inches(6.39, 6)
networkgraph(normal, 0, ax0)
networkgraph(shuffled, 0, ax1)
networkgraph(normal, 2, ax1a)
networkgraph(shuffled, 2, ax1b)
networkgraph(normal, 6, ax2)
networkgraph(shuffled, 6, ax3)
fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\reports\network_show.png"), dpi=100, format = 'png')


