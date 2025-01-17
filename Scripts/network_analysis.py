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
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pandas as pd
import scipy.io as sio
import powerlaw

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

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
        
        'legend.fontsize':'x-small',
        'legend.title_fontsize':'x-small',
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
sns.set_palette("deep")
sns.set_color_codes()

def income_edu_dists():
    
    #Graphing data
    lsoa_data = load_obj("lsoa_data")
    sns.set_palette("deep")
    sns.set_color_codes()
    maxid, minid = 103, 337 #income means max/min lsoa id

    #Income Distributions   
    fig=plt.figure()
    fig.set_size_inches(4,3)
    gs = gridspec.GridSpec(2, 2, height_ratios=[10,1], wspace=0.25, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2], sharex = ax0)
    inc1 = stats.beta.rvs(lsoa_data['income_params'][minid,0], lsoa_data['income_params'][minid,1], loc =lsoa_data['income_params'][minid,2], scale = lsoa_data['income_params'][minid,3], size = 1000)
    sns.distplot(inc1, color='b', kde=False, norm_hist=True, ax=ax0)
    ax0.plot(np.linspace(0,1,100),  stats.beta.pdf(np.linspace(0,1,100), lsoa_data['income_params'][minid, 0], lsoa_data['income_params'][minid, 1], loc = lsoa_data['income_params'][minid, 2], scale = lsoa_data['income_params'][minid, 3]) , label = 'CDF')
    ax0.axvline(x=np.median(inc1), ls='--', lw=2)
    sns.pointplot(data = inc1, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.set_xlim(0,1)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Normalised income')

    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3], sharex = ax2)
    inc2 = stats.beta.rvs(lsoa_data['income_params'][maxid,0], lsoa_data['income_params'][maxid,1], loc =lsoa_data['income_params'][maxid,2], scale = lsoa_data['income_params'][maxid,3], size = 1000)
    sns.distplot(inc2, color='b', kde=False, norm_hist=True, ax=ax2)
    ax2.plot(np.linspace(0,1,100),  stats.beta.pdf(np.linspace(0,1,100), lsoa_data['income_params'][maxid, 0], lsoa_data['income_params'][maxid, 1], loc = lsoa_data['income_params'][maxid, 2], scale = lsoa_data['income_params'][maxid, 3]) , label = 'CDF')
    ax2.axvline(x=np.median(inc2), ls='--', lw=2)
    sns.pointplot(data = inc2, ci = 'sd', orient = 'h', scale=2, ax=ax3)
    ax2.set_xlim(0,1)
    ax2.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax3.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax3.set_xlabel('Normalised income')
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\income_dists.pdf"), format = 'pdf')

    #Education Distributions
    fig, [ax0, ax1] = plt.subplots(1,2)
    fig.set_size_inches(4,3)
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
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\education_dists.pdf"), format = 'pdf')
    
    #Attractivity graphs
    fig, [ax0, ax1] = plt.subplots(1,2)
    fig.set_size_inches(4,3)
    attractivity1=inc1**(-edu1)   
    attractivtiy1_powerlaw = powerlaw.Fit(attractivity1)
    attractivtiy1_powerlaw_temp = powerlaw.Fit(attractivity1, xmin=1)
    powerlaw_plot=attractivtiy1_powerlaw_temp.plot_ccdf(original_data=True, color='b', marker='.', ms=2, lw=0, alpha=0.5, ax=ax0)
    X=attractivtiy1_powerlaw_temp.ccdf()
    x=X[0][X[0]>=attractivtiy1_powerlaw.xmin]
    y=[np.exp( (-(attractivtiy1_powerlaw.alpha-1)*(np.log(i)-np.log(attractivtiy1_powerlaw.xmin))+np.log(X[1][X[0]==attractivtiy1_powerlaw.xmin])) ) for i in x]
    ax0.plot(x, y, 'k')
    ax0.set_ylabel(r'$P(attractivity \geq x)$')
    ax0.set_xlabel(r'$attractivity$')
    
    attractivity2=inc2**(-edu2)   
    attractivtiy2_powerlaw = powerlaw.Fit(attractivity2)
    attractivtiy2_powerlaw_temp = powerlaw.Fit(attractivity2, xmin=1)
    powerlaw_plot=attractivtiy2_powerlaw_temp.plot_ccdf(original_data=True, color='b', marker='.', ms=2, lw=0, alpha=0.5, ax=ax1)
    X=attractivtiy2_powerlaw_temp.ccdf()
    x=X[0][X[0]>=attractivtiy2_powerlaw.xmin]
    y=[np.exp( (-(attractivtiy2_powerlaw.alpha-1)*(np.log(i)-np.log(attractivtiy2_powerlaw.xmin))+np.log(X[1][X[0]==attractivtiy2_powerlaw.xmin])) ) for i in x]
    ax1.plot(x, y, 'k')
    ax1.set_ylabel(r'$P(attractivity \geq x)$')
    ax1.set_xlabel(r'$attractivity$')
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\attractivity.pdf"), format = 'pdf')
    
    #centroid/driveable path distributions
    paths_matrix = load_obj("ave_paths")
    paths = paths_matrix.flatten()
    centroid_info = load_obj("centroids_beta_params_centroiddists")
    centroid_paths = np.vstack(centroid_info['euclidean_dists']).flatten()
    
    fig=plt.figure()
    fig.set_size_inches(6,3)
    gs = gridspec.GridSpec(2, 3, height_ratios=[10,1], width_ratios=[1, 1, 1], wspace=0.35, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[3], sharex = ax0)
    sns.distplot(centroid_paths, color='b', kde=False,  ax=ax0)
    ax0.axvline(x=np.median(centroid_paths), ls='--', lw=2)
    sns.pointplot(data = centroid_paths, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.set_xlim(0,30000)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Centroid path distances')    

    ax0 = fig.add_subplot(gs[1])
    ax1 = fig.add_subplot(gs[4], sharex = ax0)
    sns.distplot(paths, color='b', kde=False,  ax=ax0)
    ax0.axvline(x=np.median(paths), ls='--', lw=2)
    sns.pointplot(data = paths, ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.set_xlim(0,30000)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Driveable path distance')

    ax0 = fig.add_subplot(gs[2])
    ax1 = fig.add_subplot(gs[5], sharex = ax0)
    diffs = paths-centroid_paths
    neg_diffs = abs(diffs[diffs<0])
    sns.distplot(neg_diffs, color='r', kde=False, hist_kws=dict(alpha=0.5), ax=ax0)
    ax0.axvline(x=np.median(neg_diffs), ls='--', lw=2, c='r')
    sns.pointplot(data = neg_diffs, ci = 'sd', orient = 'h', scale=2, color = 'r', ax=ax1)
    pos_diffs=diffs[diffs>0]
    sns.distplot(pos_diffs, color='b', kde=False, hist_kws=dict(alpha=0.5), ax=ax0)
    ax0.axvline(x=np.median(pos_diffs), ls='--', lw=2)
    sns.pointplot(data = pos_diffs, ci = 'sd', orient = 'h', scale=2, ax=ax1)   
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax0.set_xlim(0,8000)
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Difference in distances')
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\driveable_centroids_paths.pdf"), format = 'pdf')

    
def plotgraph():
    import osmnx as ox
    import path_querying
    G1 = path_querying.load_graph()

    fig, ax = ox.plot.plot_graph(G1, dpi = 600, node_color = 'None', fig_width=6, fig_height=6)
    fig.set_size_inches(4,4)
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\driveable_sheffield_graph.pdf"), format = 'pdf')

#Old function to sense check the shuffling was doing as intended
# def layoutgraphs():
#     from synthetic_network import paths_shuffle
#     centroid = lsoa_data['sheff_lsoa_shape'].centroid
#     nodesx = centroid.x.tolist()
#     nodesy = centroid.y.tolist()
#     nodesx = np.asarray(nodesx)
#     nodesy = np.asarray(nodesy)
#     east_inds = np.argsort(nodesx)
    
    
#     means = paths_shuffle(lsoa_data['sheff_lsoa_shape'], lsoa_data['income_params'])
#     means = np.asarray(means)
    
#     means_norm = np.divide(np.subtract(means, means.min()),
#                                 np.subtract(means.max(), means.min())) 
    
#     income_inds = np.argsort(means_norm) #sorting means 
    

#     #sense check
#     plt.scatter(nodesx[east_inds], means_norm[income_inds])
    
    
#     cNorm  = colors.Normalize(vmin=0, vmax=np.max(means_norm))
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='gray')
#     # colorList = []
#     # for i in range(len(income_inds)):
#     #    colorList.append(scalarMap.to_rgba(means_norm[i]))
    
    
    
#     fig, [base2, base] = plt.subplots(1,2)
#     fig.set_size_inches(7,2.5)
#     lsoa_data['sheff_lsoa_shape'].plot(color='white', edgecolor='black', ax=base)
#     base.scatter(nodesx[east_inds], nodesy[east_inds], c=means_norm[income_inds], s=8., cmap='gray')
#     divider = make_axes_locatable(base)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(scalarMap, cax=cax)
#     plt.tight_layout()
#     lsoa_data['sheff_lsoa_shape'].plot(color='white', edgecolor='black', ax=base2)
#     base2.scatter(nodesx, nodesy, c=means_norm, s=8., cmap='gray')
#     divider = make_axes_locatable(base2)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(scalarMap, cax=cax)
#     #cax.set_ylabel('normalised income')
#     base.set_xticks([])
#     base.set_yticks([])
#     base2.set_xticks([])
#     base2.set_yticks([])
    
#     plt.tight_layout()
#     fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\reports\layouts.png"), dpi=100, format = 'png')

#Creating network graphs
def networkgraph(layout, m, ax_in):
    
    centroid = lsoa_data['sheff_lsoa_shape'].centroid
    nodesx = centroid.x.tolist()
    nodesy = centroid.y.tolist()
    nodes = list(zip(nodesx, nodesy))
    

    
    freq_ind = np.nonzero(layout['edge_freqs'][m])
    edge_freqs = layout['edge_freqs'][m][freq_ind]
    wid_ind = np.nonzero(layout['edge_widths'][m])
    edge_widths = layout['edge_widths'][m][wid_ind]
    
    edges = np.nonzero(layout['edge_freqs'][m])
    edges = list(zip(edges[0], edges[1]))
    
    
    edge_widths = np.log10(edge_widths)#taking logarithm prior to norm
    
    edge_freqs = (edge_freqs - edge_freqs.min()) / (edge_freqs.max() - edge_freqs.min()) #normalising freqs
    edge_widths_map = (edge_widths - edge_widths.min()) / (edge_widths.max() - edge_widths.min()) #normalising strength

    nodes2 = list(zip([i for i in range(len(nodes))], nodesx, nodesy))
    def Convert(tup, di): 
        for a, b, c in tup: 
            #di.setdefault(a, []).append((b,c)) 
            di.setdefault(a, (b,c)) 
        return di 
    di={}
    di = Convert(nodes2, di)
    
    # Add a color_map for the edges
    cNorm  = colors.Normalize(vmin=0, vmax=np.max(edge_widths_map))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.cm.Greys)
    colorList = []
    for i in range(len(edges)):
       colorList.append(scalarMap.to_rgba(edge_widths_map[i]))
    
    Ninds = np.argsort(edge_freqs)[-int(len(edge_freqs)*0.1):] #strongest 10% edges
    edge_freqs = edge_freqs[Ninds]
    colorList = [colorList[i] for i in Ninds]
    edges = [edges[i] for i in Ninds]
    
    G = nx.house_graph()
    
    for i in range(len(nodes)):
        G.add_node(i, pos = nodes[i])
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1])
    
    
    lsoa_data['sheff_lsoa_shape'].plot(color='white', edgecolor='black', ax=ax_in)
    # nx.drawing.nx_pylab.draw_networkx(G, di, node_colour='b', node_size=5, width=edge_freqs*100, edge_color=colorList,  with_labels=False, ax=base)
    #nodes_dr = nx.draw_networkx_nodes(G, di, node_colour = 'k', node_size=4, with_labels=False, ax = ax_in)
    edges_dr = nx.draw_networkx_edges(G, di, width=edge_freqs, edge_color=colorList, alpha = 0.5, ax=ax_in)
    #plt.colorbar(scalarMap, ax=ax_in)
    divider = make_axes_locatable(ax_in)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_ylabel('edge strength')
    plt.colorbar(scalarMap, cax=cax)


#Loading data ----------------------------------------------------------------
lsoa_data = load_obj("lsoa_data")
normal = load_obj("normal_layout_2000run")
minmax = load_obj("minmax_layout_2000run")
slmin = load_obj("slmin_layout_2000run")
od50 = load_obj("od50_layout_2000run")
#-----------------------------------------------------------------------------


#Plot Network figures
fig, ax0 = plt.subplots(1,1)
fig.set_size_inches(5.8,3.8)
networkgraph(normal, 0, ax0)
plt.tight_layout()
ax0.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\network_normal_m_0_5.pdf"), format = 'pdf')

fig, ax0 = plt.subplots(1,1)
fig.set_size_inches(5.8,3.8)
networkgraph(normal, 1, ax0)
plt.tight_layout()
ax0.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\network_normal_m_1.pdf"), format = 'pdf')

fig, ax0 = plt.subplots(1,1)
fig.set_size_inches(5.8,3.8)
networkgraph(normal, 2, ax0)
plt.tight_layout()
ax0.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\network_normal_m_1_5.pdf"), format = 'pdf')

fig, ax0 = plt.subplots(1,1)
fig.set_size_inches(5.8,3.8)
networkgraph(normal, 3, ax0)
ax0.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\network_normal_m_2.pdf"), format = 'pdf')
#-----------------------------------------------------------------------------




#Density ratios and productivity plot ----------------------------------------
def density_ratio(first, second):#amended so it doesn't use a list comprehension and does an all-to-all matrix operation
    first = first[:,np.newaxis]
    ratios = first/second
    ratios = ratios.flatten()
    sorted_ratios = np.sort(ratios)
    median_50 = sorted_ratios[int((len(ratios) * 0.5))]
    ratio_25 = sorted_ratios[int((len(ratios) * 0.25))]
    ratio_75 = sorted_ratios[int((len(ratios) * 0.75))]
    mean, sigma = np.mean(ratios), np.std(ratios)
    conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma / np.sqrt(len(ratios)))
    #fig, ax = plt.subplots(1,1)
    #ax.hist(ratios, bins = 20 , density=True)
    info = {
        "sorted_ratios":sorted_ratios,
        "median": median_50,
        "25th":ratio_25, 
        "75th":ratio_75,
        "mean":mean,
        "sigma":sigma, 
        "conf_int":conf_int
        }
    
    return info
#Create density ratios
normal['ratios'], minmax['ratios'], slmin['ratios'], od50['ratios'] = [], [], [], []
base = np.asarray(normal['UrbanYs'][1])
for i in range(len(normal['m_values'])):
    normal['ratios'].append(density_ratio(np.asarray(normal['UrbanYs'][i]), base))
    minmax['ratios'].append(density_ratio(np.asarray(minmax['UrbanYs'][i]), base))
    slmin['ratios'].append(density_ratio(np.asarray(slmin['UrbanYs'][i]), base))
    od50['ratios'].append(density_ratio(np.asarray(od50['UrbanYs'][i]), base))
    
dfs = []
for i in range(len(normal['m_values'])):
    dfs.append(pd.DataFrame(data = {'Normal':normal['ratios'][i]['sorted_ratios'], 
                                    'Minmax':minmax['ratios'][i]['sorted_ratios'],
                                    'slmin':slmin['ratios'][i]['sorted_ratios'],
                                    'od50':od50['ratios'][i]['sorted_ratios']}))    

fig, [ax0, ax1, ax2, ax3] = plt.subplots(4,1)
fig.set_size_inches(6,8)
ax0.set_title('m = 0.5')
ax1.set_title('m = 1')
ax2.set_title('m = 1.5')
ax3.set_title('m = 2')
sns.boxplot(data = dfs[0], notch=True, showfliers=False, showmeans=True, palette = 'Blues', ax=ax0)
sns.boxplot(data = dfs[1], notch=True, showfliers=False, showmeans=True, palette = 'Blues', ax=ax1)
sns.boxplot(data = dfs[2], notch=True, showfliers=False, showmeans=True, palette = 'Blues', ax=ax2)
sns.boxplot(data = dfs[3], notch=True, showfliers=False, showmeans=True, palette = 'Blues', ax=ax3)
# sns.pointplot(data = dfs[0], ci = 'sd', orient = 'v', join = False, ax= ax0)
# sns.pointplot(data = dfs[1], ci = 'sd', orient = 'v', join = False, ax= ax1)
# sns.pointplot(data = dfs[2], ci = 'sd', orient = 'v', join = False, ax= ax2)
ax0.axhline(y=dfs[0]['Normal'].median(), ls='--', lw=1.5)
ax1.axhline(y=dfs[1]['Normal'].median(), ls='--', lw=1.5)
ax2.axhline(y=dfs[2]['Normal'].median(), ls='--', lw=1.5)
ax3.axhline(y=dfs[3]['Normal'].median(), ls='--', lw=1.5)
ax0.axhline(y=dfs[0]['Normal'].mean(), ls='--', lw=1)
ax1.axhline(y=dfs[1]['Normal'].mean(), ls='--', lw=1)
ax2.axhline(y=dfs[2]['Normal'].mean(), ls='--', lw=1)
ax3.axhline(y=dfs[3]['Normal'].mean(), ls='--', lw=1)
plt.tight_layout()
fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\boxplots.pdf"), format = 'pdf')

#Tables for median and means of distributions
names = ['Normal', 'Minmax', 'slmin', 'od50']
meds05 = np.around(np.asarray([normal['ratios'][0]['median'], minmax['ratios'][0]['median'], slmin['ratios'][0]['median'], od50['ratios'][0]['median']]), 2)
meds1  = np.around(np.asarray([normal['ratios'][1]['median'], minmax['ratios'][1]['median'], slmin['ratios'][1]['median'], od50['ratios'][1]['median']]), 2)
meds1_5  = np.around(np.asarray([normal['ratios'][2]['median'], minmax['ratios'][2]['median'], slmin['ratios'][2]['median'], od50['ratios'][2]['median']]), 2)
meds2  = np.around(np.asarray([normal['ratios'][3]['median'], minmax['ratios'][3]['median'], slmin['ratios'][3]['median'], od50['ratios'][3]['median']]), 2)


means05 = np.around(np.asarray([normal['ratios'][0]['mean'], minmax['ratios'][0]['mean'], slmin['ratios'][0]['mean'], od50['ratios'][0]['mean']]), 2)
means1  = np.around(np.asarray([normal['ratios'][1]['mean'], minmax['ratios'][1]['mean'], slmin['ratios'][1]['mean'], od50['ratios'][1]['mean']]), 2)
means1_5  = np.around(np.asarray([normal['ratios'][2]['mean'], minmax['ratios'][2]['mean'], slmin['ratios'][2]['mean'], od50['ratios'][2]['mean']]), 2)
means2  = np.around(np.asarray([normal['ratios'][3]['mean'], minmax['ratios'][3]['mean'], slmin['ratios'][3]['mean'], od50['ratios'][3]['mean']]), 2)



def plotgraphs():
    #load paths info
    mldata = sio.loadmat('optimisedpaths.mat')
    paths_matrix = load_obj("ave_paths")
    paths = np.tril(paths_matrix) #lower half of od matrix
    paths = np.ndarray.flatten(paths)
    paths = paths[paths!=0]
    
    #------
    fig = plt.figure()
    fig.set_size_inches(4,3)
    gs = gridspec.GridSpec(2,2, height_ratios=[10,1], wspace=0.35, hspace=0, figure = fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2], sharex = ax0)
    sns.distplot(mldata['minmax'], color='b', kde=False, ax=ax0)
    ax0.set_xlim(0,30000)
    ax0.axvline(x=np.median(mldata['minmax']), ls='--', lw=2)
    sns.pointplot(data = mldata['minmax'], ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Path Distance')
    plt.tight_layout()
    
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3], sharex = ax2)
    diffs = np.subtract(paths,mldata['minmax'].reshape(59340))
    pos_diffs = diffs[diffs>0]
    sns.distplot(pos_diffs, color='b', kde=False, label = 'Expansion', ax=ax2)
    ax2.axvline(x=np.median(pos_diffs), ls='--', lw=2)
    sns.pointplot(data = pos_diffs, ci = 'sd', orient = 'h', scale=2, ax=ax3)
    neg_diffs = np.abs(diffs[diffs<0])
    sns.distplot(neg_diffs, color='r', kde=False, label = 'Contraction', ax=ax2)
    ax2.axvline(x=np.median(neg_diffs), ls='--', lw=2, color='r')
    sns.pointplot(data = neg_diffs, ci = 'sd', orient = 'h', scale=2, color = 'r', ax=ax3)
    ax2.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax2.set_xlim(0,30000)
    ax3.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax3.set_xlabel('Difference in distances')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', prop={'size':6})
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\path_differences_minmax.pdf"), format = 'pdf')
    
    #------
    fig = plt.figure()
    fig.set_size_inches(4,3)
    gs = gridspec.GridSpec(2,2, height_ratios=[10,1], wspace=0.35, hspace=0, figure = fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2], sharex = ax0)
    sns.distplot(mldata['slmin'], color='b', kde=False, ax=ax0)
    ax0.set_xlim(0,30000)
    ax0.axvline(x=np.median(mldata['slmin']), ls='--', lw=2)
    sns.pointplot(data = mldata['slmin'], ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Path Distance')
    plt.tight_layout()
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3], sharex = ax2)
    diffs = np.subtract(paths,mldata['slmin'].reshape(59340))
    pos_diffs = diffs[diffs>0]
    sns.distplot(pos_diffs, color='b', kde=False, label = 'Expansion', ax=ax2)
    ax2.axvline(x=np.median(pos_diffs), ls='--', lw=2)
    sns.pointplot(data = pos_diffs, ci = 'sd', orient = 'h', scale=2, ax=ax3)
    neg_diffs = np.abs(diffs[diffs<0])
    sns.distplot(neg_diffs, color='r', kde=False, label = 'Contraction', ax=ax2)
    ax2.axvline(x=np.median(neg_diffs), ls='--', lw=2, color='r')
    sns.pointplot(data = neg_diffs, ci = 'sd', orient = 'h', scale=2, color = 'r', ax=ax3)
    ax2.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax2.set_xlim(0,30000)
    ax3.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax3.set_xlabel('Difference in distances')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', prop={'size':6})
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\path_differences_slmin.pdf"), format = 'pdf')
    
    #------    
    fig = plt.figure()
    fig.set_size_inches(4,3)
    gs = gridspec.GridSpec(2,2, height_ratios=[10,1], wspace=0.35, hspace=0, figure = fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2], sharex = ax0)
    sns.distplot(mldata['od50'], color='b', kde=False, ax=ax0)
    ax0.set_xlim(0,30000)
    ax0.axvline(x=np.median(mldata['od50']), ls='--', lw=2)
    sns.pointplot(data = mldata['od50'], ci = 'sd', orient = 'h', scale=2, ax=ax1)
    ax0.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax1.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax1.set_xlabel('Path Distance')
    plt.tight_layout()
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3], sharex = ax2)
    diffs = np.subtract(paths,mldata['od50'].reshape(59340))
    pos_diffs = diffs[diffs>0]
    sns.distplot(pos_diffs, color='b', kde=False, label = 'Expansion', ax=ax2)
    ax2.axvline(x=np.median(pos_diffs), ls='--', lw=2)
    sns.pointplot(data = pos_diffs, ci = 'sd', orient = 'h', scale=2, ax=ax3)
    neg_diffs = np.abs(diffs[diffs<0])
    sns.distplot(neg_diffs, color='r', kde=False, label = 'Contraction', ax=ax2)
    ax2.axvline(x=np.median(neg_diffs), ls='--', lw=2, color='r')
    sns.pointplot(data = neg_diffs, ci = 'sd', orient = 'h', scale=2, color = 'r', ax=ax3)
    ax2.tick_params(axis='both', which='both',bottom=False, labelbottom=False)   
    ax2.set_xlim(0,30000)
    ax3.tick_params(axis='both', which='both',left=False,labelleft = False)   
    ax3.set_xlabel('Difference in distances')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', prop={'size':6})
    plt.tight_layout()
    fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\PIN\Hosting\graphs\path_differences_od50.pdf"), format = 'pdf')
plotgraphs()



















#
"""
Commented out all below as Hadi now has the dataframes and is finalising these graphs?
"""
#-----------------------------------------------------------------------------

# #heatmap of lsoas ------------------------------------------------------------
# def heatmap(feature):
#     base = normal[feature][0]

    
#     fig, [[ax0, ax1, ax2], [ax01, ax11, ax21], [ax02, ax12, ax22]] = plt.subplots(3,3)
#     fig.set_size_inches(7,7)
#     sns.heatmap(minmax[feature][0] - base, cmap = 'viridis',ax = ax0)
#     sns.heatmap(slmin[feature][0] - base, cmap = 'viridis',ax = ax1)
#     sns.heatmap(od50[feature][0] - base, cmap = 'viridis',ax = ax2)
#     ax0.set_xlabel('Minmax')
#     ax1.set_xlabel('slmin')
#     ax2.set_xlabel('Od50')
#     ax1.set_title('M = '+str(normal['m_values'][0]))
    
#     sns.heatmap(minmax[feature][1] - base, cmap = 'viridis',ax = ax01)
#     sns.heatmap(slmin[feature][1] - base, cmap = 'viridis',ax = ax11)
#     sns.heatmap(od50[feature][1] - base, cmap = 'viridis',ax = ax21)
#     ax01.set_xlabel('Minmax')
#     ax11.set_xlabel('slmin')
#     ax21.set_xlabel('Od50')
#     ax11.set_title('M = '+str(normal['m_values'][1]))
    
#     sns.heatmap(minmax[feature][2] - base, cmap = 'viridis',ax = ax02)
#     sns.heatmap(slmin[feature][2] - base, cmap = 'viridis',ax = ax12)
#     sns.heatmap(od50[feature][2] - base, cmap = 'viridis',ax = ax22)
#     ax02.set_xlabel('Minmax')
#     ax12.set_xlabel('slmin')
#     ax22.set_xlabel('Od50')
#     ax12.set_title('M = '+str(normal['m_values'][2]))
#     plt.tight_layout()
# #heatmap('edge_freqs')

# def adjacency_heat(m_index):
#     fig, [[ax0,ax1], [ax2,ax3]] = plt.subplots(2,2)
#     fig.set_size_inches(7,7)
#     fig.suptitle('M = '+str(normal['m_values'][m_index]))
#     lsoa_shape = lsoa_data['sheff_lsoa_shape']
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax0)
#     lsoa_shape['edge_freqs'] = normal['edge_freqs'][m_index].sum(1)/345
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax0)
#     lsoa_shape.plot(column='edge_freqs', scheme='Quantiles',
#                legend=True, legend_kwds=({'title':'Edge frequencies', 'loc':'lower left'}), ax=ax0, cmap='viridis_r')
#     ax0.set_title('Normal')
#     ax0.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
#     lsoa_shape = lsoa_data['sheff_lsoa_shape']
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax1)
#     lsoa_shape['edge_freqs'] = minmax['edge_freqs'][m_index].sum(1)/345
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax1)
#     lsoa_shape.plot(column='edge_freqs', scheme='Quantiles',
#                legend=True, legend_kwds=({'title':'Edge frequencies', 'loc':'lower left'}), ax=ax1, cmap='viridis_r')
#     ax1.set_title('Minmax')
#     ax1.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
#     lsoa_shape = lsoa_data['sheff_lsoa_shape']
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax2)
#     lsoa_shape['edge_freqs'] = slmin['edge_freqs'][m_index].sum(1)/345
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax2)
#     lsoa_shape.plot(column='edge_freqs', scheme='Quantiles',
#                legend=True, legend_kwds=({'title':'Edge frequencies', 'loc':'lower left'}), ax=ax2, cmap='viridis_r')
#     ax2.set_title('slmin')
#     ax2.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
#     lsoa_shape = lsoa_data['sheff_lsoa_shape']
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax3)
#     lsoa_shape['edge_freqs'] = od50['edge_freqs'][m_index].sum(1)/345
#     lsoa_shape.plot(color='white', edgecolor='black', ax=ax3)
#     lsoa_shape.plot(column='edge_freqs', scheme='Quantiles',
#                legend=True, legend_kwds=({'title':'Edge frequencies', 'loc':'lower left'}), ax=ax3, cmap='viridis_r')
#     ax3.set_title('od50')
#     ax3.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
#     plt.tight_layout()
# adjacency_heat(0)
# adjacency_heat(1)
# adjacency_heat(2)










# #distance changes graphs -----------------------------------------------------
# def paths_mat(layout):
#     paths = layout
#     paths = np.reshape(paths, (len(paths)))
#     paths_matrix = np.zeros((345,345))
#     lowInds = np.tril_indices(345,k=-1)
#     highInds  = np.triu_indices(345, k=1)
#     paths_matrix[lowInds] = paths
#     paths_matrix = paths_matrix + paths_matrix.T - np.diag(paths_matrix) 
#     return paths_matrix

# import scipy.io as sio 
# mldata = sio.loadmat(r'C:\Users\cip18jjp\Dropbox\PIN\hadi_scripts\optimisedpaths.mat')#import new paths
# centroid_info = load_obj("centroids_beta_params_centroiddists")
# centroid_paths_matrix = np.vstack(centroid_info['euclidean_dists'])
# normal_paths_matrix = load_obj("ave_paths")
# minmax_paths_matrix = paths_mat(mldata['minmax'])
# slmin_paths_matrix = paths_mat(mldata['slmin'])
# od50_paths_matrix = paths_mat(mldata['od50'])

# paths_change_sum = pd.DataFrame(data = {'minmax':100* ((minmax_paths_matrix.sum(1) - normal_paths_matrix.sum(1)) / normal_paths_matrix.sum(1)),
#                                         'slmin':100*((slmin_paths_matrix.sum(1) - normal_paths_matrix.sum(1)) / normal_paths_matrix.sum(1)),
#                                         'od50':100*((od50_paths_matrix.sum(1) - normal_paths_matrix.sum(1)) / normal_paths_matrix.sum(1))})
 

# paths_change = pd.DataFrame(data = {'minmax':100* ((minmax_paths_matrix[minmax_paths_matrix!=0].flatten() - normal_paths_matrix[minmax_paths_matrix!=0].flatten()) / normal_paths_matrix[minmax_paths_matrix!=0].flatten()),
#                                     'slmin':100*((slmin_paths_matrix[minmax_paths_matrix!=0].flatten() - normal_paths_matrix[minmax_paths_matrix!=0].flatten()) / normal_paths_matrix[minmax_paths_matrix!=0].flatten()),
#                                     'od50':100*((od50_paths_matrix[minmax_paths_matrix!=0].flatten() - normal_paths_matrix[minmax_paths_matrix!=0].flatten()) / normal_paths_matrix[minmax_paths_matrix!=0].flatten())}) 


# #Graph 1----------------------------------------------------------------------
# fig, [ax0,ax1,ax2] = plt.subplots(1,3)
# fig.set_size_inches(7,2.5)                             
# sns.regplot(x = normal_paths_matrix.sum(1)/1e6, y ="minmax", data = paths_change_sum, scatter_kws={'s':2}, label = 'Original distances', ax=ax0)
# sns.regplot(x = centroid_paths_matrix.sum(1)/1e6, y ="minmax", data = c_paths_change_sum,scatter_kws={'s':2},label = 'Centroid distances',  ax=ax0)
# sns.regplot(x = normal_paths_matrix.sum(1)/1e6, y ="slmin", data = paths_change_sum, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = centroid_paths_matrix.sum(1)/1e6, y ="slmin", data = c_paths_change_sum, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = normal_paths_matrix.sum(1)/1e6, y ="od50", data = paths_change_sum, scatter_kws={'s':2},ax=ax2)
# sns.regplot(x = centroid_paths_matrix.sum(1)/1e6, y ="od50", data = c_paths_change_sum, scatter_kws={'s':2},ax=ax2)
# handles, labels = ax0.get_legend_handles_labels()
# ax0.legend(handles, labels, loc='upper right', prop={'size':6})
# ax0.set_xlabel("Sum of original edges at LSOA"+ r"$[10^6]$", size = 'x-small')
# ax1.set_xlabel("Sum of original edges at LSOA"+ r"$[10^6]$",size = 'x-small')
# ax2.set_xlabel("Sum of original edges at LSOA"+ r"$[10^6]$",size = 'x-small')
# ax0.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax1.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax2.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax0.set_title("minmax")
# ax1.set_title("slmin")
# ax2.set_title("od50")
# plt.tight_layout()

# fig, [ax0,ax1,ax2] = plt.subplots(1,3)
# fig.set_size_inches(7,2.5)                             
# sns.regplot(x = normal_paths_matrix[minmax_paths_matrix!=0].flatten(), y ="minmax", data = paths_change, scatter_kws={'s':2}, label = 'Original distances', ax=ax0)
# sns.regplot(x = centroid_paths_matrix[minmax_paths_matrix!=0].flatten(), y ="minmax", data = c_paths_change,scatter_kws={'s':2},label = 'Centroid distances',  ax=ax0)
# sns.regplot(x = normal_paths_matrix[minmax_paths_matrix!=0].flatten(), y ="slmin", data = paths_change, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = centroid_paths_matrix[minmax_paths_matrix!=0].flatten(), y ="slmin", data = c_paths_change, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = normal_paths_matrix[minmax_paths_matrix!=0].flatten(), y ="od50", data = paths_change, scatter_kws={'s':2},ax=ax2)
# sns.regplot(x = centroid_paths_matrix[minmax_paths_matrix!=0].flatten(), y ="od50", data = c_paths_change, scatter_kws={'s':2},ax=ax2)
# ax0.set_xlabel("Original edges at LSOA", size = 'x-small')
# ax1.set_xlabel("Original edges at LSOA",size = 'x-small')
# ax2.set_xlabel("Original edges at LSOA",size = 'x-small')
# ax0.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax1.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax2.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax0.set_title("minmax")
# ax1.set_title("slmin")
# ax2.set_title("od50")
# ax0.set_yscale('log')
# ax1.set_yscale('log')
# ax2.set_yscale('log')
# ax0.set_xscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# plt.tight_layout()


# #Graph 2 attractivity---------------------------------------------------------
# optdata = load_obj("opt_data_2000run_lsoa")
# a1a2 = paths_mat(optdata['a1a2'])
# fig, [ax0,ax1,ax2] = plt.subplots(1,3)
# fig.set_size_inches(7,2.5)                             
# sns.regplot(x = a1a2[a1a2!=0].flatten(), y ="minmax", data = paths_change, logx= True, scatter_kws={'s':2}, ax=ax0)
# sns.regplot(x = a1a2[a1a2!=0].flatten(), y ="slmin", data = paths_change, logx= True, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = a1a2[a1a2!=0].flatten(), y ="od50", data = paths_change,  logx= True,scatter_kws={'s':2},ax=ax2)
# ax0.set_xlabel("Attractivity product of LSOA pair", size = 'x-small')
# ax1.set_xlabel("Attractivity product of LSOA pair",size = 'x-small')
# ax2.set_xlabel("Attractivity product of LSOA pair",size = 'x-small')
# ax0.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax1.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax2.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax0.set_title("minmax")
# ax1.set_title("slmin")
# ax2.set_title("od50")
# plt.tight_layout()

# #Graph 3 sum edge changes vs income means
# income_means = []
# for i,j,k,l in lsoa_data['income_params']:
#     income_means.append(stats.beta.mean(i,j,k,l))
# income_means = np.asarray(income_means)

# fig, [ax0,ax1,ax2] = plt.subplots(1,3)
# fig.set_size_inches(7,2.5)                             

# sns.regplot(x = income_means, y ="minmax", data = paths_change_sum, scatter_kws={'s':2}, label = 'Original distances', ax=ax0)

# sns.regplot(x = income_means, y ="slmin", data = paths_change_sum, scatter_kws={'s':2},ax=ax1)

# sns.regplot(x = income_means, y ="od50", data = paths_change_sum, scatter_kws={'s':2},ax=ax2)

# ax0.set_xlabel("Mean income of LSOA", size = 'x-small')
# ax1.set_xlabel("Mean income of LSOA", size = 'x-small')
# ax2.set_xlabel("Mean income of LSOA",size = 'x-small')
# ax0.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax1.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax2.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax0.set_title("minmax")
# ax1.set_title("slmin")
# ax2.set_title("od50")
# plt.tight_layout()

# #Graph 4 - populations --------------------------------------------------------
# pop = paths_mat(optdata['pop'])
# fig, [ax0,ax1,ax2] = plt.subplots(1,3)
# fig.set_size_inches(7,2.5)                             
# sns.regplot(x = pop[pop!=0].flatten(), y ="minmax", data = paths_change, scatter_kws={'s':2}, label = 'Original distances', ax=ax0)
# sns.regplot(x = pop[pop!=0].flatten(), y ="minmax", data = c_paths_change,scatter_kws={'s':2},label = 'Centroid distances',  ax=ax0)
# sns.regplot(x = pop[pop!=0].flatten(), y ="slmin", data = paths_change, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = pop[pop!=0].flatten(), y ="slmin", data = c_paths_change, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = pop[pop!=0].flatten(), y ="od50", data = paths_change, scatter_kws={'s':2},ax=ax2)
# sns.regplot(x = pop[pop!=0].flatten(), y ="od50", data = c_paths_change, scatter_kws={'s':2},ax=ax2)
# ax0.set_xlabel("Combined population of LSOA pair", size = 'x-small')
# ax1.set_xlabel("Combined population of LSOA pair",size = 'x-small')
# ax2.set_xlabel("Combined population of LSOA pair",size = 'x-small')
# ax0.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax1.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax2.set_ylabel("% change of edge length at LSOA",size = 'x-small')
# ax0.set_title("minmax")
# ax1.set_title("slmin")
# ax2.set_title("od50")
# # ax0.set_xscale('log')
# # ax1.set_xscale('log')
# # ax2.set_xscale('log')
# # ax0.set_yscale('log')
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# plt.tight_layout()

# pop = np.asarray(lsoa_data['edu_counts']).reshape((len(lsoa_data['edu_counts']), 1))
# fig, [ax0,ax1,ax2] = plt.subplots(1,3)
# fig.set_size_inches(7,2.5)                             
# sns.regplot(x = pop, y ="minmax", data = paths_change_sum, scatter_kws={'s':2}, label = 'Original distances', ax=ax0)
# sns.regplot(x = pop, y ="minmax", data = c_paths_change_sum,scatter_kws={'s':2},label = 'Centroid distances',  ax=ax0)
# sns.regplot(x = pop, y ="slmin", data = paths_change_sum, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = pop, y ="slmin", data = c_paths_change_sum, scatter_kws={'s':2},ax=ax1)
# sns.regplot(x = pop, y ="od50", data = paths_change_sum, scatter_kws={'s':2},ax=ax2)
# sns.regplot(x = pop, y ="od50", data = c_paths_change_sum, scatter_kws={'s':2},ax=ax2)
# ax0.set_xlabel("Population of LSOA", size = 'x-small')
# ax1.set_xlabel("Population of LSOA", size = 'x-small')
# ax2.set_xlabel("Population of LSOA",size = 'x-small')
# ax0.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax1.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax2.set_ylabel("% change in sum of edges at LSOA",size = 'x-small')
# ax0.set_title("minmax")
# ax1.set_title("slmin")
# ax2.set_title("od50")
# plt.tight_layout()