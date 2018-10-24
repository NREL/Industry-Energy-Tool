# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:51:30 2018

@author: cmcmilla
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import DataprepClustering as dataprep

county_energy = dataprep.CountyEnergy_import()

county_energy.replace(
    {'subsector': {42: 21, 31: '31-33', 32: '31-33', 33: '31-33' }},
    inplace=True
    )

county_energy_total = county_energy.groupby('fips_matching')['Total'].sum()


def ClusterNorm(file, county_energy_total):
    """
    Method to normalize cluster assignment numbering.
    """

    clusters = pd.read_csv(file, index_col=[0], dtype=np.int32)

    # Adjust cluster numbering to n + 1
    clusters = clusters.apply(lambda x: x + 1)

    clusters.sort_values('cluster', inplace=True)

    cluster_breaks = []

    for g in clusters.groupby(clusters.iloc[:, 0]).groups:

        grouped = clusters.groupby(clusters.iloc[:, 0]).get_group(g)

        for i in range(0, len(grouped.index) + 1):
            print(grouped.index[i])
            same_c = (grouped.iloc[i].values == grouped.iloc[i + 1].values)

            if all(same_c) == True:
                cluster_breaks.append(grouped.index[i])
                break
            else:
                continue

    cluster_dict = {}

    clusters_adj = pd.DataFrame(clusters, copy=True)

    for col in clusters_adj.columns:

        number_dict = {}

        for b in range(0, len(cluster_breaks)):

            number_dict[clusters.loc[cluster_breaks[b], col]] = \
                clusters.loc[cluster_breaks[b], 'cluster']

        clusters_adj.loc[:, col] = clusters_adj[col].replace(number_dict)

    cluster_description = {'sum': pd.DataFrame(),
                           'count': pd.DataFrame()
                           }

    for d in ['sum', 'count']:

        for n in range(0, 100):

            cluster_description[d] = \
                cluster_description[d].append(
                    pd.concat(
                        [county_energy_total, clusters_adj.iloc[:, n]], axis=1
                        ).groupby(clusters_adj.columns[n]).agg([d]).T
                    )

        cluster_description[d].reset_index(inplace=True, drop=True)

    cluster_description['avg'] = cluster_description['sum'].divide(
        cluster_description['count'], fill_value=0
        )

    cluster_description['sum_sorted'] = \
        cluster_description['sum'].apply(lambda x: np.sort(x), axis=1)

    return cluster_description, clusters_adj


def StabilityHMap(clusters_adj, name):
    """
    Method to plot heatmap of k-means cluster assignment.
    """

    ncolors=['#e66101','#fdb863','#b2abd2','#5e3c99']

    eucolors = ['#ece2f0', '#a6bddb', '#1c9099']

    c_count = len(clusters_adj.index)

    n_cluster = clusters_adj.cluster.max() + 1

    if n_cluster - 1 == 4:

        s_colors = ncolors

    else:

        s_colors = eucolors

    print(s_colors, n_cluster)

    with plt.rc_context(dict(sns.axes_style("white"),
                             **sns.plotting_context('talk')
                            )):

        ax = sns.heatmap(
            clusters_adj, cbar=True,
            cmap=LinearSegmentedColormap.from_list("", s_colors),
            cbar_kws={'pad': 0.01, 'label': 'Cluster',
            'spacing': 'proportional', 'ticks': (range(0, n_cluster))},
            xticklabels=False, yticklabels=False, vmin=1, vmax=n_cluster-1
            )

        ax.set(xlabel='Iteration (n=100)', ylabel='County (n=' +
            str(c_count) +')')

        plt.savefig('Iteration_HeatMap_' + name + '.png', dpi=200)


def IterHist(energy_by_cluster):
    """
    Create and save histogram showing total cluster energy iteration results.
    """

    if energy_by_cluster.columns.shape[0] == 4:
        c_name = 'NAICS'
        s_colors = ['#e66101','#fdb863','#b2abd2','#5e3c99']
    else:
        c_name = 'EndUse'
        s_colors = ['#ece2f0', '#a6bddb', '#1c9099']

    sns.set_palette(s_colors)
    sns.set_style('whitegrid')
    sns.set_context('talk')
    fig, ax = plt.subplots()
    for c in energy_by_cluster.columns:
        sns.distplot(energy_by_cluster[c].divide(1.05505585),
                     kde=False,
                     hist_kws={'label': 'Cluster ' + str(c),
                               'alpha':0.7})
    ax.set_ylabel('Number of Iterations')
    ax.set_xlabel('Cluster Total Energy (PJ)')
    ax.set_title('Cluster Iterations: Energy by ' + c_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        'Iteration_hist_' + c_name + '_PJ.png', dpi=100
        )


def IterBox(avg_energy_by_cluster):
    """
    Create and save boxplot showing total cluster energy iteration results.
    """

    if avg_energy_by_cluster.columns.shape[0] == 4:
        c_name = 'NAICS'
        s_colors = ['#e66101','#fdb863','#b2abd2','#5e3c99']
    else:
        c_name = 'EndUse'
        s_colors = ['#ece2f0', '#a6bddb', '#1c9099']

    sns.set_palette(s_colors)
    sns.set_style('whitegrid')
    sns.set_context('talk')

    fig, ax = plt.subplots()
    sns.boxplot(avg_energy_by_cluster)
    ax.set_xlabel('Cluster')
    ax.set_ylim(0, )
    ax.set_ylabel('Cluster Mean County Energy Use (TBtu/county)')
    ax.set_title('Cluster Iterations: Energy by ' + c_name)
    fig.tight_layout()
    fig.savefig(
        'Iteration_boxplot_avg_' + c_name + '.png', dpi=100
        )
#%%
cluster_description, clusters_adj = \
    ClusterNorm('kmeans_iterations_20181001.csv', county_energy_total)

cluster_eu_description, clusters_eu_adj =\
    ClusterNorm('cluster_iter_eu_20180115.csv', county_energy_total)

# StabilityHMap(clusters_adj, 'NAICS')
#
# StabilityHMap(clusters_eu_adj, 'EndUse')

for df in [cluster_description, cluster_eu_description]:
    IterHist(df['sum'])
    IterBox(df['avg'])
