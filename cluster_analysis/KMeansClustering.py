# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:25:03 2017

@author: colin.mcmillan@nrel.gov
"""

# Cluster algorithms in SciPy clustering package.
# scipy.cluster.vq: vector quantization and k-means algorithms.
# scipy.cluster.hierarchy: hierarchical and aggolomerative clustering.

# Cluster algorithms also in scikit package (would need to cite)
# sklearn.cluster.
#%%
import pandas as pd
import numpy as np
import DataprepClustering as dataprep
import scipy.cluster as spc
import matplotlib.pyplot as plt
import seaborn as sns
import MakeClusterMap
import ClusterPlots
import county_emps
from scipy.spatial.distance import cdist, pdist
import itertools


county_energy = dataprep.CountyEnergy_import()

county_energy.replace(
    {'subsector': {42: 21, 31: '31-33', 32: '31-33', 33: '31-33' }},
    inplace=True
    )

county_subsector = {}

for sub in county_energy.subsector.drop_duplicates():
    county_subsector[sub] = \
        pd.DataFrame(county_energy[county_energy.subsector==sub])

county_socio = dataprep.CountySocio_import()

eu_energy = dataprep.CountyEU_import()

cty_indemp = county_emps.County_IndEmp()

ag_est_file = 'U:/ICEP/Cluster analysis paper/USDA_TotFarmCounts.csv'

# Iterated KMeans results for 6-digit NAICS. Use first column as final cluster
# results
final_kmeans_cluster_NAICS = pd.read_csv('kmeans_iterations_20181001.csv',
                                         usecols=['fips_matching', 'cluster'])

# Iterated KMeans results for energy by end use. Use first column as final
# cluster results.
final_kmeans_cluster_EU = pd.read_csv('cluster_iter_eu_20180115.csv',
                                      usecols=[0, 1])
                                      
final_kmeans_cluster_EU.rename(columns={'Unnamed: 0': 'fips_matching'},
                               inplace=True)

# Sum establishment counts by employee size and by county
est_counts = pd.DataFrame(
    cty_indemp.cbp.groupby('fips_matching')[
        ['n1_4', 'n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499',
         'n500_999', 'n1000']
         ].sum()
         )

# Ag establishment counts are total by county only (no disaggregation by size)
ag_est = cty_indemp.farm_est(ag_est_file)

est_counts = pd.concat([est_counts, ag_est], axis=1)

est_counts.fillna(0, inplace=True)

K = range(1, 30)

#%%
def do_kmeans(county_energy_array, K):
    """
    K-menas clustering of county-level energy data.
    K is the number of clusters to calculate, represented as a range
    """

    # Whiten observations (normalize by dividing each column by its standard
    # deviation across all obervations to give unit variance. 
    # See scipy.cluster.vq.whiten documentation).
    # Need to whitend based on large differences in mean and variance
    # across energy use by NAICS codes.
    energy_whitened = spc.vq.whiten(county_energy_array)

    # Run K-means clustering for the number of clusters specified in K
    KM_energy = [spc.vq.kmeans(energy_whitened, k, iter=25) for k in K]

    KM_results_dict = {}

    KM_results_dict['data_white'] = energy_whitened

    KM_results_dict['KM_results'] = KM_energy

    KM_results_dict['centroids'] = [cent for (cent, var) in KM_energy]

    # Calculate average within-cluster sum of squares
    KM_results_dict['avgWithinSS'] = [var for (cent, var) in KM_energy]

    return KM_results_dict
    
# %%
# Dictionaries for K-means results

KM_results_dict = {}

KM_results_nonlin_dict = {}

KM_results_abs_dict = {}

cla_input_dict = {}

cla_input_abs_dict = {}

indemp_count_dict = {}

ctyfips = {}

# %%
# Calculate K-means for 2, 3, 4, 5, and 6-digit NAICS groupings
linearize = True
ests = True
for n in [6]:

    indemp_count_dict[n] = dataprep.cty_indemp_counts(n, cty_indemp)

    cla_input_dict[n], ctyfips[n] = \
        dataprep.naics_group(county_energy, n, indemp_count_dict[n], norm=True)

    cla_input_abs_dict[n], ctyfips[n] = dataprep.naics_group(
        county_energy, n, indemp_count_dict[n], norm=False
        )

    if ests == True:

        if linearize == True:
            KM_results_abs_dict[n] = do_kmeans(
                np.hstack((
                    np.ma.log(cla_input_abs_dict[n]['cla_array']).data, 
                    est_counts.values
                    )), K
                )

            KM_results_dict[n] = do_kmeans(
                np.hstack((
                    np.ma.log(cla_input_dict[n]['cla_array']).data, 
                    est_counts.values
                    )), K
                )

        else:
            KM_results_abs_dict[n] = do_kmeans(
                np.hstack((
                    cla_input_abs_dict[n]['cla_array'], est_counts.values
                    )), K
                )            

            KM_results_dict[n] = do_kmeans(
                np.hstack((
                    cla_input_dict[n]['cla_array'], est_counts.values
                    )), K
                )
    else:

        if linearize == True:

            KM_results_dict[n] = do_kmeans(
                np.ma.log(cla_input_dict[n]['cla_array']).data, K
                )

            KM_results_abs_dict[n] = do_kmeans(
                np.ma.log(cla_input_abs_dict[n]['cla_array']).data, K
                )
  
        else:

            KM_results_dict[n] = do_kmeans(
                cla_input_dict[n]['cla_array'], K
                )

            KM_results_abs_dict[n] = do_kmeans(
                cla_input_abs_dict[n]['cla_array'], K
                )

    # Prep end-use data for cluster analysis.
    # Linearize, too.
    # Note matching of county FIPS with ctyfips_eu (an array),
    if n == 6:
        cla_inputeu, ctyfips_eu = dataprep.naics_group_eu(
            eu_energy, indemp_count_dict[6], norm=True
            )

        cla_inputeu_abs, ctyfips_euabs = dataprep.naics_group_eu(
           eu_energy, indemp_count_dict[6], norm=False
            )

        KM_results_dict['eu_abs_est'] = do_kmeans(
            np.hstack((cla_inputeu_abs['cla_array'],
                       est_counts[est_counts.index.to_series().apply(
                           lambda x: x in ctyfips_euabs.values)])), K
            )

        KM_results_dict['eu_est'] = do_kmeans(
            np.hstack((cla_inputeu['cla_array'],
                       est_counts[est_counts.index.to_series().apply(
                           lambda x: x in ctyfips_eu)])), K
            )

        KM_results_dict['eu_abs_est_lin'] = do_kmeans(
            np.ma.log(np.hstack((cla_inputeu_abs['cla_array'],
                                 est_counts[est_counts.index.to_series().apply(
                                     lambda x: x in ctyfips_euabs.values)]))
                ).data, K
            )

        KM_results_dict['eu_est_lin'] = do_kmeans(
            np.ma.log(np.hstack((cla_inputeu['cla_array'],
                                 est_counts[est_counts.index.to_series().apply(
                                     lambda x: x in ctyfips_eu)]))
                ).data, K
            )

        KM_results_dict['eu_abs'] = do_kmeans(cla_inputeu_abs['cla_array'], K)

        KM_results_dict['eu'] = do_kmeans(cla_inputeu['cla_array'], K)

    else:
        pass   
#%%
# Descriptive stats of county-level data, including q-q plot
import pylab
import scipy.stats as stats

#Estimate optimal number of bins (Freedman-Diaconis rule)

total_norm = county_energy.groupby('fips_matching').Total.sum().divide(
        county_socio.Ind_emp
        )

total_norm.replace({np.inf: 0}, inplace=True)
total_norm.fillna(0, inplace=True)
total_norm = total_norm

total_norm = total_norm[total_norm > 0]

fig, axs = plt.subplots(2, 1)  
plt.tight_layout(h_pad=3.5)
axs[0].hist(
    np.sum(cla_input_abs_dict[6]['cla_array'], axis=1), bins=30, log=True
    )
    
axs[0].set_xlabel('County Total Energy (TBtu)')
axs[0].set_ylabel('Number of Counties')
axs[0].set_title('County Energy-- Absolute', size='medium', weight='bold')

axs[1].hist(total_norm.values, bins=30, log=True)
axs[1].set_xlabel(
    'County Total Energy per Industrial Employee (TBtu/employee)'
    )
axs[1].set_ylabel('Number of Counties')
axs[1].set_title('County Energy-- Normalized', size='medium', weight='bold')
fig.savefig('DataOverview.png', dpi=200, bbox_inches='tight')

# Plot correlations of end uses
eu_corr = pd.DataFrame(
    cla_inputeu_abs['cla_array'], columns=cla_inputeu_abs['Enduse']
    ).corr()
    
eu_corr_norm = pd.DataFrame(
    cla_inputeu['cla_array'], columns=cla_inputeu['Enduse']
    ).corr()

n_corr = pd.DataFrame(
    cla_input_abs_dict[6]['cla_array'], columns=cla_input_abs_dict[6]['naics']
    )

n_corr = pd.DataFrame(n_corr, columns=n_corr.columns.sort_values()).corr()

n_corr_norm = pd.DataFrame(
    cla_input_dict[6]['cla_array'], columns=cla_input_dict[6]['naics']
    )

n_corr_norm = pd.DataFrame(
    n_corr_norm, columns=n_corr_norm.columns.sort_values()
    ).corr()

mask = np.zeros_like(eu_corr_norm, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(eu_corr_norm, mask=mask, cmap=cmap, vmax=1, center=0,
                #xticklabels=10, yticklabels=10, 
                square=True,
                cbar_kws={"shrink": .5}
                )
    ax.set_title('Energy by End-Use-- Normalized')
    fig.savefig('EU_norm_corr.png', dpi=200)
#%%
# Calculate percent of variance explained
k_euclid_dict = {}
dist_dict = {}
wscc_dict = {}
tss_dict = {}
bss_dict = {}

KM_results_dict['6_abs'] = KM_results_abs_dict[6]
for n in KM_results_dict.keys():
    k_euclid_dict[n] = [cdist(
        KM_results_dict[n]['data_white'], cent, 'euclidean'
        ) for cent in KM_results_dict[n]['centroids']]
        
    dist_dict[n] = [np.min(ke, axis=1) for ke in k_euclid_dict[n]]
    
    wscc_dict[n] = [sum(d**2) for d in dist_dict[n]]
    
    tss_dict[n] = sum(
        pdist(KM_results_dict[n]['data_white'])**2
        ) / KM_results_dict[n]['data_white'].shape[0]
        
    bss_dict[n] = tss_dict[n] - wscc_dict[n]
#%%
# Plot results
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.tight_layout()
ax[0, 0].plot(bss_dict['6_abs'] / tss_dict['6_abs'] * 100)
ax[0, 0].set_title('Energy by NAICS- Absolute')
ax[1, 0].plot(bss_dict[6] / tss_dict[6] * 100)
ax[1, 0].set_title('Energy by NAICS- Normalized')
ax[0, 1].plot(bss_dict['eu_abs_est_lin'] / tss_dict['eu_abs_est_lin'] * 100)
ax[0, 1].set_title('Energy by End-Use- Absolute')
ax[1, 1].plot(bss_dict['eu_est_lin'] / tss_dict['eu_est_lin'] * 100)
ax[1, 1].set_title('Energy by End Use- Normalized')
for n in [0, 1]:
    ax[1, n].set_xlabel('Number of Clusters')
    ax[n, 0].set_ylabel('Variance Explained (%)')
fig.savefig("Var_explained.png", size=(8, 6), dpi=200)


#%%
# Calculate silhouette score
# ?



# %%    
# K-means for sectors
for s in [11, 21, 23, '31-33']:
    cla_input_dict[s], ctyfips[s] = \
        dataprep.naics_group(
            county_subsector[s], 6, indemp_count_dict[6], norm=True
            )

    KM_results_dict[s] = do_kmeans(cla_input_dict[s]['cla_array'], K)

# %%
# Plot K-means results for all employment-normalized n-digit NAICS groupings
fig, axs = plt.subplots(3, 2, sharex=True)
fig.tight_layout()
axs[0, 0].plot(K, KM_results_noEst_dict[2]['avgWithinSS'], 'b*-')
axs[0, 0].grid(True)
axs[0, 0].set_title('2-Digit NAICS')
axs[0, 1].plot(K, KM_results_noEst_dict[3]['avgWithinSS'], 'b*-')
axs[0, 1].grid(True)
axs[0, 1].set_title('3-Digit NAICS')
axs[1, 0].plot(K, KM_results_noEst_dict[4]['avgWithinSS'], 'b*-')
axs[1, 0].grid(True)
axs[1, 0].set_title('4-Digit NAICS')
axs[1, 1].plot(K, KM_results_noEst_dict[5]['avgWithinSS'], 'b*-')
axs[1, 1].grid(True)
axs[1, 1].set_title('5-Digit NAICS')
axs[2, 0].plot(K, KM_results_nonlin_dict[6]['avgWithinSS'], 'b*-')
axs[2, 0].grid(True)
axs[2, 0].set_title('6-Digit NAICS')
# fig.suptitle("K-Means Within Cluster Sum of Squares")
fig.savefig("Results_20171120\\kmeans_nNAICS_noEst_Linear.png", dpi=150)

# %%
# Plot K-Means results by sector
fig, axs = plt.subplots(2, 2, sharex=True)
fig.tight_layout()
axs[0, 0].plot(K, KM_results_dict[11]['avgWithinSS'], 'b*-')
axs[0, 0].grid(True)
axs[0, 0].set_title('Agriculture 6-Digit NAICS')
axs[0, 1].plot(K, KM_results_dict[21]['avgWithinSS'], 'b*-')
axs[0, 1].grid(True)
axs[0, 1].set_title('Mining 6-Digit NAICS')
axs[1, 0].plot(K, KM_results_dict[23]['avgWithinSS'], 'b*-')
axs[1, 0].grid(True)
axs[1, 0].set_title('Construction 6-Digit NAICS')
axs[1, 1].plot(K, KM_results_dict['31-33']['avgWithinSS'], 'b*-')
axs[1, 1].grid(True)
axs[1, 1].set_title('Manufacturing 6-Digit NAICS')
fig.savefig("Results_20171120\\kmeans_norm_Subsector.png", dpi=150)

#%%
# Plot K-Means results for end use energy
fig, axs = plt.subplots(2, 1, sharex=True)
fig.tight_layout()
axs[0].plot(K, KM_results_dict['eu_abs_est_lin']['avgWithinSS'], 'b*-')
axs[0].grid(True)
axs[0].set_title('End Use Energy (Absolute-Linearized, EstCounts)')
axs[1].plot(K, KM_results_dict['eu_abs']['avgWithinSS'], 'b*-')
axs[1].grid(True)
axs[1].set_title('End Use Energy (Normalized-Linearized, EstCounts)')
fig.savefig("Results_20171204\\kmeans_Enduse.png", dpi=150)

# %%
ClusterPlots.elbow_plot(avgWithinSS, K, 'NAICS_4, Whitenend')

# %%

def format_ca_results(
            KM_results, county_socio, cla_input, ctyfips, naics_agg, n
            ):
    """
    Format cluster analysis results for n=k clusters, adding cluster ids
    and socio-economic data by county.
    """

    # Calcualte cluster ids and distance (distortion) between the observation 
    # and its nearest code for a chosen number of clusters.
    cluster_id, distance = spc.vq.vq(
        KM_results[naics_agg]['data_white'],
        KM_results[naics_agg]['centroids'][n - 1]
        )

    cols = ['cluster']
                
    if naics_agg in [11, 21, 23, '31-33']:
        
        # Combine cluster ids and energy data
        cluster_id.resize((cluster_id.shape[0], 1))

        # Name columns based on selected N-digit NAICS codes
        cla_input[naics_agg]['naics'].apply(lambda x: cols.append(str(x)))

        print("cluster_id shape: ", cluster_id.shape, "\n",
            'cla_array shape: ', cla_input[naics_agg]['cla_array'].shape
            )

        id_energy = \
            pd.DataFrame(
                np.hstack((cluster_id, cla_input[naics_agg]['cla_array'])),
                       columns=cols
                       )

        id_energy.set_index(ctyfips[naics_agg], inplace=True)

    if 'eu' in naics_agg:
        cluster_id.resize((cluster_id.shape[0], 1))

        for v in cla_input['Enduse']:
            cols.append(v)

        id_energy = \
            pd.DataFrame(
                np.hstack((cluster_id, cla_input['cla_array'])),
                           columns=cols
                )

        id_energy.set_index(ctyfips, inplace=True)

    else:
        # Combine cluster ids and energy data
        cluster_id.resize((cluster_id.shape[0], 1))

        # Name columns based on selected N-digit NAICS codes
        cla_input[naics_agg]['naics'].apply(lambda x: cols.append(str(x)))

        id_energy = \
            pd.DataFrame(
                np.hstack((cluster_id, cla_input[naics_agg]['cla_array'])),
                       columns=cols
                       )

        id_energy.set_index(ctyfips[naics_agg], inplace=True)

    id_energy = pd.concat([id_energy, county_socio], axis=1, join='inner')

    id_energy.loc[:, 'TotalEnergy'] = id_energy[cols[1:]].sum(axis=1)

    return id_energy
    

# %%
# Iterate k-means to check stability of clusters (K=4) for 6-digit NAICS
k_iter = range(1, 101)

k_range = range(4, 5) 

indemp_count_dict[6] = dataprep.cty_indemp_counts(6, cty_indemp)

cla_input_dict[6], ctyfips[6] = \
        dataprep.naics_group(county_energy, 6, indemp_count_dict[6], norm=True)

cluster_iters = pd.DataFrame(index=ctyfips[6])

for k in k_iter:
    
    KM_results_dict[6] = do_kmeans(
        np.hstack((
            np.ma.log(cla_input_dict[6]['cla_array']).data, 
            est_counts.values
            )), k_range
        )
    cluster_id, distance = spc.vq.vq(
        KM_results_dict[6]['data_white'],
        KM_results_dict[6]['centroids'][0]
        )

    cols = ['cluster']

    cluster_id.resize((cluster_id.shape[0], 1))

   # Name columns based on selected N-digit NAICS codes
    id_energy = pd.DataFrame(cluster_id, columns=cols)

    id_energy.set_index(ctyfips[6], inplace=True)

    cluster_iters = pd.concat([cluster_iters, id_energy['cluster']], axis=1)

#
# %% Iterate k-means to check stability of clusters (K=3) for end uses
cla_inputeu, ctyfips_eu = dataprep.naics_group_eu(
    eu_energy, indemp_count_dict[6], norm=True
    )

cluster_iters_eu = pd.DataFrame(index=ctyfips_eu)

k_range_eu  = range(3, 4)


for k in k_iter:

    cla_inputeu, ctyfips_eu = dataprep.naics_group_eu(
        eu_energy, indemp_count_dict[6], norm=True
        )

    KM_results_dict['eu_est_lin'] = do_kmeans(
        np.ma.log(np.hstack((cla_inputeu['cla_array'],
                             est_counts[est_counts.index.to_series().apply(
                                 lambda x: x in ctyfips_eu)]))
            ).data, k_range_eu
        )

    id_energy_eu = format_ca_results(
        KM_results_dict, county_socio, cla_inputeu, ctyfips_eu, 
        naics_agg='eu_est_lin', n=1
        )

    cluster_iters_eu = pd.concat(
        [cluster_iters_eu, id_energy_eu['cluster']], axis=1
        )


# %%
def describe_clusters(clustered_fips, county_socio, energy):
    """
    Attach energy and socio-economic data to pre-clustered counties. Input is a
    dataframe of fips and cluster ids.
    """
    
    if 'Enduse' in energy.columns:

        pvt_col = 'Enduse'
        
    else:

        pvt_col = 'naics'

    final_clusters_desc = pd.DataFrame(clustered_fips, copy=True)

    final_clusters_desc.set_index('fips_matching', inplace=True)

    energy_pvt = pd.pivot_table(energy, index='fips_matching',
                                columns=pvt_col, values='Total',
                                aggfunc=np.sum)
                                
    energy_pvt.loc[:, 'TotalEnergy'] = energy_pvt.sum(axis=1)

    for df in [energy_pvt, county_socio]:

        final_clusters_desc = pd.concat([final_clusters_desc, df], axis=1)

    final_clusters_desc.fillna(0, inplace=True)

    return final_clusters_desc

final_cluster_naics_desc = describe_clusters(final_kmeans_cluster_NAICS,
                                             county_socio, county_energy)
                
final_cluster_eu_desc = describe_clusters(final_kmeans_cluster_EU,
                                             county_socio, eu_energy)                                             

# County count by cluster
for df in [final_kmeans_cluster_NAICS, final_kmeans_cluster_EU]:
    print(df.groupby('cluster').count())
# %%
# Plot boxplots of final cluster assignments
ClusterPlots.socio_bxplt(final_cluster_naics_desc, 'NAICS')

ClusterPlots.socio_bxplt(final_cluster_eu_desc, 'EndUse')

ClusterPlots.enduse_bxplt(final_cluster_eu_desc, 'EndUse')

ClusterPlots.RUCC_hist(final_cluster_naics_desc, 'NAICS')

ClusterPlots.RUCC_hist(final_cluster_eu_desc, 'EndUse')

# %%
# Format K-means results by NAICS
id_energy_dict = {}

for k in KM_results_dict.keys():
    if k == '6':
        agg = int(k)
    else:
        pass
    id_energy_dict[k] = format_ca_results(KM_results_dict[k], county)

id_energy = format_ca_results(
    KM_results_dict, county_socio, cla_input_dict, ctyfips, naics_agg=6, n=4)
 
id_energy_noest = format_ca_results(
    KM_results_noEst_dict, county_socio, cla_input_dict, ctyfips,
    naics_agg=6, n=4
    )

id_energy_nonlin = format_ca_results(
    KM_results_nonlin_dict, county_socio, cla_input_dict, ctyfips, naics_agg=6,
    n=4
    )

id_energy_abs = format_ca_results(
    KM_results_abs_dict, county_socio, cla_input_abs_dict, ctyfips, naics_agg=6,
    n=4
    )    

id_energy_grouped = id_energy.dropna().groupby('cluster')

id_energy_noest_grouped = id_energy_noest.dropna().groupby('cluster')

id_energy_nonlin_grouped = id_energy_nonlin.dropna().groupby('cluster')

id_energy_abs_grouped = id_energy_abs.dropna().groupby('cluster')

# Format K-Means results by subsector
id_energy_sector = {}
id_energy_sector_grouped = {}

for s in [11, 21, 23, '31-33']:

    if s == 21:
        elbow = 6

    else:
        elbow = 3

    id_energy_sector[s] = \
        format_ca_results(
            KM_results_dict, county_socio, cla_input_dict,
            ctyfips, s, elbow
            )
    
    id_energy_sector_grouped[s] = id_energy_sector[s].groupby('cluster')

# Format K-means results by End use
id_energy_eu = format_ca_results(
    KM_results_dict, county_socio, cla_inputeu, ctyfips_eu, 
    naics_agg='eu_est_lin', n=3
    )

id_energy_eu_grouped = id_energy_eu.dropna().groupby('cluster')

# %%
# Create county maps of clusters. 
MakeClusterMap.County_Maps().make_cluster_map(id_energy, 'NAICS', 'PuOr')
fig = plt.figure(figsize=(3, 8))

MakeClusterMap.County_Maps().make_cluster_map(id_energy_eu, 'End Use', 'green')
fig = plt.figure(figsize=(3, 8))

MakeClusterMap.County_Maps().make_cluster_map(final_cluster_naics_desc, 'NAICS', 'PuOr')
fig = plt.figure(figsize=(3, 8))

MakeClusterMap.County_Maps().make_cluster_map(final_cluster_eu_desc, 'End Use', 'green')
fig = plt.figure(figsize=(3, 8))

# %%
# Create boxplots
for s in [11, 21, 23, '31-33']:
    ClusterPlots.socio_bxplt(id_energy_sector_grouped[s], str(s))

# Create RUCC histograms for each cluster
ClusterPlots.RUCC_hist(id_energy_abs_grouped, 'Linear_Abs')

fig = plt.boxplot(
    [id_energy_noest_grouped.RUCC_2013.get_group(g) for g in id_energy_noest_grouped.groups]
    )

plt.savefig('Results_20171120\\Clusters_RUCC_BxPlt_NoEst_Lin.png', 
            dpi=100)



# %%

# Post-processing for energy use by NAICS:
# Calculate fuel type fraction by cluster
energy_FT = county_energy.groupby('fips_matching')[
        ['fips_matching', 'Coal', 'Coke_and_breeze', 'Diesel', 'LPG_NGL',
         'Natural_gas','Net_electricity', 'Other', 'Total']
         ].sum()
    
ClusterPlots.FTmix_cluster_bxplt(final_cluster_naics_desc, energy_FT, 'NAICS')

# %%
# Calculate total energy distributions

def Indmix_cluster(id_energy_df, county_energy, measure='sum'):

    abs_energy = pd.pivot_table(
        county_energy[['fips_matching', 'naics', 'Total']],
        index='fips_matching', values='Total', columns='naics', aggfunc='sum',
        fill_value=0
        )

    indmix = pd.DataFrame()

    if measure == 'sum':

        abs_energy = pd.concat([id_energy_df['cluster'], abs_energy], axis=1)

        abs_energy_grp = abs_energy.groupby('cluster')

        for c in abs_energy_grp.groups:

            indmix = pd.concat(
                [indmix, abs_energy_grp.get_group(c).iloc[:, 1:].sum()],
                axis=1
                )

            indmix.rename(columns={0: c + 1}, inplace=True)

    else:

        norm_energy = abs_energy.divide(abs_energy.sum(axis=0), axis=1)

        norm_energy = \
            pd.concat([id_energy_df['cluster'], norm_energy], axis=1)

        if measure == 'median':

            for c in norm_energy.groupby('cluster').groups:
    
                indmix = pd.concat([
                    indmix,
                    norm_energy.groupby('cluster').get_group(c).median()],
                    axis=1
                    )

                indmix.rename(columns={0: c + 1}, inplace=True)

        if measure == 'mean':

            for c in norm_energy.groupby('cluster').groups:

                indmix = pd.concat([
                    indmix,
                    norm_energy.groupby('cluster').get_group(c).mean()],
                    axis=1
                    )

                indmix.rename(columns={0: c + 1}, inplace=True)

    return indmix

final_naics_indmix_sum = Indmix_cluster(final_cluster_naics_desc, county_energy)

final_naics_indmix_med = \
    Indmix_cluster(final_cluster_naics_desc, county_energy, 'median')

# %%
# Plot all energy distribution permutations
for tf in itertools.permutations([True, False], r=2):
    
    ClusterPlots.Cluster_IndDist(final_naics_indmix_sum, mfg_only=tf[0],
                                 absv=tf[0])

    ClusterPlots.Cluster_IndDist(final_naics_indmix_sum, mfg_only=tf[0],
                                 absv=tf[1])                                
                                 


# %%
def Ind_TopNAICS(data_naics, top_n, convert=False, agg=False):
    """
    Sorts the top n-number of industry mix results and their relative 
    contribution to total cluster energy. Also returns fraction of
    total energy represented by n-number.
    Option to convert from TBtu to PJ
    """

    if convert == True:
        
        indmix = data_naics.copy(deep=True)
        
        indmix = indmix * 1.05505585
        
    else:
        
        indmix = data_naics

    top_naics = {}
    
    top_naics_abs = pd.DataFrame()
    
    top_naics_rel = pd.DataFrame()

    top_sum = pd.DataFrame(index=[indmix.columns], columns=['Overall_%'])

    for c in indmix.columns:

        top_naics_abs = pd.concat(
            [top_naics_abs,
             indmix[c].sort_values(ascending=False)[0: top_n]],
             axis=1, join='outer'
             )
        
        top_sum.loc[c, 'Overall_%'] = top_naics_abs[c].sum() / indmix[c].sum()
        
    for c in indmix.columns:
        
        top_naics_abs[c].update(indmix[c])
        
        top_naics_rel = pd.concat(
            [top_naics_rel, top_naics_abs[c].divide(indmix[c].sum())],
            axis=1
            )

#    for df in [top_naics_abs, top_naics_rel]:
#        
#       df.fillna(0, inplace=True)

    top_naics['abs'] = top_naics_abs

    top_naics['rel'] = top_naics_rel

    return top_naics, top_sum


final_naics_top_ind, final_top_ind_sum = \
    Ind_TopNAICS(final_naics_indmix_sum, 20, convert=True)

# Write results to single excel file
res_writer = pd.ExcelWriter('Ind_top20_results_2.xlsx')

for k in final_naics_top_ind.keys():
    final_naics_top_ind[k].to_excel(res_writer, k)
    
final_top_ind_sum.to_excel(res_writer, 'Overall_%')

res_writer.save()
#%%



# %%
# Create boxplots for End Uses
ClusterPlots.enduse_bxplt(id_energy_eu_grouped, 'Enduse_Enduse_boxplt')

ClusterPlots.socio_bxplt(id_energy_eu_grouped, 'Enduse_Socio_boxplt')


# Get RUCC info of clusters
for g in id_energy_grouped.groups:
    print(g, id_energy_grouped.get_group(g).RUCC_2013.median(),
          len(id_energy_grouped.get_group(g).index)
          )

