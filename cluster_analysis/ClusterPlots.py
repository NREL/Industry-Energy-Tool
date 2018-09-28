# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:16:51 2018

@author: cmcmilla
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np

def elbow_plot(avgWithinSS, K, title):
    """
    Method to plot K-means clustering results for elbow identification.
    K = number of calculated clusters
    """
    # Plot elbow curve to examine within-cluster sum of squares
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(K, avgWithinSS, 'b*-')

    #ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
    #    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)

    plt.xlabel('Number of clusters')

    plt.ylabel('Average within-cluster sum of squares')

    plt.title('Elbow for KMeans clustering--' + title)

    fig.tight_layout()

    fig.savefig('Elbow_KMeans_'+ title + '.png', dpi=100)
    

def socio_bxplt(clustered_counties, save_file_name):
    """
    Create and save a quadrant of boxplots showing total energy, poverty %,
    median household income and industry % of employment for clustered counties.
    """

    ncolors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']
    
    eucolors = ['#ece2f0', '#a6bddb', '#1c9099']

    if type(clustered_counties) == pd.core.frame.DataFrame:

        grouped_data = clustered_counties.groupby('cluster')

        if 'Facility HVAC' in clustered_counties.columns:
        
            ccolors = eucolors

        else:
            
            ccolors = ncolors

    else:

        grouped_data = clustered_counties
        
        ccolors = ncolors

    with sns.axes_style('whitegrid'):

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 6))

        bplt1 = axs[0, 0].boxplot(
            [grouped_data.TotalEnergy.get_group(g).dropna() 
                for g in grouped_data.groups], patch_artist=True
            )
        axs[0, 0].set_title('Total Energy (TBtu)')

        axs[0, 0].set_yscale('log')

        bplt2 = axs[0, 1].boxplot(
            [grouped_data['HUD_median_lowmed'].get_group(g).dropna()*100
                for g in grouped_data.groups], patch_artist=True
            )
        axs[0, 1].set_title('Poverty %, Total Population')

        axs[0, 1].set_ylim(0, 100)

        bplt3 = axs[1, 0].boxplot(
            [grouped_data.Med_income.get_group(g).dropna()
                for g in grouped_data.groups], patch_artist=True
            )
        axs[1, 0].set_title('Median Household Income ($)')
        
        axs[1, 0].set_xlabel('Cluster')
    
        bplt4 = axs[1, 1].boxplot(
            [grouped_data['Ind_emp_%'].get_group(g).dropna()*100
                for g in grouped_data.groups], patch_artist=True
            )
        axs[1, 1].set_title('Industry % of Total Employment')

        axs[1, 1].set_ylim(0, 100)
        
        axs[1, 1].set_xlabel('Cluster')
        
        for bplot in (bplt1, bplt2, bplt3, bplt4):
            for patch, color in zip(bplot['boxes'], ccolors):
                patch.set_facecolor(color)

        fig.set_size_inches(8, 8)

        fig.tight_layout()

        fig.savefig(save_file_name + '_SocioEcon_BxPlt.png', dpi=100)
        
        plt.close()


def enduse_bxplt(id_eu, save_file_name):
    """
    Create and save a quadrant of boxplots showing energy use by CHP/cogen,
    conventional boilers, machine drive, and process heating for counties
    clustered by end use.
    """

    eucolors = ['#ece2f0', '#a6bddb', '#1c9099']

    if type(id_eu) == pd.core.frame.DataFrame:

        id_eu_grouped = id_eu.groupby('cluster')

    else:
        pass

    with sns.axes_style('whitegrid'):    

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 6), sharey=True)

        eubplt1 = axs[0, 0].boxplot(
            [id_eu_grouped[
                'CHP and/or Cogeneration Process'
                ].get_group(g).dropna() for g in id_eu_grouped.groups],
            patch_artist=True)

        axs[0, 0].set_title('CHP/Cogen')
        
        axs[0, 0].set_ylim(0, 8)
        
        axs[0, 0].set_ylabel('Energy (TBtu)')

        eubplt2 = axs[0, 1].boxplot(
            [id_eu_grouped[
                'Conventional Boiler Use'
                ].get_group(g).dropna() for g in id_eu_grouped.groups],
            patch_artist=True
            )

        axs[0, 1].set_title('Conventional Boiler')

        eubplt3 = axs[1, 0].boxplot(
            [id_eu_grouped['Machine Drive'].get_group(g).dropna() for g in
            id_eu_grouped.groups], patch_artist=True
            )

        axs[1, 0].set_title('Machine Drive')

        axs[1, 0].set_ylabel('Energy (TBtu)')

        axs[1, 0].set_xlabel('Cluster')

        axs[1, 1].boxplot(
            [id_eu_grouped['Process Heating'].get_group(g).dropna() for g in
            id_eu_grouped.groups], patch_artist=True
            )

        axs[1, 1].set_title('Process Heating')
        
        axs[1, 1].set_xlabel('Cluster')

        for bplot in (eubplt1, eubplt2, eubplt3):
            for patch, color in zip(bplot['boxes'], eucolors):
                patch.set_facecolor(color)

        fig.set_size_inches(8, 8)

        fig.tight_layout()

        fig.savefig(save_file_name + 'EndUse_BxPlt.png', dpi=100)
        
        plt.close()


def RUCC_hist(clustered_counties, file_name):
    """
    Plot histograms of USDA Rural-Urban Continuum Codes (RUCC) for clustered
    counties.
    """
    

    if 'Facility HVAC' in clustered_counties.columns:
        
        grouped_data = clustered_counties.groupby('cluster')
        
        colors = ['#ece2f0', '#a6bddb', '#1c9099']

    else:

        grouped_data = clustered_counties.groupby('cluster')
        
        colors = ['#e66101','#fdb863','#b2abd2','#5e3c99']

    with sns.axes_style('whitegrid'):
        
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))

        axs[0, 0].hist(grouped_data.RUCC_2013.get_group(0), color=colors[0])

        axs[0, 0].set_title('Cluster 1')

        axs[0, 0].set_ylabel('Number of Counties')

        axs[0, 1].hist(grouped_data.RUCC_2013.get_group(1), color=colors[1])

        axs[0, 1].set_title('Cluster 2')

        axs[1, 0].hist(grouped_data.RUCC_2013.get_group(2), color=colors[2])

        axs[1, 0].set_title('Cluster 3')
        
        axs[1, 0].set_ylabel('Number of Counties')
        
        axs[1, 0].set_xlabel('USDA RUCC')

        if grouped_data.ngroups == 4:

            axs[1, 1].hist(grouped_data.RUCC_2013.get_group(3),
                            color=colors[3])
    
            axs[1, 1].set_title('Cluster 4')    
    
            axs[1, 1].set_xlabel('USDA RUCC')

        else:
            pass

        fig.set_size_inches(8, 8)

        fig.tight_layout()

        fig.savefig(file_name + '_RUCC_hist.png', dpi=100)
        
        plt.close()


def FTmix_cluster_bxplt(id_energy_df, energy_FT, title):
    """
    Draw fuel mix boxplots by cluster from indicated KMeans cluster IDs
    """

    ncolors=['#e66101','#fdb863','#b2abd2','#5e3c99']

    FTmix = pd.concat(
        [id_energy_df['cluster'], energy_FT.divide(energy_FT.Total, axis=0)],
         axis=1
        )

    FTmix.loc[:, 'cluster'] = FTmix.cluster.apply(lambda x: int(x + 1))
    
    FTmix.drop(['fips_matching', 'Total'], axis=1, inplace=True)

    FTmix = pd.melt(
        FTmix, id_vars=['cluster'],
        var_name=['FuelType'], value_name='TBtu'
        )

    FTmix.loc[:, 'TBtu'] = FTmix.TBtu*100

    sns.set_style('white')

    ax = sns.boxplot(data=FTmix, x='FuelType', y='TBtu', hue='cluster',
                     palette=ncolors)

    ax.set_ylabel('Fuel Fraction (%)')

    fig = ax.get_figure()

    plt.show()

    fig.tight_layout()

    fig.savefig(
        'FuelFrac_bxplt_' + title + '.png', bbox_inches='tight', dpi=200
        )
    
def Ind_cluster_bxplt(id_energy_df, energy_FT, title):
    """
    Draw sector boxplots and scatter plots by cluster 
    from indicated KMeans cluster IDs
    """

    ncolors=['#e66101','#fdb863','#b2abd2','#5e3c99']
             
    # melt data

    melted = pd.melt(id_energy_df.reset_index().iloc[:, 0:439],
                     id_vars=['cluster', 'index'], var_name='NAICS')
    
    melted.rename(columns={'index': 'fips_matching'}, inplace=True)
    
    naics_match = pd.DataFrame(melted.NAICS.drop_duplicates())
    
    naics_match.loc[:, 'N2'] = \
        naics_match.NAICS.apply(lambda x: int(str(x)[0:2]))
        
    naics_match.replace({'N2': {31: '31-33', 32: '31-33', 33: '31-33'}},
                        inplace=True)
        
    naics_match.loc[:, 'N3'] = \
        naics_match.NAICS.apply(lambda x: int(str(x)[0:3]))
        
    melted = pd.merge(melted, naics_match, how='inner', on=['NAICS'])
    
    fraction = melted.groupby(['fips_matching', 'N2']).value.sum().divide(
            melted.groupby('fips_matching').value.sum(), level=0
            ).reset_index()
    
    fraction = pd.merge(fraction, melted[['fips_matching', 'cluster']],
                        on=['fips_matching'], how='inner').drop_duplicates()
    
    fraction.loc[:, 'cluster'] = fraction.cluster.apply(lambda x: int(x + 1))

    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)

    sns.boxplot(data=fraction, x='N2', y='value', hue='cluster',
                palette=ncolors, fliersize=1)
    
#    sns.swarmplot(data=fraction, x='N2', y='value', hue='cluster'
#                  size=0.8, color=".3", linewidth=0, dodge=True)
    
    sns.stripplot(data=fraction, x='N2', y='value', hue='cluster',
                  color=".3", size=1, jitter=True, dodge=True)

    ax.set_xlabel('Industry Sector')
    
    ax.set_xticklabels(['Agriculture', 'Mining', 'Construction',
                   'Manufacturing'])
            
    vals = ax.get_yticks()
    
    ax.set_yticklabels(['{0:,.0%}'.format(x) for x in vals])
    
    ax.set_ylabel('Fraction County Energy (%)')

    plt.show()

    fig.tight_layout()

    fig.savefig(
        'FuelFrac_bxplt_' + title + '.png', bbox_inches='tight', dpi=200
        )

def Cluster_IndDist(indmix_sum, mfg_only=False, absv=True):
    """
    Draw cumulative energy for each cluster by NAICS, either as absolute value
    (absv=True) or as % (absv=False)
    """

    # Change NAICS values to be all six-digit (applies just to ag)

    nc = int(indmix_sum.columns.max())

    pltdata = pd.DataFrame(indmix_sum.reset_index(), copy=True)

    pltdata.rename(columns={'index': 'NAICS'}, inplace=True)

    pltdata.loc[:, 'NAICS'] = pltdata.NAICS.astype('int')

    pltdata.loc[:, 'len_N'] = pltdata['NAICS'].apply(lambda x: len(str(x)))

    pltdata.loc[pltdata.groupby('len_N').get_group(4).index, 'NAICS'] =\
         pltdata.loc[pltdata.groupby('len_N').get_group(4).index, 'NAICS']*100

    pltdata.loc[pltdata.groupby('len_N').get_group(5).index, 'NAICS'] =\
         pltdata.loc[pltdata.groupby('len_N').get_group(5).index, 'NAICS']*10     

    if mfg_only == True:
        
        pltdata.loc[:, 'ticks'] = \
            pltdata['NAICS'].apply(lambda x: float(str(x)[0:3]))
            
    else:
        
        pltdata.loc[:, 'ticks'] = \
            pltdata['NAICS'].apply(lambda x: float(str(x)[0:2]))

    tick_index = pltdata.drop_duplicates('ticks').index

    for c in ['len_N', 'ticks']:

        pltdata.drop(c, axis=1, inplace=True)

    pltdata.loc[tick_index, 'ticks'] = pltdata.loc[tick_index, 'NAICS']

    #pltdata.ticks.fillna(0, inplace=True)
    
    pltdata.set_index('NAICS', inplace=True)

    #pltdata.loc[:, 'ticks'] = pltdata.ticks.astype('int')

    ncolors=['#e66101','#fdb863','#b2abd2','#5e3c99']

    with plt.rc_context(dict(sns.axes_style("whitegrid"),
                     **sns.plotting_context('talk'))
                    ):

        fig, ax = plt.subplots()

        if mfg_only is True:

            pltdata = pd.DataFrame(pltdata[pltdata.index > 311000], copy=True)
            
            ylabmfg = ' _Mfgonly'

        else:

            ylabmfg = ''

        for y in range(1, nc + 1):

            curve_x= range(1, len(pltdata.index) + 1)

            xticks = pd.DataFrame([(x for x in curve_x), pltdata.ticks]).T

            xticks = xticks[xticks[1].notnull()][0]

            if absv == True:

                curve_y = pltdata[y].cumsum().values

                ylab = 'Cumulative Energy (TBtu)' + ylabmfg

            if absv == False:
 
                curve_y =\
                    pltdata[y].divide(
                        pltdata[y].sum()
                        ).cumsum().values*100
                    
                ylab = 'Cumulative Energy (%)' + ylabmfg

            ax.plot(
                curve_x, curve_y, ncolors[int(y)-1], linewidth=2.7,
                label='Cluster ' + str(int(y))
                )

        ax.legend()

        ax.set(xlabel='NAICS Code', ylabel=ylab, #xlim=[pltdata.index.min(), pltdata.index.max()],
               xticks=xticks# pltdata[pltdata.ticks>0].ticks.values
               )

        if mfg_only == False:

            ax.set_xticklabels(
                [str(x)[0:2] for x in pltdata[pltdata.ticks>0].ticks],
                rotation='vertical', size=12
                   )

        if mfg_only == True:
            
            ax.set_xticklabels(
                [str(x)[0:3] for x in pltdata[pltdata.ticks>0].ticks],
                rotation='vertical', size=10
                   )

        fig.savefig(
            ylab+'_Curve.png', bbox_inches='tight',
            dpi=200
            )

        plt.close()
    
def cluster_pieplt(final_cluster_naics_desc):
    """
    Create pie charts for each of the clusters that depict the contribution of 
    energy by NAICS to county energy total.
    """
    
    data = final_cluster_naics_desc.iloc[:, 0:439].copy()
    
    data = pd.pivot_table(data, index='cluster', aggfunc=np.sum)
    
    data = pd.melt(data.reset_index(), id_vars='cluster', var_name='naics')
    
    data.loc[:, 'naics_2'] = data.naics.apply(lambda x: int(str(x)[0:2]))
    
    data.replace({'naics_2': {31: '31-33', 32: '31-33', 33: '31-33'}}, 
                 inplace=True)
    
    data.loc[:, 'cluster'] = data.cluster.add(1)
    
    # Sequential colormaps for each 2-digit naics category
    cmaps = {'31-33': 'Greys_r', 23: 'Purples_r', 11: 'Blues_r',
             21: 'Oranges_r'}
    
    colors = []
    
    labels = pd.concat([data[data.cluster == 1].naics_2,
                        data.naics_2.drop_duplicates()], axis=1)
    
    labels = labels.iloc[:, 1]
    
    labels.replace({11: 'Agriculture', 21: 'Mining', 23: 'Construction',
                    '31-33': 'Manufacturing', np.nan: ""}, inplace=True)
    
    # Adjust the location of 'Manufacturing' label
    labels.iloc[74] = ""
    
    labels.iloc[256] = 'Manufacturing'
    
    for n in data.naics_2.drop_duplicates():

        # Divide by 4 because there are 4 clusters
        len_n = int(len(data[data.naics_2 == n])/4)
    
        [colors.append(plt.get_cmap(cmaps[n])(i)) for i in np.linspace(0, 1, len_n)]

    fig = plt.figure(figsize=(8, 8))
    
    pie_grid = GridSpec(2, 2)
    
    plt.subplot(pie_grid[0, 0], aspect=1, title='Cluster 1')
    
    plt.pie(data[data.cluster == 1].value, colors=colors, labels=labels)
    
    plt.subplot(pie_grid[0, 1], aspect=1, title='Cluster 2')
    
    plt.pie(data[data.cluster == 2].value, colors=colors, labels=labels)
    
    plt.subplot(pie_grid[1, 0], aspect=1, title='Cluster 3')
    
    plt.pie(data[data.cluster == 3].value, colors=colors, labels=labels)
    
    plt.subplot(pie_grid[1, 1], aspect=1, title='Cluster 4')
    
    plt.pie(data[data.cluster == 4].value, colors=colors, labels=labels)
    
    plt.tight_layout()
    
    plt.savefig('cluster_pie.png', dpi=300, bbox_inches='tight')


    
def IntensityBxplt(final_cluster_naics, county_energy, county_ind_emp):
    """
    Normalize county total energy use by county total industry employment
    counts and create and save boxplot of energy intensity of counties for
    each cluster.
    """

    df = pd.DataFrame(county_energy, copy=True,
                      columns=['fips_matching', 'Total'])

    df.rename(columns={'Total': 'Intensity'}, inplace=True)

    df = df.groupby(['fips_matching']).sum()

    emp_counts = county_ind_emp['ag'].sum(level=0).add(
        county_ind_emp['non_ag'].sum(level=0)
        )

    # Normalize by number of employees by NAICS    
    df = df.divide(emp_counts, axis='index') * 1000000

    df = pd.concat([df, final_cluster_naics.set_index('fips_matching')],
                   axis=1)

    df_grouped = df.groupby('cluster')

    ncolors=['#e66101','#fdb863','#b2abd2','#5e3c99']

    with sns.axes_style('whitegrid'):    

        fig = plt.figure(1, figsize=(8,8), tight_layout=True)

        ax = fig.add_subplot(111)

        intbplt = ax.boxplot(
            [df_grouped[
                'Intensity'
                ].get_group(g).dropna() for g in df_grouped.groups],
                patch_artist=True
            )

        ax.set_title('County Energy Intensity')

        ax.set_ylabel('MMBtu per Industrial Employee')

        ax.set_yscale('log')

        ax.set_ylim(0.01, 100)

        ax.set_xlabel('Cluster')

        for median in intbplt['medians']:
            median.set(color='#000000', linewidth=2)

        for patch, color in zip(intbplt['boxes'], ncolors):
            patch.set_facecolor(color)

        fig.savefig('ClusterEnergyIntensity_bxplt', dpi=100)

        plt.close()

#fig = plt.plot(curve_x, curve_y)
#ax.set(xticks=pltdata[pltdata.ticks>0].ticks.values)