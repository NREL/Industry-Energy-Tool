# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:00:05 2018

@author: cmcmilla
"""
import pandas as pd
import numpy as np


def switch_fuels(initial_stock, new_stock, county_enduse, fuel_types, change_2050,
                 ind=None, enduses=None, temp_band=None):
    """
    Method for estimating impacts of switching from fossil fuels to
    electricity. Fuel switching affects new stock (i.e., post-2014 stock) only.
    Fuel switching is based on user-defined industries (ind: specified either
    as AEO industry or 4-digit NAICS code), end uses (enduses:
    'Conventional Boiler', 'CHP and/or Cogeneration', 'Process Heating',
    'Process Cooling', 'Machine Drive', 'Electro-chemical', 'Other'), and/or
    temperature range (temp_band: <100, 100-249, 250-499, 500-999, >1000).
    Method modifies new stock by fuel type and returns a dictionary of
    modified dataframes.
    """
    change_2050 = change_2050/100

    ffuels = list(fuel_types)

    if 'Total_energy_use' in ffuels:
        ffuels.remove('Total_energy_use')

    else:
        pass

    years = new_stock[list(new_stock.keys())[0]].columns.values

    switch = pd.DataFrame({'switch': False},
                          index=new_stock['Total_energy_use'].index).reset_index()

    switch.loc[:, 'NAICS_4'] = county_enduse.reset_index()['naics'].apply(
        lambda x: int(str(x)[0:4])
        )

    def temp_select(temp_band, new_stock):

        temp_switch = np.where(new_stock[temp_band[0]][2014] > 0, True, False)

        if len(temp_band) > 0: 

            for t in range(0, len(temp_band)):

                temp_switch = np.add(
                    temp_switch,
                    np.where(new_stock[temp_band[t]][2014] > 0, True, False)
                    )

        return temp_switch

    if ind is None:

        if temp_band is None:

            if enduses is None:

                pass

            else:

                for eu in enduses:

                    switch.loc[
                        switch.groupby('ST_enduse').get_group(eu).index,
                        'switch'
                        ] = True

        else:

            temp_switch = temp_select(temp_band, new_stock)

    else:

        if temp_band is None:

            if enduses is None:

                pass

            else:

                for eu in enduses:

                    switch.loc[
                        switch.groupby('ST_enduse').get_group(eu).index,
                        'switch'
                        ] = True

        else:

            temp_switch = temp_select(temp_band, new_stock)

        ind = pd.Series(ind)    

        for i in ind:

            if type(i) == np.int:
    
                switch.loc[
                    switch.groupby('NAICS_4').get_group(i).index,
                    'switch'
                    ] = True

            # This should be built out for resetting index with 4-digit NAICS
            if type(i) == np.str:

                if i == 'All':

                    switch.loc[:, 'switch'] = True

                else:
                
                    switch.loc[
                        switch.groupby('AEO_industry').get_group(i).index,
                        'switch'
                        ] = True

    if temp_band is not None:

        switch.loc[:, 'switch'] = switch.switch.multiply(temp_switch)

    else:
        pass

    switch.set_index(['region', 'AEO_industry', 'ST_enduse'], inplace=True)

    ff_change = pd.DataFrame(
        np.linspace(0, change_2050,
                    len(range(years[0], years[-1]+1))),
        index=range(years[0], years[-1]+1)).T

    # The fraction of new stock that is new additions (i.e., replacement stock)
    addition_fraction = \
        initial_stock['new_additions'].divide(initial_stock['new'])

    ff_addition_fraction = addition_fraction.iloc[:, 2:].copy()

    ff_addition_fraction.loc[:, :] = \
        np.multiply(addition_fraction.iloc[:, 2:].values,
                    ff_change.values)

    # The fraction of new stock that is fuel switched from fossil to electric.
    ff_switch_fraction = initial_stock['new'].multiply(
        ff_addition_fraction
        ).cumsum(axis=1).divide(initial_stock['new'])

    def merge_switch(df, switch_df, years):
        """
        Use merge operation to eep only region, AEO_industry, and 
        ST_enduse values that have been selected for electrification.
        """
        merged = \
            pd.merge(df[years],switch[switch.switch == True],
                     left_index=True, right_index=True, how='inner')

        merged.drop(['switch', 'NAICS_4'], axis=1, inplace=True)

        return merged

    ff_switch_fraction = merge_switch(ff_switch_fraction, switch, years)

    elect_stock_adj_new = merge_switch(new_stock['Net_electricity'], switch, years)

    for ff in ffuels:

        stock_adj_new = merge_switch(new_stock[ff], switch, years)

        if ff == 'Net_electricity':

            continue

        else:

            change = stock_adj_new.multiply(
                ff_switch_fraction[years], fill_value=0
                )

            new_stock[ff].sort_index(inplace=True)

            new_stock[ff].update(stock_adj_new.subtract(change, 
                                                        fill_value=0))

            elect_stock_adj_new = \
                elect_stock_adj_new.add(change,
                                               fill_value=0)

    new_stock['Net_electricity'].update(elect_stock_adj_new)

    return new_stock
