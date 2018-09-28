# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:03:37 2017

@author: cmcmilla
"""


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



# Implement DBSCAN
# Two parameters: min_samples and eps.
# From sklearn documentation: Higher min_samples or lower eps indicate higher
# density required to  form a cluster.
# ...Dataset such that there exist min_samples other samples within a distance 
# of eps, which are defined as neighbors of the core sample. This tells us that 
# the core sample is in a dense area of the vector space. 

# Data are first standardized with StandardScalar to remove mean and
# set variance = 1.


# Implement DBSCAN clustering
db = DBSCAN(eps=0.01, min_samples=10).fit(cla_input_dict[6]['cla_array'])
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]:
              
db_eu = DBSCAN(eps=0.01, min_samples=10).fit(cla_inputeu['cla_array'])
labels_eu = db_eu.labels_
n_clusters_eu_ = len(set(labels_eu)) - (1 if -1 in labels_eu else 0)
