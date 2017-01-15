# Each plot compares two algorithms for clustering trajectories visually and in terms of the diamter and number of points per cluster
# ==========
# Checklist:
# ==========
# 1. Make sure that the scales are the same on both subplots of an ax for every ax
# 2. This code will just compare  two algorithms, visually, and statistically. The core engine for
#    extracting data remains the same. Which two algorithms need to be compared along with their parameters
#    are specified through the command-line or near the top of the file. Either are easy and the argparse module
#    will allow you to write robust command-line options. 
# 3. Useful function appendix
#    ax_clusters[0].axhline(y=0.5, color='r', linestyle='-') # Red vertical line. For RMS / maximum comparison
#    ax_clusters[1].axvline(x=0.5, color='b', linestyle='-') # Blue horizontal line
import matplotlib.pyplot as plt
import numpy as np
import rGather as rg
import sys
import time
import scipy as sp
from scipy import io
from   termcolor import colored
import itertools as it

#============================================================================================================
# Input Options
memFlag               = True
indicesOfCarsPlotted  = range(9000, 9050) # This can be an arbitrary selection of column indices if you wish
numSamples            = 20
r                     = 5
algo1                 = rg.Algo_Dynamic_4APX_R2_Linf
#algo2                 = pass


#=============================================================================================================

numCars = len(indicesOfCarsPlotted) # Total number of cars selected to run the data on.

# Trajectory data from Shenzhen
data      = io.loadmat('DynamicInput/finalShenzhen9386V6.mat')
all_lats  = data.get('lat')  # Latitudes of ALL cars in the data
all_longs = data.get('long') # Longitudes of ALL cars in the data

# Extract only the columns corresponding to the cars we are interested in
# I presume we can also extract only a subset of rows to speed up the computations.
# albeit give us a low-resolutiom image.
lats  = all_lats  [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]
longs = all_longs [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]

def diameter_of_point_set(distanceFn, points):
        """ distanceFn : is the distance function of a metric space
	    points     : is the set of points for which you want to compute the diameter
	The algorithm used here is just the dumb O(n^2) brute force
	Returns the diamter of a set of points in an arbitrary metric space.
        """
	diameter = 0.0
        for i, pt_a in enumerate(points):
		for pt_b in points[(i+1):]: # if i+1 is greater than array length, the list slicing returned will be empty.
		     d_ab = distanceFn (pt_a, pt_b)
		     if d_ab > diameter: 
			     diameter = d_ab # rebind

	return diameter

def rms (a):
    """ Return the root-mean-square of an array of numbers
    Found this nice "WYSIWIG"-like implementation here 
    http://stackoverflow.com/a/7433184
    """
    from numpy import mean, sqrt, square

    na= np.array(a) # np.array is an idempotent function. use this incase, a list was passed.
    return sqrt(mean(square(a)))

		     
# Initiate the figures and clusters to plot.
fig_clusters , ax_clusters  = plt.subplots(1,2)
fig_diameters, ax_diameters = plt.subplots(1,2)
fig_numpoints, ax_numpoints = plt.subplots(1,2)


plt.show()

