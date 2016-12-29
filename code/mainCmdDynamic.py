# In the paper we show that the 2-approximation rGather problem applies to
# trajectory data with zero regroupings. When we allow  max of k regroupings
# there is a nice, Dyanmic programming algorithm which we also implement.
# The MATLAB file provided by Prof. Gao can be thought of like an Excel sheet:
# The columns corresponds to cars, and rows corresponds to time.
# Each entry in the matrix, contains a tuple of x-y coordinates correponding to
# the positions of taxi-cabs in the R2 plane.
# As a mnemonic think 'c' for columns/cars and 'r' for rows/record. incase you
# forget. For the most part you will be focusing only on a submatrix of the
# full matrix. (Can we use some funky ideas related to core-sets on matrics I had seen before?)
# Since you will be using the 2Approx metric space algorithm at least for the case of zero regroupings
# each column correponds to a point in the metric spoce. Clearly, you will have to use list-slicing
# in this case. Just make note of that!!!!

import rGather as rg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import argparse
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys
from scipy import io
import math
import sys

np.random.seed(0)

print "Started Program"

fig, ax = plt.subplots(1,2)
ax[0].set_aspect( 1.0 )
ax[1].set_aspect( 1.0 ) # For recording some statistics and what not.

r  = 3 # Atleast 3 trajectories per cluster.

# We can choose an arbitrary subset of cars. Specify the corresponding
# the column numbers in indicesOfCarsPlotted
indicesOfCarsPlotted  =  range(9000,9005)
numCars               =  len(indicesOfCarsPlotted) # Total number of cars selected to run the data on.


# Trajectory data from Shenzhen
data = io.loadmat('DynamicInput/finalShenzhen9386V6.mat')
all_lats   = data.get('lat')  # Latitudes of ALL cars in the data
all_longs  = data.get('long') # Longitudes of ALL cars in the data
numSamples = 10 # all_lats.shape[0] # Total number of GPS samples for each car. Is also equal to the number of time-stamps
                           # For the dyunamic algorithm, you might need to choose only a small subset of the time-stamps.


# Extract only the columns corresponding to the cars we are interested in
# The data-set provided is massive!!! That's why we have selecte only a small
# subset of cars as indicated above. This also gives you a very natural need for
# a concept like a core-set. 


lats  = all_lats  [ np.ix_( range(0, numSamples) , indicesOfCarsPlotted   ) ]
longs = all_longs [ np.ix_( range(0, numSamples) , indicesOfCarsPlotted   ) ]

trajectories = []
# Traverse the excel sheet column by column
# The set of trajectories is [ [(Double,Double)] ] i.e. list of lists of tuples. 
# len(trajectories) = number of cars
# len(trajectories[i]) = number of GPS samples taken for the ith car. For shenzhen data set this is
# constant for all cars.
for car in range(0,1): # Columns
    trajectories.append([]) # Create an empty entry which will contain the (x,y) coordinates of the trajectories of the car of the current loop
    for t in range(numSamples): # Rows
         (x,y) = (lats[t][car], longs[t][car])
         trajectories[car].append((x,y))# Append the gps coordinate of 'car' at time 't' to its trajectory.



run = rg.AlgoJieminDynamic( r=r, pointCloud = trajectories )
clusterCenters = run.generateClusters() # Generate clusters

# Colour all trajectories in one group with the same colour. 
# You can also animate them in time nicely. So you can have two
# kinds of routines, a static routine and a dynamic-animation
# routime. You can pass a flag depending on what you want.
#run.plotClusters( ax[0], pointSize=40, annotatePoints=True ) 

# mark the centers.
#latsCenters  = [ lats[index] for index in clusterCenters  ]
#longsCenters = [ longs[index] for index in clusterCenters  ]
#ax[0].plot( latsCenters, longsCenters, 'ro', markersize=10)

#plt.show()
