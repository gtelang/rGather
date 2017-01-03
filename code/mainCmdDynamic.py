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

# Things to do
# Make the findNearestNeighborSearch in base class depend on both the 2R and the k nearest neighbors via options.
# Given kwarg arguments.
# Ensure that brute force search is kept.

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


def checkTriangleInequality(points, distFn):
	""" Check if the triangle inequality is satisfied.
	pairwise for the given points. Iterate through all 
        (N,3) combinations using the python;s combinatorics module.
        """
        import itertools as it
        from termcolor import colored 

	numpts = len(points)
	d      = distFn

	for (i,j,k) in it.combinations(range(numpts), 3):
		(u,v,w) = (points[i], points[j], points[k])
		triangle_ineq_verify = d(u,w)  < d(u,v) + d(v,w)

                if triangle_ineq_verify == False:

			print colored('Failure','white','on_red',['bold'])
			print 'Points are ', u,v,w
			sys.exit()
                else:
		    # The last three should be zero.
		    print str([i,j,k]), '  ', d(u,w), ' ' , (d(u,v) + d(v,w)), ' ' , d(u,u),  ' ' , d(v,v), ' ', d(w,w)
			


fig, ax = plt.subplots()
# We can choose an arbitrary subset of cars. Specify the corresponding
# the column numbers in indicesOfCarsPlotted
indicesOfCarsPlotted  = range(7000,7020)
numSamples            = 5 # all_lats.shape[0] # Total number of GPS samples for each car. Is also equal to the number of time-stamps
                           # For the dyunamic algorithm, you might need to choose only a small subset of the time-stamps.
numCars               = len(indicesOfCarsPlotted) # Total number of cars selected to run the data on.
r                     = 3

# Trajectory data from Shenzhen
data = io.loadmat('DynamicInput/finalShenzhen9386V6.mat')
all_lats   = data.get('lat')  # Latitudes of ALL cars in the data
all_longs  = data.get('long') # Longitudes of ALL cars in the data

# Extract only the columns corresponding to the cars we are interested in
# The data-set provided is massive!!! That's why we have selecte only a small
# subset of cars as indicated above. This also gives you a very natural need for
# a concept like a core-set. 

lats  = all_lats  [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]
longs = all_longs [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]

trajectories = []
# Traverse the excel sheet column by column
# The set of trajectories is [ [(Double,Double)] ] i.e. list of lists of tuples. 
# len(trajectories) = number of cars
# len(trajectories[i]) = number of GPS samples taken for the ith car. For shenzhen data set this is
# constant for all cars.
for car in range(numCars): # Columns
    trajectories.append([]) # Create an empty entry which will contain the (x,y) coordinates of the trajectories of the car of the current loop
    for t in range(numSamples): # Rows
         (x,y) = (lats[t][car], longs[t][car])
         trajectories[car].append((x,y))# Append the gps coordinate of 'car' at time 't' to its trajectory.
    trajectories[car] = np.rec.array( trajectories[car], dtype=[('x', 'float64'),('y', 'float64')] )

trajectories = np.array(trajectories)
#print trajectories.shape
#print trajectories

#run = rg.AlgoJieminDynamic( r=3, pointCloud = trajectories ) # Set the r-parameter and trajectories as points of the metric space.

# This is the correct way to test the benchmark function.
# trajectories[i] is the trajectory of the ith car. 
#run.dist(trajectories[0], trajectories[1])

#checkTriangleInequality(trajectories, run.dist)
#sys.exit()
#trajectories = np.array( trajectories ) # Yay! Input done!
#print trajectories
#---------------------------------------------------------------------------------------------------
# Plot the trajecory of each car.
#for car in range(numCars):
	# xdata = [point[0] for point in trajectories[car]]
	# ydata = [point[1] for point in trajectories[car]]

	# #print "Gold", trajectories[car]
	# #print "Bench", zip(xdata, ydata)
	
	# line, = ax.plot( xdata, ydata, 's-')
        # line.set_color(np.random.rand(3,1))

#plt.show()
run = rg.AlgoJieminDynamic( r=r, pointCloud = trajectories ) # Set the r-parameter and trajectories as points of the metric space.
clusterCenters = run.generateClusters() # Generate clusters. Each cluster center itself must be a trajectory.
print run.computedClusterings
# Colour all trajectories in one group with the same colour. 
# You can also animate them in time nicely. So you can have two
# kinds of routines, a static routine and a dynamic-animation
# routime. You can pass a flag depending on what you want.
run.plotClusters( ax, trajThickness=6 ) 

# mark the centers.
#latsCenters  = [ lats[index] for index in clusterCenters  ]
#longsCenters = [ longs[index] for index in clusterCenters  ]
#ax[0].plot( latsCenters, longsCenters, 'ro', markersize=10)

plt.show()
