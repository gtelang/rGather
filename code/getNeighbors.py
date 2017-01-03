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
import yaml
from termcolor import colored 

indicesOfCarsPlotted  = range(7000,7010)
numSamples            = 10
numCars               = len(indicesOfCarsPlotted) 

# Trajectory data from Shenzhen
data = io.loadmat('DynamicInput/finalShenzhen9386V6.mat')
all_lats   = data.get('lat')  # Latitudes of ALL cars in the data
all_longs  = data.get('long') # Longitudes of ALL cars in the data

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
    #trajectories[car] = np.rec.array( trajectories[car], dtype=[('x', 'float64'),('y', 'float64')] )

trajectories = np.array(trajectories)


print colored('Making Table', 'white', 'on_magenta', ['bold'])
run        = rg.AlgoJieminDynamic( r=None, pointCloud = trajectories ) # Set the r-parameter and trajectories as points of the metric space.
pointCloud = trajectories
numpts     = len(pointCloud)
distances  = []
indices    = []

for i in range(numpts):
	print colored ('Calculating distance from '+str(i), 'white', 'on_magenta',['underline','bold']) 
        traj_i = pointCloud[i]
        distances_and_indices = []

        for j in range(numpts):
                
            traj_j = pointCloud[j]
            dij = run.dist( traj_i, traj_j)
            distances_and_indices.append((dij,j))
	    print '......to j= '  , j, '  dij= ', dij
		     
        # Now sort the distances of all points from point i. 
        distances_and_indices.sort(key=lambda tup: tup[0]) # http://tinyurl.com/mf8yz5b
	    
        distances.append( [ d   for (d,idx) in distances_and_indices ]  )
        indices.append  ( [ idx for (d,idx) in distances_and_indices ]  )

print colored('Finished Brute Force Neighbors !!', 'white', 'on_green', ['bold', 'underline'])

#print np.array(distances)
#print np.array(indices)

stream = file('distances_and_indices.yaml', 'w')
#print yaml.dump_all(distances, stream)
#print yaml.dump_all(indices, stream)

print yaml.dump({'Distances':distances, 'indices':indices}, stream)


stream = open('distances_and_indices.yaml', 'r')
print yaml.load(stream)
