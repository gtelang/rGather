
# To load the file generated with this code, do
# stream = open('distances_and_indices.yaml','r')
# filecontents = yaml.load(stream) # This will be a dictionary
import sys
sys.path.insert(0, '../') # the rGather module is not on the path. Hence add the directory.
import rGather as rg
from scipy import io
import numpy as np
import yaml
from termcolor import colored 

#gett arguments from terminal regarding car info. zeroth argument is the name of the script file.
(firstCarIdx, lastCarIdx, numSamples) = map(int,(sys.argv[1], sys.argv[2], sys.argv[3]))
# ----------------------------------------------------------------------

indicesOfCarsPlotted  = range(firstCarIdx, lastCarIdx)
nbrFile_prefix        = 'distIdx'
numCars               = len(indicesOfCarsPlotted) 

nbrFile               = nbrFile_prefix + '_cars_'    + str(numCars) \
                                       + '_samples_' + str(numSamples) \
                                       + '_@' + str(firstCarIdx) + '_' + str(lastCarIdx) \
                                       + '.yaml'


# Trajectory data from Shenzhen
data = io.loadmat('finalShenzhen9386V6.mat')
all_lats   = data.get('lat')  # Latitudes of ALL cars in the data
all_longs  = data.get('long') # Longitudes of ALL cars in the data

lats  = all_lats  [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]
longs = all_longs [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]

# Traverse the excel sheet column by column
# The set of trajectories is [ [(Double,Double)] ] i.e. list of lists of tuples. 
trajectories = [] # len(trajectories) = number of cars, and len(trajectories[i]) = number of GPS samples taken for the ith car. For shenzhen data set this is constant for all cars.
for car in range(numCars): # Columns
    trajectories.append([]) # Create an empty entry which will contain the (x,y) coordinates of the trajectories of the car of the current loop
    for t in range(numSamples): # Rows
         (x,y) = (lats[t][car], longs[t][car])
         trajectories[car].append((x,y))# Append the gps coordinate of 'car' at time 't' to its trajectory.

trajectories = np.array(trajectories)

print colored('Making Table', 'white', 'on_magenta', ['bold'])
run        = rg.AlgoJieminDynamic( r=None, pointCloud= trajectories ) # Just to be able to extract the distance function
pointCloud = trajectories
numpts     = len(pointCloud)
distances, indices  = ([], [])

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

stream = file( nbrFile , 'w') # Write all data to file.
print yaml.dump({'Metadata' : { 'Cars: '     : str(numCars), \
                                ' Samples: ': str(numSamples), \
                                ' Range:  ': str([firstCarIdx,lastCarIdx])}, 
                 'Distances': distances,
                 'Indices'  : indices}, stream)
