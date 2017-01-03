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
import numpy as np, math
import sys, argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl, colorsys
from scipy import io

# We can choose an arbitrary subset of cars. Specify the corresponding the column numbers in indicesOfCarsPlotted
indicesOfCarsPlotted  = range(9000,9040)
nbrFile               = './DynamicInput/distIdx_cars_40_samples_40_@9000_9040.yaml'
numSamples            = 40 # all_lats.shape[0] # Total number of GPS samples for each car. 
r                     = 3



def checkTriangleInequality(points, distFn):
	""" Check if the triangle inequality is satisfied.
	pairwise for the given points. Iterate through all 
        (N,3) combinations using the python;s combinatorics module.
        """
        import itertools as it
        from termcolor import colored 

	numpts = len(points)
	d      = distFn

	for (i,j,k) in it.permutations(range(numpts), 3):
		(u,v,w) = (points[i], points[j], points[k])
		triangle_ineq_verify = d(u,w)  < d(u,v) + d(v,w)

                if triangle_ineq_verify == False:

			print colored('Failure','white','on_red',['bold'])
			print 'Points are ', u,v,w
			sys.exit()
                else:
		    # The last three should be zero.
		    print str([i,j,k]), '  ', d(u,w), ' ' , (d(u,v) + d(v,w)), ' ' , d(u,u),  ' ' , d(v,v), ' ', d(w,w)



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

# Traverse the excel sheet column by column
# The set of trajectories is [ [(Double,Double)] ] i.e. list of lists of tuples. 
# number of GPS samples taken for the ith car. For shenzhen data set this is constant for all cars.
trajectories = []
for car in range(numCars): # Columns
    trajectories.append([]) # Create an empty entry which will contain the (x,y) coordinates of the trajectories of the car of the current loop
    for t in range(numSamples): # Rows
         (x,y) = (lats[t][car], longs[t][car])
         trajectories[car].append((x,y))# Append the gps coordinate of 'car' at time 't' to its trajectory.
    trajectories[car] = np.rec.array( trajectories[car], dtype=[('x', 'float64'),('y', 'float64')] )

trajectories = np.array(trajectories)

# Set the r-parameter and trajectories as points of the metric space.
run            = rg.AlgoJieminDynamic( r= r, pointCloud= trajectories , memoizeNbrSearch=False, distances_and_indices_file= nbrFile) 
clusterCenters = run.generateClusters() # Generate clusters. Each cluster center itself is a trajectory.
print run.computedClusterings

# Colour all trajectories in one group with the same colour. 
fig, ax = plt.subplots()
run.plotClusters( ax, trajThickness=6 ) 
plt.show()
