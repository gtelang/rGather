import rGather as rg
import numpy as np, math
import sys, argparse, re
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl, colorsys
from scipy import io
import os.path
from termcolor import colored
import argparse
print colored("Script started",'yellow','on_blue')
mpl.rcParams['legend.fontsize'] = 10

# https://docs.python.org/2/library/argparse.html#module-argparse
parser = argparse.ArgumentParser(prog='mainGuiDynamic.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-memFlag' , type=bool, default=True    , help='Memoize the neighbor search')
# parser.add_argument('-plotxyt' , type=bool, default=True    , help='Plot in xy-t space. Yes or no') #### TODO: Fix this. somehow the parser does not acept this.
parser.add_argument('-range'   , nargs=2  , type=int        , default=[9000, 9020], help='Ending Index from Shenzen data set')
parser.add_argument('-samples' , type=int , default=20      , help='Number of Samples')
parser.add_argument('-r'       , type=int , default=2       , help='Minimum number of elements per cluster')
parser.add_argument('-algo'    , type=str , default='4apx'  , help='The algorithm for clustering trajectories')
args = parser.parse_args()

#We can choose an arbitrary subset of cars. Specify the corresponding the column numbers in indicesOfCarsPlotted
memFlag               = args.memFlag
indicesOfCarsPlotted  = range(args.range[0], args.range[1]) # This can be an arbitrary selection of column indices if you wish
numSamples            = args.samples # All_lats.shape[0] # Total number of GPS samples for each car. 
r                     = args.r
algo                  = args.algo
plotxyt               = True

if plotxyt == True: # plot in xy-t space
	
   fig = plt.figure()
   ax  = fig.gca(projection='3d') # key-step!
   print "plotting xy-t"
   
elif plotxyt == False:# plot in xy space
   fig, ax = plt.subplots()
   print "plotting xy"

else:
    print "Please mention if you want to plot in xyt space or xy space."
    sys.exit()
   
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

if algo == '4apx':

   # Set the r-parameter and trajectories as points of the metric space.
   run            = rg.Algo_Dynamic_4APX_R2_Linf( r= r, pointCloud= trajectories , memoizeNbrSearch=memFlag) 
   run.generateClustersSimple(config ={'mis_algorithm':'networkx_random_choose_20_iter_best' }) 
   
elif algo == '2apx':

   # Set the r-parameter and trajectories as points of the metric space.
   run = rg.AlgoJieminDynamic( r= r, pointCloud= trajectories , memoizeNbrSearch=memFlag) 
   run.generateClusters() 
	
else:
    print "Algorithm Option not recognized"
    sys.exit()


run.plotClusters( ax, trajThickness=2 , plot_xytspace = plotxyt) 
ax.legend()
ax.set_axis_on()
plt.show()
