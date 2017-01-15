import rGather as rg
import numpy as np, math
import sys, argparse, re
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl, colorsys
from scipy import io
import os.path
from termcolor import colored
import argparse
print colored("Script started",'yellow','on_blue')

# https://docs.python.org/2/library/argparse.html#module-argparse
parser = argparse.ArgumentParser(prog='mainGuiDynamic.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-memFlag' , type=bool, default=True, help='Memoize the neighbor search')
parser.add_argument('-range'   , nargs=2  , type=int    , default=[9000, 9050], help='Ending Index from Shenzen data set')
parser.add_argument('-samples' , type=int , default=20  , help='Number of Samples')
parser.add_argument('-r'       , type=int , default=2   , help='Minimum number of elements per cluster')
args = parser.parse_args()

#We can choose an arbitrary subset of cars. Specify the corresponding the column numbers in indicesOfCarsPlotted
memFlag               = args.memFlag
indicesOfCarsPlotted  = range(args.range[0], args.range[1]) # This can be an arbitrary selection of column indices if you wish
numSamples            = args.samples # All_lats.shape[0] # Total number of GPS samples for each car. 
r                     = args.r


# Colour all trajectories in one group with the same colour. 
fig, ax = plt.subplots(1,2)

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

trajectories = []
for car in range(numCars): # Columns
    trajectories.append([]) # Create an empty entry which will contain the (x,y) coordinates of the trajectories of the car of the current loop
    for t in range(numSamples): # Rows
         (x,y) = (lats[t][car], longs[t][car])
         trajectories[car].append((x,y))# Append the gps coordinate of 'car' at time 't' to its trajectory.
    trajectories[car] = np.rec.array( trajectories[car], dtype=[('x', 'float64'),('y', 'float64')] )

trajectories = np.array(trajectories)


# Set the r-parameter and trajectories as points of the metric space.
# The two approximation algorithm
run1            = rg.AlgoJieminDynamic( r= r, pointCloud= trajectories , memoizeNbrSearch=memFlag) 
clusterCenters  = run1.generateClusters() # Generate clusters. Each cluster center itself is a tra1ectory.
run1.plotClusters( ax[0], trajThickness=2 ) 

# The 4-approximation decentralized algorithm
run2            = rg.Algo_Dynamic_4APX_R2_Linf ( r= r, pointCloud= trajectories, memoizeNbrSearch=memFlag) 
clusterCenters  = run2.generateClusters() 
run2.plotClusters( ax[1], trajThickness=2 ) 


#--------------------------------Plotting area
def wrapperkeyPressHandler( fig, ax, keyStack=[] ): # the key-stack argument is mutable! I am using this hack to my advantage.
    def _keyPressHandler(event):
        if event.key in ['r', 'R']: # Signal to start entering an r-value

            run1.clearComputedClusteringsAndR() # Neighbor map remain intact.
            run2.clearComputedClusteringsAndR() # Neighbor map remain intact.
            keyStack.append('r')

        elif event.key in ['0','1','2','3','4','5','6','7','8','9'] and \
             len(keyStack) >= 1                                     and \
             keyStack[0] == 'r': # If the 'r'-signal has already been given, accept only numeric characters

             keyStack.append(event.key)
             print event.key

        elif event.key == 'enter': # Give the signal to interpret the numbers on the stack

           rStr = ''
           for elt in keyStack[1:]:
               if elt in ['0','1','2','3','4','5','6','7','8','9'] :
                   rStr += elt
               else:
                   break # You just hit an non-numeric character entered by mistake

           r = int(rStr)
           print "R-value interpreted: ", r
           keyStack[:] = [] # Empty for further integers.

           # Run the algorithm again, with the new r. 
           run1.r = r
	   run2.r = r
           clusterCenters = run1.generateClusters() # Generate clusters. Each cluster center itself is a trajectory.
	   run2.generateClusters()
	   #print run.computedClusterings
	   ax[0].cla() # Clear the previous canvas
	   ax[1].cla()
           run1.plotClusters( ax[0], trajThickness=2 ) 
           run2.plotClusters( ax[1], trajThickness=2 ) 
           fig.canvas.draw()

    return _keyPressHandler

keyPress     = wrapperkeyPressHandler(fig, ax)
fig.canvas.mpl_connect('key_press_event', keyPress   )
plt.show()
