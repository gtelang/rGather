#!/home/gaurish/anaconda2/bin/python2
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
import sys, argparse, re
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl, colorsys
from scipy import io
import os.path
from termcolor import colored
print colored("Script started",'yellow','on_blue')
import argparse

# https://docs.python.org/2/library/argparse.html#module-argparse
parser = argparse.ArgumentParser(prog='mainGuiDynamic.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-memFlag' , type=bool, default=True, help='Memoize the neighbor search')
parser.add_argument('-range'   , nargs=2  , type=int    , default=[9000, 9050], help='Ending Index from Shenzen data set')
parser.add_argument('-samples' , type=int , default=20  , help='Number of Samples')
parser.add_argument('-r'       , type=int , default=2   , help='Minimum number of elements per cluster')
args = parser.parse_args()

#We can choose an arbitrary subset of cars. Specify the corresponding the column numbers in indicesOfCarsPlotted
memFlag               = args.memFlag
indicesOfCarsPlotted  = range(args.range[0], args.range[1])#range(350, 400)
numSamples            = args.samples # All_lats.shape[0] # Total number of GPS samples for each car. 
r                     = args.r

#Extract integers from the file-name. From http://stackoverflow.com/a/4289348
# nbrFile ='samples_10_@90_160.yaml' 
# (numSamples, startIdx, endIdx) = map(int, re.findall(r'\d+', nbrFile))
# indicesOfCarsPlotted           = range(startIdx, endIdx)
# r = None

# print numSamples, startIdx, endIdx
# print len(indicesOfCarsPlotted)
# # If the neighbor file-does not yet exist create it.
# if not os.path.isfile(nbrFile) :
#     print colored("Oops...Neighbor file does not exist...Creating it....", 'white', 'on_red', ['bold'])
#     from subprocess import call
#     os.chdir('./DynamicInput')
#     call(["./getNeighbors.py", str(numSamples), str(startIdx), str(endIdx)])
#     os.chdir('..')

# Colour all trajectories in one group with the same colour. 
fig, ax = plt.subplots()

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
run            = rg.AlgoJieminDynamic( r= r, pointCloud= trajectories , memoizeNbrSearch=memFlag) 
clusterCenters = run.generateClusters() # Generate clusters. Each cluster center itself is a tra1ectory.
#print run.computedClusterings
run.plotClusters( ax, trajThickness=2 ) 

#--------------------------------Plotting area
def wrapperkeyPressHandler( fig, ax, keyStack=[] ): # the key-stack argument is mutable! I am using this hack to my advantage.
    def _keyPressHandler(event):
        if event.key in ['r', 'R']: # Signal to start entering an r-value

            run.clearComputedClusteringsAndR() # Neighbor map remain intact.
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
           run.r = r
           clusterCenters = run.generateClusters() # Generate clusters. Each cluster center itself is a trajectory.
	   #print run.computedClusterings
	   ax.cla() # Clear the previous canvas
           run.plotClusters( ax, trajThickness=2 ) 

           fig.canvas.draw()

        elif event.key == 'a': # Animate the clusters
		ax.cla()
		run.animateClusters(ax, fig, lats, longs)
		fig.canvas.draw()

    return _keyPressHandler


keyPress     = wrapperkeyPressHandler(fig, ax)
fig.canvas.mpl_connect('key_press_event', keyPress   )
plt.show()
