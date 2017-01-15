#!/home/gaurish/anaconda2/bin/python2
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
from utilities import wrapperkeyPressHandler

stop = sys.exit
parser = argparse.ArgumentParser(prog='mainGuiDynamic.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-memFlag' , type=bool, default=True, help='Memoize the neighbor search')
parser.add_argument('-range'   , nargs=2  , type=int    , default=[9000, 9020], help='Ending Index from Shenzen data set')
parser.add_argument('-samples' , type=int , default=20  , help='Number of Samples')
parser.add_argument('-r'       , type=int , default=2   , help='Minimum number of elements per cluster')
args = parser.parse_args()

#We can choose an arbitrary subset of cars. Specify the corresponding the column numbers in indicesOfCarsPlotted
memFlag               = args.memFlag
indicesOfCarsPlotted  = range(args.range[0], args.range[1]) # This can be an arbitrary selection of column indices if you wish
numSamples            = args.samples # All_lats.shape[0] # Total number of GPS samples for each car. 
r                     = args.r
numCars               = len(indicesOfCarsPlotted) # Total number of cars selected to run the data on.
data                  = io.loadmat('DynamicInput/finalShenzhen9386V6.mat')
all_lats              = data.get('lat')  # Latitudes of ALL cars in the data
all_longs             = data.get('long') # Longitudes of ALL cars in the data
lats                  = all_lats  [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]
longs                 = all_longs [ np.ix_( range(0, numSamples), indicesOfCarsPlotted) ]

trajectories = []
for car in range(numCars): # Columns
    trajectories.append([]) # Create an empty entry which will contain the (x,y) coordinates of the trajectories of the car of the current loop
    for t in range(numSamples): # Rows
         (x,y) = (lats[t][car], longs[t][car])
         trajectories[car].append((x,y))# Append the gps coordinate of 'car' at time 't' to its trajectory.
    trajectories[car] = np.rec.array( trajectories[car], dtype=[('x', 'float64'),('y', 'float64')] )

trajectories   = np.array(trajectories)
run            = rg.Algo_Dynamic_4APX_R2_Linf( r= r, pointCloud= trajectories , memoizeNbrSearch=memFlag) 
#clusterCenters = run.generateClusters(config ={'mis_algorithm':'sweep' }) 
clusterCenters = run.generateClusters() 

# Colour all trajectories in one group with the same colour. 
fig, ax = plt.subplots()
run.plotClusters( ax, trajThickness=2 ) 


keyPress     = wrapperkeyPressHandler(fig, ax, run)
fig.canvas.mpl_connect('key_press_event', keyPress   )
plt.show()
