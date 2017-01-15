#!/home/gaurish/anaconda2/bin/python2
# e.g. ./mkVideo.py -r 20 -range 20 100 -samples 60 Other tunables can be found in the argumentParser function
# Run the r-Gather algorithm on the initial supplied input data. Different values of r, can be tried
# out interactively once the matplotlib window opens. If you want to change the data, you will have to
# restart code, and provide the desired range as matplotlib arguments.
import rGather as rg
import utilities as ut
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math
import sys, argparse
from termcolor import colored

print colored("Script started", 'white','on_blue', ['bold'])

# Get input data
args = ut.argumentParser().parse_args()
[r, trajectories, lats, longs ] = ut.interpretCommandLineArguments(args, inputFile='../DynamicInput/finalShenzhen9386V6.mat') 


run            = rg.Algo_Dynamic_4APX_R2_Linf( r= r, pointCloud= trajectories) 
clusterCenters = run.generateClusters() 


# Set up plots with the resulting output data.
fig, ax = plt.subplots(1,2)
run.plotClusters( ax[0], trajThickness=2 ) 


keyPress     = ut.wrapperkeyPressHandler(fig, ax, run, lats, longs)
fig.canvas.mpl_connect('key_press_event', keyPress   )
plt.show()
