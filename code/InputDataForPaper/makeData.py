import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys 
import numpy as np
from scipy import io
import math
import sys


# Trajectory data from Shenzhen
data = io.loadmat('finalShenzhen9386V6.mat')
lats = data.get('lat')  # Latitudes
longs= data.get('long') # Longitudes


# there are 288 rows in the data-file
timeSliceIndex = 144

# We can choose an arbitrary subset of cars. Specify the corresponding 
# the column numbers in indicesOfCarsPlotted
indicesOfCars  =  range( 3000 , 3060 )
numCars        =  len(indicesOfCars)

f = open("shenzhenCars60.txt", "w")

print "Writing to file"
for i, index in  zip(  range(len(indicesOfCars)), indicesOfCars  ):

    # Coordinates of car labelled index
    x = lats[timeSliceIndex][index]
    y = longs[timeSliceIndex][index]
    f.write( str(i+1) + ' ')
    f.write( str(x) + ' ')
    f.write( str(y) + '\n')



