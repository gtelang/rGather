
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from scipy import io 
import math

data = io.loadmat('finalShenzhen9386V6.mat') # Trajectory data from Shenzhen 
lats = data.get('lat')  # Latitudes
longs= data.get('long') # Longitudes

numCars          = lats.shape[1] # Number of columns
numSamplesPerCar = lats.shape[0] # Number of rows

cars = [] # cars[i] will be trajectory data for car i 
for i in range(numCars):

    latCar  = lats [:,i]
    longCar = longs[:,i]

    coordinatesCar = []
    for x, y in zip(latCar, longCar):
        coordinatesCar.append( (x,y) )

    cars.append(  coordinatesCar     )
    print "Set co-ordinates for car ", i

[xmin, xmax] = [ min(lats.ravel() ), max( lats.ravel() )   ]
[ymin, ymax] = [ min(longs.ravel()), max(longs.ravel() ) ]

fig, ax = plt.subplots()
ax.set_title('Trajectories', fontdict={'fontsize':40})
ax.set_xlabel('Latitude', fontdict={'fontsize':20})
ax.set_ylabel('Longitude', fontdict={'fontsize':20})

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin,ymax)
