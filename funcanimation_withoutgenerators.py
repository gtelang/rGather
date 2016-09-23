
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys 
import numpy as np
from scipy import io
import math
import sys

# For ensuring that the same sequence of colors are generated from run to run
np.random.seed(0)
# http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
# For creating visually distinct colors
def getRandomColor():
    # use golden ratio
    golden_ratio_conjugate = 0.618033988749895
    h = np.random.rand() # use random start value
    h += golden_ratio_conjugate
    h %= 1.0
    return colorsys.hsv_to_rgb(h, 0.7, 0.9)

fig = plt.figure()
ax  = plt.subplot(111)

# Trajectory data from Shenzhen
data = io.loadmat('finalShenzhen9386V6.mat')
lats = data.get('lat')  # Latitudes
longs= data.get('long') # Longitudes

# Total number of GPS samples for each car
numSamples   = lats.shape[0]

# We can choose an arbitrary subset of cars. Specify the corresponding 
# the column numbers in indicesOfCarsPlotted
indicesOfCarsPlotted  =  range(8000,8200)
numCars               =  len(indicesOfCarsPlotted)

# Time interval between successive frames for rendering in milliseconds.
dtFrame = 220

# To turn of the visualization of the trajectory trail set this to zero
# These are alpha values which should lie in the interval [0,1]
lineTransparency   = 0.1
markerTransparency = 1.0


# Extract only the columns corresponding to the cars we are interested in
lats  = lats  [ np.ix_( range(0, numSamples) , indicesOfCarsPlotted   ) ]
longs = longs [ np.ix_( range(0, numSamples) , indicesOfCarsPlotted   ) ]

print "Finished extracted the data..."


#-------------------------------------------------------------------------
trajectories = []
for i in range(numCars):

    linecolor = getRandomColor()
    linecolor = ( linecolor[0], linecolor[1], linecolor[2] , lineTransparency) # Augment with a transparency
    markercolor = (linecolor[0], linecolor[1], linecolor[2], markerTransparency)

    line, = ax.plot([],[], lw=3, markerfacecolor=markercolor, markersize=5)

    line.set_marker('o')
    line.set_c(linecolor)

    trajectories.append(line)

# X and Y axis limits
[xmin, xmax] = [ min(lats.ravel() ), max( lats.ravel() )   ]
[ymin, ymax] = [ min(longs.ravel()), max(longs.ravel() ) ]
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title('Trajectories', fontdict={'fontsize':40})
ax.set_xlabel('Latitude', fontdict={'fontsize':20})
ax.set_ylabel('Longitude', fontdict={'fontsize':20})


# Update the state of rGather
def update(i, *args):
   print i
   for car in range(numCars):
      xdata = lats [0:i+1,car]
      ydata = longs[0:i+1,car]
      trajectories[car].set_data( xdata, ydata )

      if i>1:
          trajectories[car].set_markevery(  (i,i)  )

   return trajectories

# Call the animator.  blit=True means only re-draw the parts that have changed.
# Ensures better speed
anim = animation.FuncAnimation(fig, update, interval=dtFrame, blit=True)
#anim.save('shenzen_animation.mp4', fps=2, extra_args=['-vcodec', 'libx264'])
plt.show()
