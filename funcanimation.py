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

fig, ax = plt.subplots(nrows=2,ncols=2)
#fig1, ax1 = plt.subplots(nrows=2,ncols=2)

# Trajectory data from Shenzhen
data = io.loadmat('finalShenzhen9386V6.mat')
lats = data.get('lat')  # Latitudes
longs= data.get('long') # Longitudes

# Total number of GPS samples for each car
numSamples   = lats.shape[0]

# We can choose an arbitrary subset of cars. Specify the corresponding 
# the column numbers in indicesOfCarsPlotted
indicesOfCarsPlotted  =  range(8000,8150)
numCars               =  len(indicesOfCarsPlotted)

# Time interval between successive frames for rendering in milliseconds.
dtFrame = 40

# To turn of the visualization of the trajectory trail set this to zero
# These are alpha values which should lie in the interval [0,1]
lineTransparency   = 0.15
markerTransparency = 1.0

# Extract only the columns corresponding to the cars we are interested in
lats  = lats  [ np.ix_( range(0, numSamples) , indicesOfCarsPlotted   ) ]
longs = longs [ np.ix_( range(0, numSamples) , indicesOfCarsPlotted   ) ]

print "Finished extracted the data..."

#-------------------------------------------------------------------------

# For each car create a trajectory object. 
trajectories = []
for i in range(numCars):

    linecolor = getRandomColor()
    linecolor = ( linecolor[0], linecolor[1], linecolor[2] , lineTransparency) # Augment with a transparency
    markercolor = (linecolor[0], linecolor[1], linecolor[2], markerTransparency)

    line, = ax[0][0].plot([],[], lw=3, markerfacecolor=markercolor, markersize=5)

    line.set_marker('o')
    line.set_c(linecolor)

    trajectories.append(line)


line = matplotlib.lines.Line2D([],[],marker='s')
ax[0][1].add_line(line)
ax[0][1].set_xlim(0,2*np.pi)
ax[0][1].set_ylim(-6,6)
ax[0][1].set_title("Hello!")

# X and Y axis limits
[xmin, xmax] = [ min(lats.ravel() ), max( lats.ravel() )   ]
[ymin, ymax] = [ min(longs.ravel()), max(longs.ravel() ) ]
# A special dumb initial function.
# Absolutely essential if you do blitting
# otherwise it will call the generator as an
# initial function, leading to trouble
def init():
    global ax
    print "Initializing "


# Update the state of rGather
def rGather():
    """ Run the online r-gather algorithm as the cars
    move around. TODO: Make this function itself call
    another generator which is revealing the data piece
    by piece. Generators all the way down! Chaining of
    several functions and lazy evaluation!!
    """
    for i in range(numSamples):
        for car in range(numCars):
            xdata = lats [0:i+1,car]
            ydata = longs[0:i+1,car]
            trajectories[car].set_data( xdata, ydata )

        yield trajectories, i


# Separating the animateData and the rGather generator function allows
def animateData(state, fig, ax):
    """ Render the trajectories rendered by the rGather algorithms
    and add fancy effects.
    """
    trajectories = state[0] # All trajectories
    currentTime  = state[1] # The time at which to animate

    if currentTime > 1:
        for car in range(len(trajectories)):
            trajectories[car].set_markevery(  (currentTime,currentTime)  )

    ax[0][0].set_xlim(xmin, xmax)
    ax[0][0].set_ylim(ymin, ymax)
    ax[0][0].set_title(str(len(trajectories)) + ' Trajectories'+ '\nTime: ' + str(currentTime) + ' of ' + str(numSamples), fontdict={'fontsize':20})
    ax[0][0].set_xlabel('Latitude', fontdict={'fontsize':20})
    ax[0][0].set_ylabel('Longitude', fontdict={'fontsize':20})


    xdata = np.arange(0,2*np.pi,0.1)
    ydata = 5*np.sin(xdata) + np.random.rand(len(xdata))

    lineset = ax[0][1].get_lines()
    lineset[0].set_data(xdata,ydata)
    lineset[0].set_linestyle('-')



# Call the animator.  blit=True means only re-draw the parts that have changed.
# Ensures better speed

anim = animation.FuncAnimation(fig, animateData, rGather(),
                               init_func=init, interval=dtFrame, fargs=(fig,ax))
anim.save('shenzen_animation_' +  str(numCars)  + '_cars.mp4', fps=2, extra_args=['-vcodec', 'libx264'])
plt.show()
