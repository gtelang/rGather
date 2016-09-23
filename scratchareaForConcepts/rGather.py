#!/usr/bin/ipython
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from scipy import io # For reading MATLAB's binary data format
import math
#----------------------------------------------------------
# Load trajectory data and visualize the movement of cars.
#---------------------------------------------------------

# Trajectory data of 9386 cars driving around Shenzhen for 1 day.
# Sampling of the coordinates is done every 5 minutes.
data           = io.loadmat('finalShenzhen9386V6.mat')


# latitudes and longitudes are 2 dimensional arrays.
# latitudes[i] and longitudes[i] are respectively the x and y
# co-ordinates of the cars as measured at sampling_rate*i minutes
lats       = data.get('lat')
longs      = data.get('long')


# The ith column in lats and longs are respectively  the latitudes and longitudes of the
# ith car at different times of the day
numCars          = lats.shape[1] # i.e the number of columns
numSamplesPerCar = lats.shape[0] # Number of samples taken for each car.

cars = []
for i in range(numCars):

    # Get the latitude and longitude of the ith car in python
    latCar  = lats [:,i]
    longCar = longs[:,i]

    coordinatesCar = []
    for x, y in zip(latCar, longCar):
        coordinatesCar.append( (x,y) )

    cars.append(  coordinatesCar     )
    print "Extracted coordinates for car ", i

#---------------------------------------------------------------------------------
# Plot the coordinates of each car. Here, the lats and longs array are useful
#--------------------------------------------------------------------------------

# Axis limits as determined from the full data set
[xmin, xmax] = [ min(lats.ravel() ), max( lats.ravel() )   ]
[ymin, ymax] = [ min(longs.ravel()), max(longs.ravel() ) ]

fig, ax = plt.subplots()
ax.set_title('Trajectories', fontdict={'fontsize':40})
ax.set_xlabel('Latitude', fontdict={'fontsize':20})
ax.set_ylabel('Longitude', fontdict={'fontsize':20})

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin,ymax)

endTimeForPlotting = numSamplesPerCar
numberOfCarTrajectories =  numCars 
for car in range(0,numberOfCarTrajectories):

    ax.plot( lats[0:endTimeForPlotting,car],
             longs[0:endTimeForPlotting,car], 'o-', lw=3, alpha=0.2 )


#----------------------------------------------------
# Draw a light grid, with major and minor grid lines
#----------------------------------------------------
[numXticksMajor, numXticksMinor]  = [ 10, 30]
[numYticksMajor, numYticksMinor]  = [ 10, 30]

major_xticks = [xmin + k*(xmax-xmin)/numXticksMajor for k in range( numXticksMajor )] 
minor_xticks = [xmin + k*(xmax-xmin)/numXticksMinor for k in range( numXticksMinor )] 

major_yticks = [ymin + k*(ymax-ymin)/numYticksMajor for k in range( numYticksMajor )] 
minor_yticks = [ymin + k*(ymax-ymin)/numYticksMinor for k in range( numYticksMinor )] 

# set your ticks manually
ax.xaxis.set_ticks( major_xticks  )
ax.xaxis.set_ticks( minor_xticks, minor=True)

# set your ticks manually
ax.yaxis.set_ticks( major_yticks  )
ax.yaxis.set_ticks( minor_yticks, minor=True)

# or if you want different settings for the grids:
ax.grid(b='True',which='minor', alpha=0.3, color= [0.3, 0.3, 0.3] , linestyle='--')
ax.grid(b='True',which='major', alpha=0.6, color= [0.3, 0.3, 0.3] , linestyle='--')


plt.savefig('shenzhen_trajectory_data.eps', format='eps')
plt.show()




# Plot the cars as if they were actually moving along the trajectories
# The two main functions for accomplishing animation are ArtistAnimation
# and FuncAnimation from the matplotlib.animations module.  It is possible
# the former might take up more space, but the animation will be rendered
# much quicker. It requires the artist objects to be rendered already and
# then drawn. In funcanimation, the frames are plotted on the fly.
# saving of space. In either case, the technique of blitting which I will
# implement later, will speed up the animation. 

# # We first create a dumb animation here, that of a particle moving randomly
# # in a square, but each link of the trajectory takes the same time to traverse.
# # Thus the speeds of motion will have to be adjusted accordingly. 

# fig = plt.figure()
# ax  = plt.axes( xlim=(0,1), ylim = (0,1))
# line, = ax.plot([], [], lw=2, alpha=0.2)

# numSeconds    =  10
# coods         =  np.random.rand(numSeconds,2)


# # These two arrays will contain the x-y coordinates of the trail
# # For the purpose of drawing.
# xs_trail = [  ]
# ys_trail = [  ]

# # Initial position and time
# current_x = coods[0,0]
# current_y = coods[0,1]

# # Increment position in small intervals of time, for a smooth visualization
# current_time =  0.0
# dt           =  0.01


# def init():
#     line.set_data([], [])
#     return line,


# def animate(i):


#     global current_x
#     global current_y
#     global current_time


#     # Which time interval am I in? Important to round to an integer
#     k   = int(math.floor(current_time))

#     # Get the velocity of this interval
#     vx  = (coods[k+1,0] - coods[k,0])
#     vy  = (coods[k+1,1] - coods[k,1])

#     # Punch in your current time! Waqt kabhi nahi rukta!!
#     current_time = current_time + dt

#     next_x = current_x + dt*vx
#     next_y = current_y + dt*vy


#     # Yay, safe to reset!!!
#     current_x = next_x
#     current_y = next_y
    
#     # Print slope
#     print (current_y - coods[k,1]) / (current_x - coods[k,0])
    
#     # I don't like the i+1 here: Reminds me too much of matlab's notation.
#     xs_trail = coods[0:k+1,0].tolist() + [current_x]
#     ys_trail = coods[0:k+1,1].tolist() + [current_y]
    
#     line.set_data(xs_trail, ys_trail)
#     line.set_marker("o")
#     return line,






# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate,
#                                init_func=init,
#                                interval=80,
#                                frames=int(numSeconds/dt - 1),
#                                blit='True',
#                                repeat = False ) # interval draws every interval milliseconds.


# # Plot the original points and make sure you have hit them as intended
# plt.plot(coods[:,0],coods[:,1],'rs-',alpha=0.1)

#plt.show() 
