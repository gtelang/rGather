#!/usr/bin/ipython
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from scipy import io # For reading MATLAB's binary data format
import math

def main():

    # Keep this as a state: This is the incoming trajectory
    class myClass:
        def __init__(self, initial):
            self.state = initial

        def modify(self, num):
            self.state.append(num)


    gaurish = myClass([]); # State is an empty array

    fig = plt.figure()
    ax  = plt.axes( xlim=(0,1), ylim = (0,1))
    line, = ax.plot([], [], lw=2, alpha=0.2)

    numSeconds    =  10
    coods         =  np.random.rand(numSeconds,2)

    # These two arrays will contain the x-y coordinates of the trail
    # For the purpose of drawing.
    xs_trail = [  ]
    ys_trail = [  ]

    # Initial position and time
    current_x = coods[0,0]
    current_y = coods[0,1]

    # Increment position in small intervals of time, for a smooth visualization
    current_time =  5.7
    dt           =  0.1

    def init():
        line.set_data([], [])
        return line,

    def animate(i, gary):

        gary.modify(np.random.rand())

        print gary.state

        # # Which time interval am I in? Important to round to an integer
        
        # k   = int(math.floor(current_time))

        # # Get the velocity of this interval
        # vx  = (coods[k+1,0] - coods[k,0])
        # vy  = (coods[k+1,1] - coods[k,1])

        # # Punch in your current time! Waqt kabhi nahi rukta!!
        # current_time = current_time + dt


        # next_x = current_x + dt*vx
        # next_y = current_y + dt*vy


        # # Yay, safe to reset!!!
        # current_x = next_x
        # current_y = next_y

        # # Print slope
        # print (current_y - coods[k,1]) / (current_x - coods[k,0])

        # # I don't like the i+1 here: Reminds me too much of matlab's notation.
        # xs_trail = coods[0:k+1,0].tolist() + [current_x]
        # ys_trail = coods[0:k+1,1].tolist() + [current_y]

        # line.set_data(xs_trail, ys_trail)
        # line.set_marker("o")
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   fargs = ( gaurish,),
                                   init_func=init,
                                   interval=80,
                                   frames=5,
                                   blit='True',
                                   repeat = False ) # interval draws every interval milliseconds.


    # Plot the original points and make sure you have hit them as intended
    plt.plot(coods[:,0],coods[:,1],'rs-',alpha=0.1)
    plt.show()


if __name__ == "__main__":
    main()
