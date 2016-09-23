
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from time import sleep

# Critical to set the axes limits if you are adding
# line data. independently of the a call to ax.plot()!!
# Gave me a lot of headeaches.
fig, ax = plt.subplots()
line = mpl.lines.Line2D([],[],marker='s')
ax.add_line(line)
ax.set_xlim(0,2*np.pi)
ax.set_ylim(-6,6)

# The data here is returned by the generator
def animate(data, ax, mystring):
    lineset = ax.get_lines()
    line    = lineset[0]
    line.set_data(data[0],data[1])
    line.set_linestyle('-')
    ax.set_title(mystring)
    return ax.lines

def nextstep():
    """ As you can see, not a single line of code, has been
    about rendering to the screen! The power of yield!
    """
    print "Inside the generator, yay!"
    while True:
        xdata = np.arange(0,2*np.pi,0.1)
        ydata = 5*np.sin(xdata) + np.random.rand(len(xdata))
        yield (xdata, ydata)


def init():
    global ax
    print "Initializing"
    return ax.lines

mystring = "Hello World!"
ani = animation.FuncAnimation(fig, animate, nextstep, init_func= init, fargs=(ax,mystring), blit=True, interval=50)
plt.show()
