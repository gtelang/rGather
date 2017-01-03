import rGather as rg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import argparse
import pprint as pp

import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp

def givePoints (rbox_config):
     """ Get a distribution of n-points in D dimensions 
     with a pre-specified distribution. If you need to 
     plot a two dimensional point-cloud generated from this 
     you will need to unpack the matrix, columnwise and then 
     feed it to matplotlib. 

     This function is essentially a Python wrapper around 
     rbox, which generates point-clouds in various dimensions
     in different configurations, including grids, lenses, spirals etc.
     """
     rboxoutput = subprocess.check_output(rbox_config.split()).encode("utf-8")

     # first couple of lines are just metadata. Discard them.
     return np.array ([map (float,row.split())
		       for row in (rboxoutput.splitlines()[2:]) ])


def applyAxCorrection( ax ):

      ax.grid(b=True)
      ax.set_xlim([xlim[0], xlim[1]])
      ax.set_ylim([ylim[0], ylim[1]])
      ax.set_aspect(1.0)


def clearAxPolygonPatches( ax ):
    """ Now remove the patches which were rendered for each cluster
    Unfortunately, this step has to be done manually, the canvas patch
    of a cluster and the corresponding object in memory are not reactively
    connected.I presume, this behavioue can be achieved by sub-classing."""

    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()

    # Remove line patches. These get inserted during the r=2 case,
    # For some strange reason matplotlib does not consider line objects
    # as patches.
    ax.lines[:]=[]

    #pp.pprint (ax.patches) # To verify that none of the patches are polyon [atches corresponding to clusters.
    applyAxCorrection(ax)


def wrapperkeyPressHandler( fig, ax, run1, run2, keyStack=[] ): # the key-stack argument is mutable! I am using this hack to my advantage.
    def _keyPressHandler(event):
        if event.key in ['r', 'R']: # Signal to start entering an r-value
            keyStack.append('r')


        elif event.key in ['0','1','2','3','4','5','6','7','8','9'] and \
             len(keyStack) >= 1                                     and \
             keyStack[0] == 'r': # If the 'r'-signal has already been given, accept only numeric characters

             keyStack.append(event.key)
             print event.key

        elif event.key == 'enter': # Give the signal to interpret the numbers on the stack

           # Incase there are patches present from the previous clustering, just clear them
           clearAxPolygonPatches( ax[0] )
           clearAxPolygonPatches( ax[1] )

           # Time to interpret the numbers in the stack.
           #r = int(''.join( keyStack[1:] ))

           rStr = ''
           for elt in keyStack[1:]:
               if elt in ['0','1','2','3','4','5','6','7','8','9'] :
                   rStr += elt
               else:
                   break # You just hit an non-numeric character entered by mistake

           r = int(rStr)
           print "R-value interpreted: ", r
           keyStack[:] = [] # Empty for further integers.

           # Call the visualization algorithm here!!!!
           run1.r = r
           run1.generateClusters()
           run1.plotClusters(ax[0], pointSize=140, annotatePoints=False)
           applyAxCorrection(ax[0])


           run2.r = r
           run2.generateClusters()
           run2.plotClusters(ax[1], pointSize=140, annotatePoints=False)
           applyAxCorrection(ax[1])



           #pp.pprint(ax.patches)
           #pp.pprint(ax.lines)
           
           fig.canvas.draw()

        elif event.key in ['n','N']: # Stands for get new clustering for the same point cloud but different parameter.

            #print 'Pressed Me!!'
            #sys.stdout.flush()

            run1.clearComputedClusteringsAndR() # We will now compute a new clustering with a new r FOR THE SAME POINT CLOUD.
            run2.clearComputedClusteringsAndR() # We will now compute a new clustering with a new r FOR THE SAME POINT CLOUD           

            clearAxPolygonPatches( ax[0] )
            clearAxPolygonPatches( ax[1] )


            fig.canvas.draw()

    return _keyPressHandler



# Setting up the input events. 
fig, ax =  plt.subplots( 1, 2  )

ax[0].grid( b=True )
ax[1].grid( b=True )

r = 5
pointCloud = givePoints('rbox 20 D2 s W0.3')
run1 = rg.AlgoAggarwalStaticR2L2(r=r, pointCloud=pointCloud) 
run2 = rg.AlgoJieminDecentralizedStatic( r=r, pointCloud = pointCloud ) 


xlim= [-1,1] # Depending on the axes limits, set the radius of the circle. 
ylim= [-1,1]


ax[ 0 ].set_xlim( [xlim[0], xlim[1]] )
ax[ 0 ].set_ylim( [ylim[0], ylim[1]] )
ax[ 0 ].set_aspect( 1.0 )

ax[ 1 ].set_xlim( [xlim[0], xlim[1]] )
ax[ 1 ].set_ylim( [ylim[0], ylim[1]] )
ax[ 1 ].set_aspect( 1.0 )


keyPress     = wrapperkeyPressHandler(fig, ax, run1, run2)
fig.canvas.mpl_connect('key_press_event'    , keyPress   )

# Plot the point-cloud generated with rbox which is an extremely useful tool as you can see
ax[0].plot(pointCloud[:,0],pointCloud[:,1],'bo')
ax[1].plot(pointCloud[:,0],pointCloud[:,1],'bo')
plt.show()
