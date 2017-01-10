import rGather as rg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import argparse
import pprint as pp

# Input configurations.
r            = 5

# numCars     = 1000
# #lats        = np.random.rand(numCars)
# #longs       = np.random.rand(numCars)
# #pointCloud  = zip(lats,longs)

# Setting up the input events. 
fig, ax =  plt.subplots()
ax.grid(b=True)
run1 = rg.Algo_Static_4APX_R2_L2 (r=r, pointCloud=[]) # Empty shell. 

xlim= [0,1] # Depending on the axes limits, set the radius of the circle. 
ylim= [0,1]


ax.set_xlim([xlim[0], xlim[1]])
ax.set_ylim([ylim[0], ylim[1]])
ax.set_aspect(1.0)


def applyAxCorrection(ax):

      ax.grid(b=True)
      ax.set_xlim([xlim[0], xlim[1]])
      ax.set_ylim([ylim[0], ylim[1]])
      ax.set_aspect(1.0)


def clearAxPolygonPatches(ax):
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



def wrapperEnterRunPoints(fig, ax, run):
    """ Create a closure for the mouseClick event.
    """
    def _enterPoints(event):
        if event.name     == 'button_press_event' and \
           event.button   == 1                    and \
           event.dblclick == True                 and \
           event.xdata    != None                 and \
           event.ydata    != None:

            newPoint = (event.xdata, event.ydata)
            run.pointCloud.append( newPoint  )

            patchSize  = (xlim[1]-xlim[0])/140.0

            ax.add_patch( mpl.patches.Circle( newPoint,
                                              radius = patchSize,
                                              facecolor='blue',
                                              edgecolor='black'   )  )

            ax.set_title('Points Inserted: ' + str(len(run.pointCloud)), fontdict={'fontsize':40})
            #ax.set_title('Decentralized Algorithm on 60 points', fontdict={'fontsize':30})

            # It is inefficient to clear the polygon patches inside the enterpoints loop as done here.
            # I have just done this for simplicity: the intended behaviour at any rate, is
            # to clear all the polygon patches from the axes object, once the user starts entering in MORE POINTS TO THE CLOUD
            # for which the clustering was just computed and rendered. The moment the user starts entering new points,
            # the previous polygon patches are garbage collected. 
            clearAxPolygonPatches(ax)

            applyAxCorrection(ax)
            fig.canvas.draw()


    return _enterPoints


def wrapperkeyPressHandler(fig,ax, run, keyStack=[]): # the key-stack argument is mutable! I am using this hack to my advantage.
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
           clearAxPolygonPatches(ax)

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
           run.r = r
           run.generateClustersSimple( config={'mis_algorithm':'riksuggestion'}  )
           run.plotClusters(ax, pointSize=140, annotatePoints=False)

           applyAxCorrection(ax)

           #pp.pprint(ax.patches)
           #pp.pprint(ax.lines)
           
           fig.canvas.draw()

        elif event.key in ['c', 'C']: # Clear the screen and the states of all the objects

            run.clearAllStates()
            ax.cla()
            ax.grid(b=True)

            applyAxCorrection(ax)


            fig.canvas.draw()


        elif event.key in ['n','N']: # Stands for get new clustering for the same point cloud but different parameter.

            #print 'Pressed Me!!'
            #sys.stdout.flush()

            run.clearComputedClusteringsAndR() # We will now compute a new clustering with a new r FOR THE SAME POINT CLOUD.

            clearAxPolygonPatches(ax)

            fig.canvas.draw()

    return _keyPressHandler

mouseClick   = wrapperEnterRunPoints (fig,ax, run1)
keyPress     = wrapperkeyPressHandler(fig,ax, run1)

fig.canvas.mpl_connect('button_press_event' , mouseClick )
fig.canvas.mpl_connect('key_press_event'    , keyPress   )
plt.show()
