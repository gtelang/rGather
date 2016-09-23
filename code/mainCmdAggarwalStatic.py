import rGather as rg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import argparse
import pprint

#np.random.seed(0)

print "Started Program"

fig, ax = plt.subplots(1,2)
ax[0].set_aspect( 1.0 )
ax[1].set_aspect( 1.0 )

r           = 3
numCars     =10
lats        = np.random.rand( numCars )
longs       = np.random.rand( numCars )
pointCloud  = zip( lats,longs )


run0 = rg.AlgoAggarwalStaticR2L2( r=r, pointCloud = pointCloud )
clusterCenters = run0.generateClusters()
run0.plotClusters( ax[0], pointSize=40, annotatePoints=True )

# mark the centers.
latsCenters  = [ lats[index] for index in clusterCenters  ]
longsCenters = [ longs[index] for index in clusterCenters  ]
ax[0].plot( latsCenters, longsCenters, 'ro', markersize=10)

run1 = rg.AlgoJieminDecentralizedStatic( r=r, pointCloud = pointCloud )
run1.generateClusters(config = {'mis_algorithm' : 'riksuggestion'} )
run1.plotClusters( ax[1], pointSize=40, annotatePoints=True )





# ax.plot( lats , longs, 'bo' )

# for index in  clusterCenters:
#     circle = plt.Circle(  (lats[index],longs[index]) , 2*R, color=np.random.rand(3,1), fill=False)
#     ax.add_artist(circle)
#     ax.plot( lats[index], longs[index], 'ro'  ) # Plotting cluster centers


# ax.set_xlim([-1,2])
# ax.set_ylim([-1,2])
# ax.set_aspect(1.0)
# plt.show()



plt.show()






