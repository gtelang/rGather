import rGather as rg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import argparse

r           = 2
numCars     = 10
lats        = np.random.rand(numCars)
longs       = np.random.rand(numCars)
pointCloud  = zip(lats,longs)


# Setting up the input events.
fig, ax =  plt.subplots(1,2)
ax[0].set_aspect(1.0)



run = rg.Algo_Static_4APX_R2_L2( r=r, pointCloud = pointCloud )

run.generateClustersSimple( config = {'mis_algorithm': 'riksuggestion' })

run.plotClusters(ax[0],pointSize=40, annotatePoints=False)
run.plotStatistics( { ax[1]:'clusterPopulationSizes' })



plt.show()
