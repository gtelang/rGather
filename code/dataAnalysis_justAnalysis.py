def dist(p,q):
    return np.linalg.norm(  [ p[0] - q[0],
                              p[1] - q[1] ]  )



from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp
import sys
import time
import yaml
import pprint as pp
import rGather as rg
import matplotlib.patches as mpatches

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)




def statisticsComparison_90pmaxVersion ( filepath2Approx,
                                         filepath4Approx_randomMIS,
                                         filepath4Approx_riksuggestion ):
    with open( filepath2Approx, 'r') as stream:
        try:
            D2 = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)

    with open( filepath4Approx_randomMIS, 'r') as stream:
        try:
            D4_randomMIS = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)


    with open( filepath4Approx_riksuggestion, 'r' ) as stream:
        try:
            D4_riksuggestion = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)

    # Same for all the three 
    pointCloud = D4_riksuggestion[ 'coordinates' ] 
    rArray     = sorted( D4_riksuggestion['r'].keys() ) 


    def compute90pMaxClusterDiameter( clusterings  ):

        maxcollection = []

        for cluster in clusterings:
            distancesCluster  = []
            for i in cluster:
                 for j in cluster:
                     if j>i:
                         distancesCluster.append( dist( pointCloud[i], pointCloud[j] ) )
            maxcollection.append(  max( distancesCluster )    )

        return np.percentile( maxcollection, 90 )

    def compute90pMaxRthNearestNeighbors( pointCloud, r ):
            X    = np.array( pointCloud )
            nbrs = NearestNeighbors(n_neighbors=r, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            dist290prnearest = [ distances[i][-1] for i in range(len(distances))  ]
            return np.percentile( dist290prnearest, 90 )



    fig, ax = plt.subplots()

    # 2-Approximation: Blue
    #rMax90DiametersAgg = [ compute90pMaxClusterDiameter( D2['r'][r]  )  for r in rArray  ]

    # 4-ApproximationRANDOM: Red
    rMax90DiametersJie_randomMIS = [ compute90pMaxClusterDiameter( D4_randomMIS['r'][r] )  for r in rArray ]

    # 4-ApproximationRIK: Pink
    rMax90DiametersJie_riksuggestion = [ compute90pMaxClusterDiameter( D4_riksuggestion['r'][r]) for r in rArray  ]

    # Get Max over the max of the r nearest neighbours of the point-cloud.
    rNearestNeighbor90Max = [ compute90pMaxRthNearestNeighbors( pointCloud , r )  for r in rArray  ]


    #ax.plot( rArray, rMax90DiametersAgg               , 'bo-'  )
    ax.plot( rArray, rMax90DiametersJie_randomMIS     , 'ro-'  )
    ax.plot( rArray, rMax90DiametersJie_riksuggestion , 'mo-'  )
    ax.plot( rArray, rNearestNeighbor90Max             ,'go-'  )

    ax.set_xlabel(r'$r$',fontsize=16)
    ax.set_ylabel(r'90th percentile of the Cluster Diameters', fontsize=16)
    ax.set_title(r'Comparing two variants of the Decentralized algorithm')

    plt.grid(b=True)
    #plt.savefig('plots/statisticsComparison_maxVersion.pdf')
    plt.show()







def statisticsComparison_maxVersion ( filepath2Approx, filepath4Approx_randomMIS, filepath4Approx_riksuggestion ):
    with open( filepath2Approx, 'r') as stream:
        try:
            D2 = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)

    with open( filepath4Approx_randomMIS, 'r') as stream:
        try:
            D4_randomMIS = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)


    with open( filepath4Approx_riksuggestion, 'r' ) as stream:
        try:
            D4_riksuggestion = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)

    # Same for all the three
    pointCloud = D4_randomMIS[ 'coordinates' ] 
    rArray     = sorted( D4_randomMIS['r'].keys() ) 

    def computeMaxClusterDiameter( clusterings  ):
        maxDiameterAcrossClusters = 0
        for cluster in clusterings:
             for i in cluster:
                 for j in cluster:
                     if j>i:
                         D = dist( pointCloud[i], pointCloud[j] )
                         if D > maxDiameterAcrossClusters:
                                maxDiameterAcrossClusters = D
        return maxDiameterAcrossClusters

    def computeMaxOverRthNearestNeighbors( pointCloud, r ):
            X    = np.array( pointCloud )
            nbrs = NearestNeighbors(n_neighbors=r, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            dist2rnearest = [ distances[i][-1] for i in range(len(distances))  ]
            return max( dist2rnearest )


    fig, ax = plt.subplots()

    # 2-Approximation: Blue
    #rMaxDiametersAgg = [ computeMaxClusterDiameter( D2['r'][r]  )  for r in rArray  ]

    # 4-ApproximationRANDOM: Red
    rMaxDiametersJie_randomMIS = [ computeMaxClusterDiameter( D4_randomMIS['r'][r] )  for r in rArray ]

    # 4-ApproximationRIK: Pink
    rMaxDiametersJie_riksuggestion = [ computeMaxClusterDiameter( D4_riksuggestion['r'][r]) for r in rArray  ]

    # Get Max over the max of the r nearest neighbours of the point-cloud.
    rNearestNeighborMax = [ computeMaxOverRthNearestNeighbors( pointCloud , r )  for r in rArray  ]


    #ax.plot( rArray, rMaxDiametersAgg               , 'bo-'  )
    ax.plot( rArray, rMaxDiametersJie_randomMIS     , 'ro-'  )
    ax.plot( rArray, rMaxDiametersJie_riksuggestion , 'mo-'  )
    ax.plot( rArray, rNearestNeighborMax             ,'go-'  )

    ax.set_xlabel(r'$r$',fontsize=16)
    ax.set_ylabel(r'Maximum Cluster Diameter', fontsize=16)
    #ax.set_title(r'Comparing the 2-Approx and 4-Approx algorithms')
    fig.suptitle('Comparing two variants of the Decentralized algorithm', fontsize=22)
    # blue_patch = mpatches.Patch(color='blue', label='2-Approximation')
    # plt.legend(handles=[blue_patch])


    # red_patch = mpatches.Patch(color='red', label='RandomMIS')
    # plt.legend(handles=[red_patch])



    # mauve_patch = mpatches.Patch(color='magenta', label='MISbyRthNearestNeighbor')
    # plt.legend(handles=[mauve_patch])


    # green_patch = mpatches.Patch(color='green', label='Max Dist rth Nearest Neighbor')
    # plt.legend(handles=[green_patch])




    plt.grid(b=True)
    #plt.savefig('plots/statisticsComparison_maxVersion.pdf')
    plt.show()



def main( argv ):

    print "Started the program"
    statisticsComparison_90pmaxVersion( './2ApproxStaticOutput/cars60.yaml',
                                     './4ApproxStaticOutput/cars1500_randomMIS.yaml',
                                     './4ApproxStaticOutput/cars1500_riksuggestion.yaml' )


if __name__=="__main__":
    main( sys.argv )

