import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp
import sys
import time
import yaml
import pprint as pp
import rGather as rg


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def writeClusteringResultsFiles(  configurations  ):
    """ Run the algorithms on the input files and write out the corresponding
        clusters for different values of r for each algorithm.
        THE ANALYSIS AND PLOTTING IS LEFT FOR A SEPARATE FUNCTION. """

    def readPointCloudFile( filepath ):
        """ Read Point Cloud file. Containing X-Y GPS coordinates """
        xs, ys = [ ], [ ]

        with open( filepath ) as f: # Read file line by line and extract 2nd and 3rd coordinates.
            for line in f:
                lineString =  line.split()
                xs.append( float( lineString[ 1 ] ) )
                ys.append( float( lineString[ 2 ] ) )
        return zip( xs, ys )




    # For each input file, located at inputpath, run the algorithms and record
    # results in outputpath
    for inputpath, outputpath, rArray, comment in configurations:
        # Initialize dictionary, to be written out to a YAML output file
        D = {}
        D[ 'filename' ]    = inputpath
        D[ 'comment' ]     = comment

        points = readPointCloudFile( inputpath )
        D[ 'coordinates' ] = points

        # Clusterings computed for different values of r stored in D['r'] which is itself a dictionary
        # i.e. for r=2, D['r'][2] means the clustering obtained for r=2
        D[ 'r' ]           = { } # Populated in the loop below.

        # Generate clusters for the specified r's
        for r in rArray: # For N values of r starting at 2, compute clusters

            run = rg.AlgoJieminDecentralizedStatic( r = r, pointCloud = points )
            run.generateClusters( config = {'mis_algorithm': 'networkx_random_choose_20_iter_best' } )
            #run = rg.AlgoAggarwalStaticR2L2( r=r, pointCloud = points )
            #run.generateClusters()

            clusters = run.computedClusterings
            D[ 'r' ][ r ] = clusters


        # Write dictionary to file now that all the constituents were computed.
        with open( outputpath, 'w' ) as outfile:
            outfile.write( yaml.dump( D ) )






def compareAlgorithms( filepath2Approx, filepath4Approx ):


    def visualComparison(D2, D4):

        pointCloud = D2[ 'coordinates' ] # All (x,y) 's of the point cloud. The points are exactly the same for both
        rArray     = sorted( D2['r'].keys() ) # Sorting because I don't know what order the keys are returned in. Being safe here. Again, both must be the same for either algorithms.

        # Save a different figure for each r.  
        for r in rArray:

            fig, ax = plt.subplots(1,2)
            ax[0].set_aspect(1.0)
            ax[1].set_aspect(1.0)


            runAgg = rg.AlgoAggarwalStaticR2L2(r = r, pointCloud = pointCloud)
            runAgg.computedClusterings = D2['r'][r]
            runAgg.plotClusters(ax[0], pointSize=40, annotatePoints=False)


            runJie = rg.AlgoJieminDecentralizedStatic( r = r, pointCloud = pointCloud )
            runJie.computedClusterings = D4['r'][ r ]
            runJie.plotClusters(ax[1], pointSize=40, annotatePoints=False)

            plt.savefig('plots/visualComparison-r' + str(r) + '.pdf')



    def statisticsComparison_maxVersion( D2, D4  ):


        pointCloud = D2[ 'coordinates' ] # All (x,y) 's of the point cloud. The points are exactly the same for both
        rArray     = sorted( D2['r'].keys() ) # Sorting because I don't know what order the keys are returned in. Being safe here. Again, both must be the same for either algorithms.


        def computeMaxClusterDiameter( clusterings  ):

            def dist(p,q):
                return np.linalg.norm(  [ p[0] - q[0],
                                          p[1] - q[1] ]  )

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

            def dist(p,q):
                return np.linalg.norm(  [ p[0] - q[0],
                                          p[1] - q[1] ]  )


            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            import sys

            X    = np.array( pointCloud )
            nbrs = NearestNeighbors(n_neighbors=r, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

            dist2rnearest = [ distances[i][-1] for i in range(len(distances))  ]

            return max( dist2rnearest )



        fig, ax = plt.subplots()

        # 2-Approximation: Blue
        rMaxDiametersAgg = [ computeMaxClusterDiameter( D2['r'][r]  )  for r in rArray  ]

        # 4-Approximation: Red
        rMaxDiametersJie = [ computeMaxClusterDiameter( D4['r'][r] )  for r in rArray ]

        # Get Max over the max of the r nearest neighbours of the point-cloud.
        rNearestNeighborMax = [ computeMaxOverRthNearestNeighbors( pointCloud , r )  for r in rArray  ]


        ax.plot( rArray, rMaxDiametersAgg   , 'bo-'  )
        ax.plot( rArray, rMaxDiametersJie   , 'ro-'  )
        ax.plot( rArray, rNearestNeighborMax , 'go-'  )

        ax.set_xlabel(r'$r$',fontsize=16)
        ax.set_ylabel(r'Maximum Cluster Diameter', fontsize=16)
        ax.set_title(r'Comparing the 2-Approx and 4-Approx algorithms')

        plt.grid(b=True)
        plt.savefig('plots/statisticsComparison_maxVersion.pdf')


        plt.show()



    def statisticsComparison_90thpercVersion(D2, D4):

        pointCloud = D2[ 'coordinates' ] # All (x,y) 's of the point cloud. The points are exactly the same for both
        rArray     = sorted( D2['r'].keys() ) # Sorting because I don't know what order the keys are returned in. Being safe here. Again, both must be the same for either algorithms.



        def computeMax90pClusterDiameter( clusterings  ):

            def dist(p,q):
                return np.linalg.norm(  [ p[0] - q[0],
                                          p[1] - q[1] ]  )

            maxDiameterAcrossClusters = 0
            for cluster in clusterings:
                for i in cluster:
                    for j in cluster:
                        if j>i:
                            D = dist( pointCloud[i], pointCloud[j] )
                            if D > maxDiameterAcrossClusters:
                                maxDiameterAcrossClusters = D

            return maxDiameterAcrossClusters


        def computeMaxOver90pRthNearestNeighbors( pointCloud, r ):

            def dist(p,q):
                return np.linalg.norm(  [ p[0] - q[0],
                                          p[1] - q[1] ]  )


            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            import sys

            X    = np.array( pointCloud )
            nbrs = NearestNeighbors(n_neighbors=r, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

            dist2rnearest = [ distances[i][-1] for i in range(len(distances))  ]

            return max( dist2rnearest )



        fig, ax = plt.subplots()
        # All meat goes here
        # 2-Approximation: Blue
        rMaxDiametersAgg = [ computeMaxClusterDiameter( D2['r'][r]  )  for r in rArray  ]

        # 4-Approximation: Red
        rMaxDiametersJie = [ computeMaxClusterDiameter( D4['r'][r] )  for r in rArray ]

        # Get Max over the max of the r nearest neighbours of the point-cloud.
        rNearestNeighborMax = [ computeMaxOverRthNearestNeighbors( pointCloud , r )  for r in rArray  ]


        ax.plot( rArray, rMaxDiametersAgg   , 'bo-'  )
        ax.plot( rArray, rMaxDiametersJie   , 'ro-'  )
        ax.plot( rArray, rNearestNeighborMax , 'go-'  )


        ax.set_title('')



        plt.savefig('statisticsComparison_90thpercVersion.pdf')
        plt.show()


    def statisticsLargeDataSet( D4  ):

        pointCloud = D4[ 'coordinates' ] # All (x,y) 's of the point cloud. The points are exactly the same for both
        rArray     = sorted( D4['r'].keys() ) # Sorting because I don't know what order the keys are returned in. Being safe here. Again, both must be the same for either algorithms.


        def computeMaxClusterDiameter( clusterings  ):

            def dist(p,q):
                return np.linalg.norm(  [ p[0] - q[0],
                                          p[1] - q[1] ]  )

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

            def dist(p,q):
                return np.linalg.norm(  [ p[0] - q[0],
                                          p[1] - q[1] ]  )


            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            import sys

            X    = np.array( pointCloud )
            nbrs = NearestNeighbors(n_neighbors=r, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

            dist2rnearest = [ distances[i][-1] for i in range(len(distances))  ]

            return max( dist2rnearest )


        fig, ax = plt.subplots()

        # 4-Approximation: Red
        rMaxDiametersJie = [ computeMaxClusterDiameter( D4['r'][r] )  for r in rArray ]

        # Get Max over the max of the r nearest neighbours of the point-cloud.
        rNearestNeighborMax = [ computeMaxOverRthNearestNeighbors( pointCloud , r )  for r in rArray  ]


        ax.plot( rArray, rMaxDiametersJie   , 'ro-'  )
        ax.plot( rArray, rNearestNeighborMax , 'go-'  )

        ax.set_xlabel(r'$r$',fontsize=16)
        ax.set_ylabel(r'Maximum Cluster Diameter', fontsize=16)

        ax.set_title(r'4-Approx algorithm on 1500 cars')
        plt.grid(b=True)


        plt.savefig('plots/statisticsBigDataSet.pdf')
        plt.show()





    with open( filepath2Approx, 'r') as stream:
        try:
            D2 = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)

    with open( filepath4Approx, 'r') as stream:
        try:
            D4 = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)

    #visualComparison(D2 , D4)
    #statisticsComparison_maxVersion( D2, D4 )
    #statisticsComparison_90thpercVersion( D2, D4 )
    statisticsLargeDataSet( D4 )


def main( argv ):

    print "Started the program"

    if len(argv) == 1:
        print "Use exactly one of the options: \'--writeClusters\'  , \'--analyseResults\' "

    elif argv[1] == '--writeClusters':
        configurations= [ ( 'StaticInput/shenzhenCars60.txt' ,'4ApproxStaticOutput/cars60_randomMIS.yaml', [2,4,6,8,10,12,14,16] , ''),
                          ( 'StaticInput/shenzhenCars1500.txt' ,'4ApproxStaticOutput/cars1500_randomMIS.yaml', [10,20,30,40,50,60,70,80,90,100,110] , '')      ]
        writeClusteringResultsFiles( configurations )

    elif argv[1] == '--compareAlgorithms':
        compareAlgorithms( './2ApproxStaticOutput/cars60.yaml', './4ApproxStaticOutput/cars1500.yaml' )

    else:
        print "Ouch! Argument not recognized! :-c "

if __name__=="__main__":
    main( sys.argv )
