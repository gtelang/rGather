
#!/usr/bin/python
import matplotlib as mpl, numpy as np, scipy as sp, sys, math, colorsys 
from matplotlib import pyplot as plt, animation 
import networkx as nx, sklearn as sk
from abc import ABCMeta, abstractmethod
from haversine import haversine # https://pypi.python.org/pypi/haversine
from termcolor import colored



class AlgoAggarwalStaticR2L2( AlgoAggarwalStatic ):
    
   def __init__(self, r, pointCloud):

      self.r                    = r     
      self.pointCloud           = pointCloud 
      self.computedClusterings  = []  
      self.algoName             = 'Metric Space Static r-Gather applied to R2L2'

      #super(  AlgoAggarwalStaticR2L2, self ).__init__( self.r, self.pointCloud  )
 
   def clearAllStates(self):
         self.r                   = None
         self.pointCloud          = [] 
         self.computedClusterings = []
         
   def clearComputedClusteringsAndR(self):
            self.r                   = None
            self.computedClusterings = []

   def dist(self, p,q):
      """ Euclidean distance between points p and q in R^2 """
      return np.linalg.norm( [ p[0]-q[0] , 
                               p[1]-q[1] ]  )


   def findNearestNeighbours(self,pointCloud, k):
      """  pointCloud : 2-d numpy array. Each row is a point
      k          : The length of the neighbour list to compute. 
      """
      from sklearn.neighbors import NearestNeighbors
      import numpy as np
      import sys
      
      X    = np.array(pointCloud)
      nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
      distances, indices = nbrs.kneighbors(X)

      return distances, indices


   def rangeSearch(self, pointCloud, radius):
      """ A wrapper for a good neighbour search routine provided by Scipy.
          Given a point-cloud, return the neighbours within a distance of 'radius'
          for every element of the pointcloud. return the neighbour indices , sorted 
          according to distance. """
      import numpy as np
      import sys
      from scipy import spatial
      
      X        = np.array( pointCloud )
      mykdtree = spatial.KDTree( X )
      nbrlists = list( mykdtree.query_ball_point( X, radius) )
     

      distances = []
      for index  in  range(len(nbrlists)):

         def fn_index( i ): # Distance function local to this iteration of the loop
            return np.linalg.norm(  [  X[i][0] - X[index][0]   , 
                                       X[i][1] - X[index][1]    ]    )

         # Replace the unsorted array with the sorted one. 
         nbrlists[index]  = sorted( nbrlists[index], key = fn_index  ) 

         # Get corresponding distances, which will now naturally be in sorted order. 
         distances.append( map( fn_index, nbrlists[ index ] ) ) 
 

      indices = nbrlists # Just a hack, too lazy to change nbrlists to the name indices above. 

      return distances, indices 


   
   
   def plotClusters(self,  ax    , 
                  pointSize=200, 
                  marker='o'   , 
                  pointCloudInfo='',
                  annotatePoints=True):
       
   
         from scipy import spatial
         import numpy as np, matplotlib as mpl
         import matplotlib.pyplot as plt
    
         # Plot point-cloud 
         xs = [x for (x,y) in self.pointCloud]
         ys = [y for (x,y) in self.pointCloud]
         ax.plot(xs,ys,'bo', markersize=3) 
         ax.set_aspect(1.0)    
   
         if annotatePoints==True:
               # Annotate each point with a corresponding number. 
               numPoints = len(xs)
               labels = ['{0}'.format(i) for i in range(numPoints)]
               
               for label, x, y in zip(labels, xs, ys):
                     ax.annotate(  label                       , 
                                   xy         = (x, y)         , 
                                   xytext     = (-3, 0)      ,
                                   textcoords = 'offset points', 
                                   ha         = 'right'        , 
                                   va         = 'bottom')
                     
   
         # Overlay with cluster-groups.
         for s in self.computedClusterings:
         
           clusterColor = getRandomColor()
           xc = [ xs[i]  for i in s   ]
           yc = [ ys[i]  for i in s   ]
   
           # Mark all members of a cluster with a nice fat dot around it. 
           #ax.scatter(xc, yc, c=clusterColor, 
           #           marker=marker, 
           #           s=pointSize) 
   
           #ax.plot(xc,yc, alpha=0.5, markersize=1 , markerfacecolor=clusterColor , linewidth=0)
           #ax.set_aspect(1.0)
   
           # For some stupid reason sp.spatial.ConvexHull requires at least three points for computing the convex hull. 
           
           if len(xc) >= 3 : 
                 hull = spatial.ConvexHull(  np.array(zip(xc,yc)) , qhull_options="QJn" ) # Last option because of this http://stackoverflow.com/q/30132124/505306
                 hullPoints = np.array( zip( [ xc[i] for i in hull.vertices ],  
                                             [ yc[i] for i in hull.vertices ] ) )
                 ax.add_patch( mpl.patches.Polygon(hullPoints, alpha=0.5, 
                                                   facecolor=clusterColor) )
          
   
           elif len(xc) == 2:
                  ax.plot( xc,yc, color=clusterColor )
               
   
           ax.set_aspect(1.0)
           ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':15})
           ax.set_xlabel('Latitude', fontdict={'fontsize':10})
           ax.set_ylabel('Longitude',fontdict={'fontsize':10})
   
           #ax.get_xaxis().set_ticks( [] ,  fontdict={'fontsize':10})
           #ax.get_yaxis().set_ticks( [],  fontdict={'fontsize':10} ) 
   
           ax.grid(b=True)
   
   
   
   
   def plotStatistics(self, axStatsDict ):
      """ axStatsDict, specifies the mapping of axes objects to the statistic
          being plotted.""" 
   
      def plotConvexHullDiameters(ax):
         pass
     
      def plotMinBoundingCircleDiameters(ax):
         pass
   
      def plotClusterPopulationSizes(ax):
         barHeights = map(len, self.computedClusterings )
         numBars    = len(barHeights)
   
         ax.bar( range(numBars) ,barHeights, width=1.0, align='center')
         ax.set_title('Number of points per Cluster', fontdict={'fontsize':30})
   
         ax.set_aspect(1.0)
         ax.grid(b=True)
   
      for ax, statistic in axStatsDict.iteritems():
          
           if statistic == 'convexHullDiameters': 
              plotConvexHullDiameters(ax) 
           
           elif statistic == 'minBoundingCircleDiameters':
              plotMinBoundingCircleDiameters(ax)
   
           elif statistic == 'clusterPopulationSizes':
              plotClusterPopulationSizes(ax)
   
           else:
              pass
   
    

class AlgoJieminDecentralizedStatic:
    
    def __init__(self, r, pointCloud):
      """  r          : Cluster parameter
           pointCloud : An n x 2 numpy array where n is the number 
                        of points in the cloud and each row contains 
                        the (x,y) coordinates of a point."""
      
      self.r                    = r     
      self.pointCloud           = pointCloud  
      self.computedClusterings  = []  
      self.algoName             = 'Decentralized Static r-Gather'
    
    
    
    def clearAllStates(self):
      self.r                   = None
      self.pointCloud          = [] 
      self.computedClusterings = []
    
    
    def clearComputedClusteringsAndR(self):
      self.r                   = None
      self.computedClusterings = []
    
    
    
    
    
    def generateClusters(self, config={'mis_algorithm': 'networkx_random_choose_20_iter_best'}):
      """ config : Configuration parameters which might be needed 
                   for the run. 
      Options recognized are (ALL LOWER-CASE)
      1. mis_algorithm:
           A. 'networkx_random_choose_20_iter_best', default 
           B. 'riksuggestion'
      """
    
    
      import itertools
      import numpy as np
      import pprint as pp
      import copy
      
      def findNearestNeighbours(pointCloud, k):
        """  pointCloud : 2-d numpy array. Each row is a point
             k          : The length of the neighbour list to compute. 
        """
        from sklearn.neighbors import NearestNeighbors
        import sklearn
        import numpy as np
        import sys
      
      
      
        X    = np.array(pointCloud)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit( X )
        distances, indices = nbrs.kneighbors(X)
      
        return distances, indices
      
      def findMaximalIndependentOfNeighbourhoods(  nbds , mis_algorithm  ):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(len(nbds)))
      
        # If two neighbourhoods intersect, draw 
        # a corresponding edge in the graph. 
        for i in range(len(nbds)):
          for j in range(i+1,len(nbds)):
            intersection_of_nbds_ij = [  val  for val in nbds[i] if val in nbds[j]    ] 
            if len(intersection_of_nbds_ij) >= 1:
              G.add_edge(i,j)
      
        # Having constructed the neighbourhood, we proceed to find a good MIS
        # The quality of the solution is affected by the size of the MIS
        # The larger the maximal independent set, the better it is
        if mis_algorithm == 'networkx_random_choose_20_iter_best': 
          candidateSindices = [ nx.maximal_independent_set(G) for i in range(20)  ]
      
          #for candidate in candidateSindices: # for debugging
          #  print candidate
      
          sIndices = [] # Start value for finding the maximum
          for candidate in candidateSindices: # Pick the largest independent set over 10 iterations
            if len(candidate) > len(sIndices): # Yay! Found a larger independent set!
              print "Larger set!"
              sIndices = candidate
      
      
        elif mis_algorithm == 'riksuggestion':
          
          # Give cluster centers a special attribute marking it as a center. 
          distanceFromRthNearestNeighbourDict = {}
       
          for nbd, i in zip( nbds, range( len(nbds) )): # Note that each neighbourhood's 0th element is the center, and that the nbd indices are sorted by distance from this zeroth element. So -1 makes sense
              nbdCenterCoords                      = self.pointCloud[ nbd[0] ] 
              nbdFarthestNeighbourCoords           = self.pointCloud[ nbd[-1] ]
              distanceFromRthNearestNeighbourDict[i] = np.linalg.norm( [ nbdCenterCoords[0] - nbdFarthestNeighbourCoords[0] ,
                                                                         nbdCenterCoords[1] - nbdFarthestNeighbourCoords[1] ]  )# Abstract this away with the distance function later. 
      
          nx.set_node_attributes( G, 'distanceFromRthNearestNeighbour', distanceFromRthNearestNeighbourDict )
      
          import collections
          # Generate the order to remove the vertices
          orderOfVerticesToDelete = collections.deque(sorted(  range(len(nbds)) , key = lambda x: G.node[x][ 'distanceFromRthNearestNeighbour' ]    ))
          
          #print orderOfVerticesToDelete
          #for i in orderOfVerticesToDelete:
          #  print G.node[i]['distanceFromRthNearestNeighbour']
          sIndices = [ ]
      
      
          for i in orderOfVerticesToDelete:
      
            try:
               node = orderOfVerticesToDelete[i]
      
               nlist = G.neighbors( node )
      
               for n in nlist:
                 try:
                   G.remove_edge( node, n ) # Remove all edges emanating
                 except nx.NetworkXError:
                   continue
      
               G.remove_node( node ) # Remove the node itself
      
                
               for n in nlist:
                 try:
                   G.remove_node( n ) # Remove all the neighbours.
                 except nx.NetworkXError:
                   continue
      
               sIndices.append( node ) 
      
            except nx.NetworkXError:
                continue
      
      
          # while( len( orderOfVerticesToDelete ) >= 1 ): # This list changes during the iteration. 
      
          #     try:
          #       node  = orderOfVerticesToDelete[0]
      
          #     except nx.NetworkXError:
          #         print "Removing carcass"
          #         orderOfVerticesToDelete.popleft()
      
          #     else:
          #       sIndices.append( node ) # The very fact no exception was thrown means that you can freely add it to the independent set
          #       nlist = G.neighbors( node )
      
          #       # Delete all the edges emanating from  elements of nlist. 
          #       # The fact that this did not throw an exception means 'node' still exists in the graph G
          #       for n in nlist:
          #          G.remove_edge( node, n ) # Remove all edges emanating
      
          #       G.remove_node( node ) # Remove the node itself
      
          #       for n in nlist:
          #         G.remove_node( n ) # Remove all the neighbours.
                  
          #       orderOfVerticesToDelete.popleft()
      
        else:
          import sys
          print "Maximum independent Set Algorithm option not recognized!"
          sys.exit()
      
      
        # If two neighbourhoods intersect, draw 
        # a corresponding edge in the graph. 
        # print sIndices
        for i in sIndices:
           for j in sIndices:
             if j > i:
               intersection_of_nbds_ij = [val for val in nbds[i] if val in nbds[j] ]
               if len(intersection_of_nbds_ij) >= 1:
                     print "Neighbourhoods intersect!"
                     sys.exit()
      
        # print "Exiting!"
        # import sys
        # sys.exit()
      
        return [ nbds[s] for s in sIndices ]
      
      def extractUniqueElementsFromList( L ):
          
          uniqueElements = []
          for elt in L:
              if elt not in uniqueElements: # Just discovered a brand new element!!
                  uniqueElements.append(elt)
      
          return uniqueElements
      
    
    
      NrDistances, Nr = findNearestNeighbours( self.pointCloud, 
                                               self.r )
      S               = findMaximalIndependentOfNeighbourhoods( Nr.tolist( ), 
                                                                config[ 'mis_algorithm' ] )
    
      indicesOfPointsCoveredByS = set(list(itertools.chain.from_iterable(S)))
      indicesOfPointsMissedByS  = set(range(len(self.pointCloud))).difference(indicesOfPointsCoveredByS)
    
      assert(indicesOfPointsCoveredByS.union(indicesOfPointsMissedByS ) == set(range(len(self.pointCloud))) )
    
      # For each point missed by S, find which elements of its r-neighbourhood lies inside a member of S. 
      pNrS = {} # A dictionary which maintains this information.  
      for index in indicesOfPointsMissedByS:
    
         pNrS[index] = [] 
    
         #Coordinates of the point whose index is 'index'
         ptIndex     = np.array( self.pointCloud[index] )
       
         neighborIndices = Nr[index][1:] 
    
         for nbIndex in neighborIndices:
           for s in S:
             if nbIndex in s:
      
               ptnbIndex = np.array(self.pointCloud[nbIndex])
    
               dist = np.linalg.norm( ptIndex - ptnbIndex  ) # Euclidean distance between the points
               pNrS[index].append(  (s, dist)    )
               break # since members of S are disjoint there is no reason to continue to iterate over members of S to check containment of nbindex
                     # Move onto the next member of neighbourIndices. 
    
      # print "\nNr   = "     , Nr
      # print "\nS    = "     , S
      # print "\npointsMissed", indicesOfPointsMissedByS
      # print "\npNrS = "     ; pp.pprint(pNrS, width=20 )
    
    
      # Now for each point select the member of S that is closest using this dictionary. 
      # Edit this dictionary in place, by keeping only the closest neighbourhood. 
      pNrS_trimmed = {}
      for (key, value) in pNrS.iteritems():
          distmin = float("inf") # Positive infinity
    
          for (s, dist) in value:
            if dist<distmin:
                smin    = s
                distmin = dist
                 
    
          #pNrS_trimmed[key] = (smin,distmin) # For debugging purposes. 
          pNrS_trimmed[key] = smin
    
      #print "\npNrS_trimmed = "; pp.pprint(pNrS_trimmed, width=1) 
    
    
    
      # With pNrS_trimmed we obtain the final clustering. Yay!
      # by "inverting" this key-value mapping
      augmentedSets = [s for s in S if s not in pNrS_trimmed.values()] # The sets just included are not augmented at all. 
      
      pNrS_codomain = extractUniqueElementsFromList(pNrS_trimmed.values())
     
      for s in pNrS_codomain:
        smodified = copy.copy(s) # This copying step is SUPER-CRUCIAL!!! if you just use =, you will just be binding object pointed to by s to smod. Modifying smod, will then modify s, which will trip up your future iterations! I initially implemented it like this and got tripped up 
        for key, value in pNrS_trimmed.iteritems():
          if s == value:
            smodified.append(key) # augmentation step
    
        augmentedSets.append(smodified)
    
    
      self.computedClusterings = augmentedSets
      
      #print "\nself.computedClusterings = "; pp.pprint(self.computedClusterings,width=1)
      print   "Numpoints = "                   , len( self.pointCloud )       ,  \
              " r = "                          , self.r                       ,  \
              " Number of Clusters Computed = ", len( self.computedClusterings ), \
              " Algorithm used: "              , self.algoName
      sys.stdout.flush() 
    
    
    def plotClusters(self,  ax    , 
                   pointSize=200, 
                   marker='o'   , 
                   pointCloudInfo='',
                   annotatePoints=True):
        
    
          from scipy import spatial
          import numpy as np, matplotlib as mpl
          import matplotlib.pyplot as plt
     
          # Plot point-cloud 
          xs = [x for (x,y) in self.pointCloud]
          ys = [y for (x,y) in self.pointCloud]
          ax.plot(xs,ys,'bo', markersize=3) 
          ax.set_aspect(1.0)    
    
          if annotatePoints==True:
                # Annotate each point with a corresponding number. 
                numPoints = len(xs)
                labels = ['{0}'.format(i) for i in range(numPoints)]
                
                for label, x, y in zip(labels, xs, ys):
                      ax.annotate(  label                       , 
                                    xy         = (x, y)         , 
                                    xytext     = (-3, 0)      ,
                                    textcoords = 'offset points', 
                                    ha         = 'right'        , 
                                    va         = 'bottom')
                      
    
          # Overlay with cluster-groups.
          for s in self.computedClusterings:
          
            clusterColor = getRandomColor()
            xc = [ xs[i]  for i in s   ]
            yc = [ ys[i]  for i in s   ]
    
            # Mark all members of a cluster with a nice fat dot around it. 
            #ax.scatter(xc, yc, c=clusterColor, 
            #           marker=marker, 
            #           s=pointSize) 
    
            #ax.plot(xc,yc, alpha=0.5, markersize=1 , markerfacecolor=clusterColor , linewidth=0)
            #ax.set_aspect(1.0)
    
            # For some stupid reason sp.spatial.ConvexHull requires at least three points for computing the convex hull. 
            
            if len(xc) >= 3 : 
                  hull = spatial.ConvexHull(  np.array(zip(xc,yc)) , qhull_options="QJn" ) # Last option because of this http://stackoverflow.com/q/30132124/505306
                  hullPoints = np.array( zip( [ xc[i] for i in hull.vertices ],  
                                              [ yc[i] for i in hull.vertices ] ) )
                  ax.add_patch( mpl.patches.Polygon(hullPoints, alpha=0.5, 
                                                    facecolor=clusterColor) )
           
    
            elif len(xc) == 2:
                   ax.plot( xc,yc, color=clusterColor )
                
    
            ax.set_aspect(1.0)
            ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':15})
            ax.set_xlabel('Latitude', fontdict={'fontsize':10})
            ax.set_ylabel('Longitude',fontdict={'fontsize':10})
    
            #ax.get_xaxis().set_ticks( [] ,  fontdict={'fontsize':10})
            #ax.get_yaxis().set_ticks( [],  fontdict={'fontsize':10} ) 
    
            ax.grid(b=True)
    
      
    
    
    def plotStatistics(self, axStatsDict ):
       """ axStatsDict, specifies the mapping of axes objects to the statistic
           being plotted.""" 
    
       def plotConvexHullDiameters(ax):
          pass
      
       def plotMinBoundingCircleDiameters(ax):
          pass
    
       def plotClusterPopulationSizes(ax):
          barHeights = map(len, self.computedClusterings )
          numBars    = len(barHeights)
    
          ax.bar( range(numBars) ,barHeights, width=1.0, align='center')
          ax.set_title('Number of points per Cluster', fontdict={'fontsize':30})
    
          ax.set_aspect(1.0)
          ax.grid(b=True)
    
       for ax, statistic in axStatsDict.iteritems():
           
            if statistic == 'convexHullDiameters': 
               plotConvexHullDiameters(ax) 
            
            elif statistic == 'minBoundingCircleDiameters':
               plotMinBoundingCircleDiameters(ax)
    
            elif statistic == 'clusterPopulationSizes':
               plotClusterPopulationSizes(ax)
    
            else:
               pass
    
    

class AlgoJieminDynamic( AlgoAggarwalStatic ):
     
    def __init__(self, r,  pointCloud,  memoizeNbrSearch = False, distances_and_indices_file=''):
       """ Initialize the AlgoJieminDynamic
 
           memoizeNbrSearch = this computes the table in the constructor itself. no need for a file. The file option below, is only useful for large runs.
           distances_and_indices_file = must be a string identifer for the file-name on disk. 
                                        containing the pairwise-distances and corresponding index numbers
                                        between points. I had to appeal to this hack, since sklearn's algorithm to search in arbitrary metric spaces does not work for my case. 
                                        Also the brute-force computation, which I initially implemented took far too long. 
                                        Since  don't know how to do the neighbor computation for arbitrary metric spaces, I just precompute 
                                        everything into a table, stored in a YAML file.
       """

       from termcolor import colored
       import yaml

       # len(trajectories) = number of cars
       # len(trajectories[i]) = number of GPS samples taken for the ith car. For shenzhen data set this is
       # constant for all cars.

       self.r                    = r     
       self.pointCloud           = pointCloud # Should be of type  [ [(Double,Double)] ] 
       self.computedClusterings  = []  
       self.algoName             = 'r-Gather for trajectory clustering'
       self.superSlowBruteForce  = False

       if memoizeNbrSearch :
             numpts     = len(self.pointCloud)
             (self.nbrTable_dist, self.nbrTable_idx) = ([], [])

             for i in range(numpts):

                     print colored ('Calculating distance from '+str(i), 'white', 'on_magenta',['underline','bold']) 
                     traj_i = pointCloud[i]
                     distances_and_indices = []

                     for j in range(numpts):
              
                          traj_j = pointCloud[j]
                          dij = self.dist( traj_i, traj_j)
                          distances_and_indices.append((dij,j))
                          print '......to j= '  , j, '  dij= ', dij
                   
                     # Now sort the distances of all points from point i. 
                     distances_and_indices.sort(key=lambda tup: tup[0]) # http://tinyurl.com/mf8yz5b
                     self.nbrTable_dist.append( [ d   for (d,idx) in distances_and_indices ]  )
                     self.nbrTable_idx.append ( [ idx for (d,idx) in distances_and_indices ]  )

       elif distances_and_indices_file != '': # Non empty file name passed

             print colored("Started reading neighbor file", 'white','on_magenta',['bold','underline'])              
             stream       = open(distances_and_indices_file,'r')
             filecontents = yaml.load(stream) # This will be a dictionary
             print colored("Finished reading neighbor file", 'white','on_green',['bold','underline'])              

             self.nbrTable_dist = filecontents['Distances']
             self.nbrTable_idx  = filecontents['Indices']

       else:
             self.superSlowBruteForce = True


    def clearAllStates(self):
          self.r                   = None
          self.pointCloud          = [] 
          self.computedClusterings = []
          
    def clearComputedClusteringsAndR(self):
             self.r                   = None
             self.computedClusterings = []

    def dist(self, p,q):
       """ distance between two trajectories p and q. The trajectories form a metric space under this distance 
       If you visualize the given table as a microsoft excel sheet, where each column represents the trajectory 
       of a car, then the distance between two trajectories is the max of L infinity norm of the difference of two 
       columns. 

       p,q :: [ [Double,Double] ]. The length of p or q, indicates the number of GPS samples taken
       
       """
       #print "Inside distance function"
       #print "p is ", p.shape, ' ' , p
       #print "q is ", q.shape, ' ' , q

       dpq = 0
       for t in range(len(p)):
            # M is the euclidean distance between two points at time t.  
            M = np.sqrt( abs( (p[t][0]-q[t][0])**2 + (p[t][1]-q[t][1])**2 ) ) 
            if M > dpq:
                dpq = M
       
       #print p, q, dpq, ' ' , np.sqrt( (p[0][0]-q[0][0])**2 + (p[0][1]-q[0][1])**2)
       #from termcolor import colored 
       #print colored( str(dpq) , 'white', 'on_red', ['bold'] ) # This to make sure that dpq being returned is a sane number.
       return dpq


    def findNearestNeighbours(self, pointCloud, k):
       """Return the k-nearest nearest neighbours"""
       import numpy as np, itertools as it
       from termcolor import colored 
       numpts = len(pointCloud)

       # Calling sklearn works only on R2L2 case for some reason. So for the moment, the only option is to use brute-force techniques.
       if self.superSlowBruteForce : 
                  print colored('Calling Super-slow brute Force kNN' , 'white', 'on_magenta', ['bold'])
                  
                  distances, indices = ([], [])
                  for i in range(numpts):
                           traj_i = pointCloud[i]
                           distances_and_indices = []

                           for j in range(numpts):
                      
                                  traj_j = pointCloud[j]
                                  dij = self.dist( traj_i, traj_j)
                                  distances_and_indices.append((dij,j))
                
                           # Now sort the distances of all points from point i. 
                           distances_and_indices.sort(key=lambda tup: tup[0]) # http://tinyurl.com/mf8yz5b
                           distances.append( [ d   for ( d,  _ ) in distances_and_indices[0:k] ]  )
                           indices.append  ( [ idx for ( _, idx) in distances_and_indices[0:k] ]  )
       
                  #print "Distance matrix is ", np.array(distances) 
                  #print "Index matrix is  "  , np.array(indices) 
                  print colored('Finished Super-slow brute Force' , 'white', 'on_green', ['bold', 'underline'])
                  return distances, indices

       else: # This means the table has already been computed or read in from a file in the constructor itself
                  print colored('Calling  Memoized brute Force kNN' , 'white', 'on_magenta', ['bold'])
                  
                  #zipDistIdx = zip (self.nbrTable_dist, self.nbrTable_idx)
                  #print zipDistIdx[0][0:k]

                  distances = [ [d   for d   in self.nbrTable_dist[i][0:k]] for i in range(numpts)]        
                  indices   = [ [idx for idx in self.nbrTable_idx[i][0:k] ] for i in range(numpts)]        

                  #print "Distance matrix is ", np.array(distances) 
                  #print "Index matrix is  "  , np.array(indices) 
                  print colored('Finished Memoized brute Force kNN' , 'white', 'on_green', ['bold', 'underline'])
                  return distances, indices





    def rangeSearch(self, pointCloud, radius):
          """ A range search routine.
          Given a point-cloud, return the neighbours within a distance of 'radius'
          for every element of the pointcloud. return the neighbour indices , sorted 
          according to distance. """
          import numpy as np
          from termcolor import colored 
          import itertools as it
          import sys, time

          print colored("Inside trajectory rangeSearch",'white', 'on_magenta',['bold'])


          numpts              = len(pointCloud)


          if self.superSlowBruteForce:
                
                distances, indices = ([], [])
                for i in range(numpts):
                     traj_i = pointCloud[i]
                     distances_and_indices = []

                     for j in range(numpts):
                      
                          traj_j = pointCloud[j]
                          dij = self.dist( traj_i, traj_j)
                          if dij < radius: # We are doing range search 
                                distances_and_indices.append((dij,j))
               
                     # Now sort the distances of all points from point i. 
                     distances_and_indices.sort(key=lambda tup: tup[0]) # http://tinyurl.com/mf8yz5b

                     distances.append([d   for (d, _ ) in distances_and_indices])
                     indices.append  ([idx for (_,idx) in distances_and_indices])
       
                #print "Radius specified was ", colored(str(radius), 'white', 'on_green', ['bold'])
                #print "Distance matrix is \n", np.array(distances) 
                #print "Index matrix is  \n"  , np.array(indices) 
                print colored('Finished rangeSearch Neighbors', 'magenta', 'on_grey', ['bold', 'underline'])
                return distances, indices

          else: # This means the table has already been computed or read in from a file in the constructor itself
                print colored('Calling  Memoized brute Force rangeSearch' , 'yellow', 'on_magenta', ['bold'])
                  
                start = time.time()
                distances, indices = ([], [])
                       
                for i in range(numpts):
                       d_npbr   = np.array(self.nbrTable_dist[i])
                       idx_npbr = np.array(self.nbrTable_idx[i], dtype=int)
                       distances_and_indices = zip ( d_npbr, idx_npbr  )

                       #################################### Bench
                       tmpd   = []
                       tmpidx = []
                       for (d, idx) in distances_and_indices:
                              if d<radius:
                                    tmpd.append(d)
                                    tmpidx.append(idx)

                       distances.append(tmpd)
                       indices.append(tmpidx)                

                       ######################################### Gold : But this compares distance twice 
                       #distances.append([d   for (d ,  _ ) in distances_and_indices if d<radius ])  
                       #indices.append  ([idx for (d , idx) in distances_and_indices if d<radius ])

                end = time.time()       
                print "Time taken for Range Search is ", end-start
                #print "Distance matrix is ", np.array(distances) 
                #print "Index matrix is  "  , np.array(indices) 
                print colored('Finished Memoized brute Force rangeSearch' , 'yellow', 'on_blue', ['bold', 'underline'])
                return distances, indices
   
    def plotClusters(self,  ax            , 
                     trajThickness  = 10 , 
                     marker         = 'o' , 
                     pointCloudInfo = ''  ,
                     annotatePoints = False):
        """ Plot the trajectory clusters computed by the algorithm."""

        from scipy import spatial
        import numpy as np, matplotlib as mpl
        import matplotlib.pyplot as plt
        import colorsys
        import itertools as it

        trajectories = self.pointCloud
        numCars      = len(trajectories)
        numClusters  = len(self.computedClusterings)

        # Generate equidistant colors
        colors       = [(x*1.0/numClusters, 0.5, 0.5) for x in range(numClusters)]
        colors       = map(lambda x: colorsys.hsv_to_rgb(*x), colors)

        # An iterator tht creates an infinite list.Ala Haskell's cycle() function.
        marker_pool  =it.cycle (["o", "v", "s", "D", "h", "x"])
         

        for clusIdx, cluster in enumerate(self.computedClusterings):
             clusterColor = colors[clusIdx]  # np.random.rand(3,1)

             for carIdx in cluster:
                    xdata = [point[0] for point in trajectories[carIdx]]
                    ydata = [point[1] for point in trajectories[carIdx]]

                    # Every line in a cluster gets a unique color     
                    line, = ax.plot(xdata, ydata, 'o-')
                    line.set_color(clusterColor)
                    line.set_markeredgecolor('k')

                    # Cluster center i.e. cluster[0] is made bolder and thicker. Think of it as a highway
                    isClusterCenter = (carIdx == cluster[0])
                    line.set_linewidth(trajThickness + 3*isClusterCenter)
                    line.set_alpha(0.5 + 0.5*isClusterCenter)
 
                    # Only highways are marked with markers 
                    if isClusterCenter:
                         line.set_marker( next(marker_pool) )
                         line.set_markersize(14)
                         line.set_markeredgewidth(2)
                         line.set_markeredgecolor('k')
                         #line.set_markevery(3)
                    


        ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':20})
        ax.set_xlabel('Latitude', fontdict={'fontsize':15})
        ax.set_ylabel('Longitude',fontdict={'fontsize':15})
        #ax.grid(b=True)


    def animateClusters(self, ax, fig, lats, longs,
                     interval_between_frame=200,
                     lineTransparency   = 0.55,
                     markerTransparency = 1.0,
                     saveAnimation=False):
       """Instead of viewing the trajectories like a bowl of spaghetti, watch them 
       evolve in time. Each cluster gets assigned a unique color just like in plotClusters
       interval_between_frames is in milliseconds.
       """ 
       print lats, longs
       numCars      = len(self.pointCloud)
       numClusters  = len(self.computedClusterings)
       numSamples   = len(self.pointCloud[0])
       
       # Generate equidistant colors
       colors       = [(x*1.0/numClusters, 0.5, 0.5) for x in range(numClusters)]
       colors       = map(lambda x: colorsys.hsv_to_rgb(*x), colors)
       
       
       # For each car create a trajectory object. 
       trajectories = []
       for clusIdx, cluster in enumerate(self.computedClusterings):
           print "Setting line"
           linecolor = colors[clusIdx]
           linecolor = ( linecolor[0], linecolor[1], linecolor[2] , lineTransparency) # Augment with a transparency
           markercolor = (linecolor[0], linecolor[1], linecolor[2], markerTransparency)
       
           for traj in cluster:
               print "---< Line Set"
               line, = ax.plot([],[], lw=3, markerfacecolor=markercolor, markersize=5)
               line.set_marker('o')
               line.set_c(linecolor)
       
               trajectories.append(line)
       
       #ax.set_title('r= ' + str(self.r) + + ' Clusters= ', str(numClusters), fontdict={'fontsize':40})
       ax.set_xlabel('Latitude', fontdict={'fontsize':20})
       ax.set_ylabel('Longitude', fontdict={'fontsize':20})
       
       # A special dumb initial function.
       # Absolutely essential if you do blitting
       # otherwise it will call the generator as an
       # initial function, leading to trouble
       def init():
           #global ax
           print "Initializing "
           return ax.lines
       
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
       
           return trajectories
       
       # Call the animator.  blit=True means only re-draw the parts that have changed.
       # Ensures better speed
       
       anim = animation.FuncAnimation(fig, animateData, rGather(),
                                      init_func=init, interval=200, blit=False, fargs=(fig,ax))
       # The draw commands are very important for the animation to be rednered.
       fig.canvas.draw()
       plt.show()
       #anim.save('shenzen_show.mp4', fps=5, extra_args=['-vcodec', 'libx264'])


def getRandomColor():
    """ Ripped from http://goo.gl/SMlEaU"""

    golden_ratio_conjugate = 0.618033988749895

    h = np.random.rand() 
    h += golden_ratio_conjugate
    h %= 1.0
    return colorsys.hsv_to_rgb(h, 0.7, 0.9)
