
#!/usr/bin/python
import matplotlib as mpl, numpy as np, scipy as sp, sys, math, colorsys 
from matplotlib import pyplot as plt, animation 
import networkx as nx, sklearn as sk
from abc import ABCMeta, abstractmethod
from haversine import haversine # https://pypi.python.org/pypi/haversine

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
            ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':5})
            ax.set_xlabel('Latitude', fontdict={'fontsize':5})
            ax.set_ylabel('Longitude',fontdict={'fontsize':5})
    
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
    
    

class AlgoAggarwalStatic:
  __metaclass__ = ABCMeta

  def __init__(self,r,pointCloud):
    """ Even though this is an abstract class, a subclass is 
        allowed to call the constructor via super. 
        However, a user cannot instantiate a class with this 
        method from his code."""
    pass 

 
  @abstractmethod
  def dist(p,q):
    """ A distance function of a metric space.
        distance between points p and q. Implemented 
        by the subclass. """
    pass

  @abstractmethod
  def rangeSearch( pointCloud, radius):
    """ Given a set of points in the metric space, and a radius value
        find all the neighbours for a point in 'pointCloud' in a ball of radius, 
        'radius', for all points in 'points'. Depending on the metric space 
        an efficient neighbour search routine will use different tricks """ 
    pass 

  def generateClusters(self):
    from   colorama import Fore, Style 
    import pprint as pp 
    import networkx as nx, numpy as np, random, time 
    import scipy as sp
    import matplotlib.pyplot as plt
    import sys
    points    = self.pointCloud # a conveninent alias 
    numPoints = len( self.pointCloud )
  
    def firstConditionPredicate( R ):
        import time
    
        # # Team Scipy
        # start = time.time()
        # #print R
        # distances, everyonesBall2R_Neighbors = self.rangeSearch( self.pointCloud, 2*R )
        # end = time.time()
        # print(end - start), "seconds"
        #assert( len( everyonesBall2R_Neighbors ) == len( self.pointCloud ) )
        # Check if everyone has sufficiently many neighbours.
        #return  all(   [True if len(nbrList) >= self.r else False 
        #                     for nbrList in everyonesBall2R_Neighbors]   )
    
        # Team Sklearn
        start = time.time()
        distances, r_nearest_indices = self.findNearestNeighbours( self.pointCloud, self.r  )      
    
        flags = []
        for i in range( numPoints ):
            flagi = [True if self.dist( self.pointCloud[i], self.pointCloud[nbr]  ) <= 2*R  else False for nbr in r_nearest_indices[i] ] 
            flags.append( all(flagi) )
         
            #if all( flagi ): # All points within the distance of 2*R
            #    flags.append( all(flagi)  )
    
        end   = time.time()
        print (end-start), " seconds"
    
        #print distances, r_nearest_indices
    
        #print flags
        #sys.exit()
    
        return all( flags ) 
    
    def makeClusterCenters( R,
                            points = self.pointCloud, 
                            dist   = self.dist      , 
                            r      = self.r         ):
          """ Marking loop for choosing good cluster centers """
          import numpy as np
          from sklearn.neighbors import NearestNeighbors
    
          numPoints               = len( points )
          markers                 = [ False for i in range( numPoints ) ]
          potentialClusterCenters = [ ] # Populated in the while loop below.  
     
          # Warning: The n_neighbors=r was chosen by me arbitrarily. Without this, the default parameter chosen by sklearn is 5
          # Might have to do replace this with something else in the future me thinks.  
          nbrs_datastructure = NearestNeighbors (n_neighbors=r, radius=2*R , algorithm='ball_tree',metric=self.dist , n_jobs=-1).fit( points ) 
          # See note above. It might be very important! 
          # The following while loop replacement to the confusing tangle spelled out in the Aggarwal 
          # paper was suggested by Jie and Jiemin in the email thread with Rik, after I cried for help. 
          while( all( markers ) !=  True ): 
               
              unmarkedIndices =  [ index for ( index,boolean ) 
                                         in zip( range( numPoints ), markers) 
                                         if boolean == False ]
           
              randomIndex = random.choice ( unmarkedIndices ) 
    
              # WARNING: THE INDICES ARE NOT SORTED ACCORDING TO THE DISTANCE FROM the RANDOMINDEX point
              (_, idx_nbrs) = nbrs_datastructure.radius_neighbors( X=[points[randomIndex]], radius=2*R ) 
              # this list needs to be analysed properly.
             
              ball2R_neighbor_list = idx_nbrs[0]
             
              #print ball2R_neighbor_list 
     
              # Mark all the neighbours including the point itself. 
              for nbrIndex in ball2R_neighbor_list:
                     markers[ nbrIndex ] = True 
      
              potentialClusterCenters.append( ( randomIndex, ball2R_neighbor_list ) ) 
    
    
          print " All points marked! "
          # Cluster centers are those which have atleast r points in their neighbourhood. 
          clusterCenters = [ index for ( index, ball2R_neighbor_list ) in potentialClusterCenters 
                              if len( ball2R_neighbor_list ) >= r  ]
    
    
          # Having marked all the points, return the cluster centers. 
          return clusterCenters # There are two such assumptions. 
    def makeFlowNetwork( R                       ,
                         clusterCenters          ,
                         points = self.pointCloud,
                         r      = self.r         ): 
    
        # Set the nodes of the network and some attributes
        numPoints = len( points )
        G = nx.DiGraph() # Initialize an empty flow network
    
        G.add_node( 's', demand = -r*len(clusterCenters) ) # Source
        G.add_node( 't', demand =  r*len(clusterCenters) ) # Sink
        G.add_nodes_from( range(numPoints) ) # The actual points 
    
    
        # Give cluster centers a special attribute marking it as a center. 
        isClusterCenterDict = { } 
    
        for i in range( numPoints ):
            if i in clusterCenters:
                isClusterCenterDict[ i ] = True
            else:
                isClusterCenterDict[ i ] = False
    
        # Source and sink are "fake" nodes and hence not centers.
        isClusterCenterDict['s'] = False
        isClusterCenterDict['t'] = False
    
        nx.set_node_attributes( G,'isCenter', isClusterCenterDict )
    
        # Set the EDGES of the network and its sttributes
        # Source edges
        for i in clusterCenters:
            G.add_edge( 's', i , capacity = r )
    
         # Interior edges i.e those whose endpoints are neither 's' not 't'
         #distances, nbrlistsClusterCenters = self.rangeSearch( [ points[ i ] for i in clusterCenters ] , 2*R   ) # For each cluster center, get neighbours in the point-cloud within distance 2*R.
    
         #      print clusterCenters, R
         #      print nbrlistsClusterCenters
         #      import sys
         #      sys.exit() 
    
         #      for i in clusterCenters:
         #          for j in nbrlistsClusterCenters : # For each of i's neighbours, except itself, add an edge in the flow network emanating from i's node
         #              if i != j:
         #                G.add_edge (i, j, capacity = 1.0)  
                
    
        for i in clusterCenters:
             for j in range( numPoints ):
    
                 if i != j and self.dist( points[ i ], points[ j ] ) <= 2*R:
                    G.add_edge( i, j, capacity = 1.0 ) 
    
    
    
        # Sink edges
        for i in range( numPoints ):
            G.add_edge( i, 't', capacity =  1.0 )
    
        return G
    def makeClusters( bestRflowDict, clusterCenters, bestRflowNetwork , R ):
        """ Construct the clusters out of the network obtained.  """ 
    
        pp.pprint ( bestRflowDict )
        clusterings = [ ] 
        for v in bestRflowNetwork.nodes():
            
            if bestRflowNetwork.node[ v ]['isCenter'] == True: 
              # Every cluster center becomes 
              # the first node of its cluster. 
              cluster = [ v ]
    
              for successor in bestRflowNetwork.successors( v ): 
    
                if successor != 't':
                   #print "v= ", v, " successor= ", successor
                   if bestRflowDict[ v ][ successor ] > 1-0.001: # Have to be careful.since comparing to 1.0 may be problematic. Hence the little cushion of 0.001
                       cluster.append( successor )
    
              assert( len( cluster ) >= self.r  )
              # Wrap up by registering this newly reported cluster.
              clusterings.append( cluster )
    
        # Some nodes (FORGET ABOUT 'S' AND 'T', THEY DON' COUNT ANY MORE) were probably missed 
        # by the clusters. Add them to one of the clusters obtained above. 
        coveredNodes = set([ i for cluster in clusterings for i in cluster ] )
        missedNodes  = set(range( numPoints ) ).difference( coveredNodes )
    
        #print missedNodes
    
        for missedNode in missedNodes:
           
            # Find the cluster whose center is nearest to missedNode
            dist2NearestClusterCenter = float("inf")
            for i in range( len( clusterings ) ):
                clusterCenter      = clusterings[i][0]# Head of the cluster is the center. 
                dist2clusterCenter = self.dist( points[ missedNode ] , points[ clusterCenter ] ) 
                if dist2clusterCenter <= min( dist2NearestClusterCenter, 2*R):
                    dist2NearestClusterCenter = dist2clusterCenter
                    nearestClusterIndex       = i # WARNING! This does NOT index into points. It indexes into clusterings array
    
            clusterings[ nearestClusterIndex ].append( missedNode )        
    
    
        # Add missed nodes to clusters  
        # for missedNode in missedNodes:
        #     for cluster in clusterings:
        #         clusterCenter      = cluster[ 0 ] # That's how the clusters were constructed in the for loop
        #         dist2clusterCenter = self.dist( points[ missedNode ], points[ clusterCenter ]) 
        #         if dist2clusterCenter <= 2*R: # TODO
        #             print Fore.CYAN, dist2clusterCenter, " <= ", 2*R, Style.RESET_ALL 
        #             print "Appending missed node ", missedNode, " to cluster with Center ", clusterCenter 
        #             cluster.append( missedNode )
             
    
        #print Fore.YELLOW, clusterings, Style.RESET_ALL
    
        # Make sure all points have been covered in the clustering
        return clusterings
  
    #print "Started filtering!"
  
    #print "The points are ", points   
    #print "Number of points", numPoints
  
    #import sys
    #sys.exit()
  
    dijHalfs = [0.5 * self.dist( points[ i ], points[ j ] ) 
                      for i in range( numPoints ) 
                      for j in range( i+1, numPoints ) ]
    # Find all dijs satisfying condition 1 on page 4
  
    print "dijhalfs computed", len(dijHalfs)
    dijHalfsFiltered =  filter( firstConditionPredicate, dijHalfs )  #smallest to highest
    print "dijHalfsFiltered done!"
  
    # 'FOR' Loop to find the minimum 'R' from these filtered dijs satisfying 
    #  condition 2 on page 4 of the paper. 
    bestR, bestRflowNetwork, bestRflowDict = float( 'inf' ), nx.DiGraph(), {} 
    bestRCenters = []
  
    for R in dijHalfsFiltered : # The first R that goes through the else block is the required R
  
      clusterCenters = makeClusterCenters( R )
      flowNetwork    = makeFlowNetwork( R, clusterCenters )
  
      try: # Check if a feasible flow exists in the constructed network.  
            flowDict = nx.min_cost_flow( flowNetwork )
  
      except nx.NetworkXUnfeasible:# If not, try the next R
            print Fore.RED, "Unfeasible R detected: R= ", R, Style.RESET_ALL
            continue 
      else: # Found a feasible R.  
          print "Found a feasible R! R= ", R
          if R < bestR: # Yippee a smaller and feasible R! Update bestR. 
              print Fore.RED, " In fact, it is the best thus far ", Style.RESET_ALL 
              bestR            = R
              bestRflowNetwork = flowNetwork
              bestRflowDict    = flowDict
              bestRCenters     = clusterCenters
  
  
    #Use the best network to construct the needed clusters. 
    self.computedClusterings = makeClusters( bestRflowDict, bestRCenters, bestRflowNetwork, R)
  
    # Sanity check on the computed clusters. They should all be of size r and should cover the full point set
    assert( all( [ len(cluster) >= self.r for cluster in self.computedClusterings ] ) )
    assert( len( { i for cluster in self.computedClusterings for i in cluster } ) == numPoints   )
    print Fore.YELLOW, "Yay All points Covered!!", Style.RESET_ALL
   
    print "BestRCenters are ", bestRCenters 
    return  bestRCenters


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
           ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':5})
           ax.set_xlabel('Latitude', fontdict={'fontsize':5})
           ax.set_ylabel('Longitude',fontdict={'fontsize':5})
   
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
     
    def __init__(self, r,  pointCloud):
       # len(trajectories) = number of cars
       # len(trajectories[i]) = number of GPS samples taken for the ith car. For shenzhen data set this is
       # constant for all cars.

       self.r                    = r     
       self.pointCloud           = pointCloud # Should be of type  [ [(Double,Double)] ] 
       self.computedClusterings  = []  
       self.algoName             = 'r-Gather for trajectory clustering'
  

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

       p,q :: [(Double,Double)]. The length of p or q, indicates the number of GPS samples taken
       
       """
       dpq = 0
       for t in range(len(p)):
            
            # M is the euclidean distance between two points at time t.  
            M = np.sqrt( abs( (p[t][0]-q[t][0])**2 + (p[t][1]-q[t][1])**2 ) ) 
            if M > dpq:
                dpq = M

       return dpq


    def findNearestNeighbours(self, pointCloud, k):
       """return the k-nearest nearest neighbours"""
       import numpy as np
       from sklearn.neighbors import NearestNeighbors

       # build a data-structure for fast retrieval
       (distances, indices) = NearestNeighbors(n_neighbors = k, 
                                               algorithm   = 'ball_tree', 
                                               metric      = self.dist, 
                                               n_jobs=-1).fit( pointCloud ).kneighbors( pointCloud )
       print "Inside findNearestNeighbor"
       return distances, indices


    def rangeSearch(self, pointCloud, radius):
        print "Warning! Rangesearch used for trajectories"
        pass


    
    
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
            ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':5})
            ax.set_xlabel('Latitude', fontdict={'fontsize':5})
            ax.set_ylabel('Longitude',fontdict={'fontsize':5})
    
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
    
    


def getRandomColor():
    """ Ripped from http://goo.gl/SMlEaU"""

    golden_ratio_conjugate = 0.618033988749895

    h = np.random.rand() 
    h += golden_ratio_conjugate
    h %= 1.0
    return colorsys.hsv_to_rgb(h, 0.7, 0.9)
