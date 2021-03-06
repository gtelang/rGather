
#!/usr/bin/python
import matplotlib as mpl 
from matplotlib import pyplot as plt, animation 
import numpy as np
import scipy as sp
from scipy import spatial
import sys
import math
import networkx as nx, sklearn as sk
from abc import ABCMeta, abstractmethod
import colorsys 
from termcolor import colored
import itertools
import pprint as pp
import copy
import yaml
from sklearn.neighbors import NearestNeighbors

def getRandomColor():
    """ Ripped from http://goo.gl/SMlEaU"""
    h = np.random.rand() 
    h +=  0.618033988749895 # golden_ratio_conjugate
    h %= 1.0
    return colorsys.hsv_to_rgb(h, 0.7, 0.9)

# Here we use the 2-approximation algorithm for general metric spaces 
# described in the aggarwal paper on r-Gather.
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
    
        start = time.time()
        distances, r_nearest_indices = self.findNearestNeighbours( self.pointCloud, self.r  ) # This is the bottleneck inside your code.      
        endknn = time.time()
        print 'Just the knn inside firstConditionPredicatetook ', (endknn-start), "seconds"
    
        #------------------ Gold
        #flags = []
        #for i in range( numPoints ):
        #    flagi = [True if self.dist( self.pointCloud[i], self.pointCloud[nbr]  ) <= 2*R  else False for nbr in r_nearest_indices[i] ] 
        #    flags.append( all(flagi) )
         
        #    #if all( flagi ): # All points within the distance of 2*R
        #    #    flags.append( all(flagi)  )
    
        #return all( flags ) 
    
        #------------------ Bench
        for i in range( numPoints ):
            for j in range(len(r_nearest_indices[i])):
                 if distances[i][j] >= 2*R :
                      endfn   = time.time()
                      print 'firstConditionPredicate took ' , (endfn-start) , "seconds"
                      return False
    
        endfn   = time.time()
        print 'firstConditionPredicate took ' , (endfn-start) , "seconds"
        return True # non of flagis tested negative.
    
    
    def makeClusterCenters( R,
                            points = self.pointCloud, 
                            dist   = self.dist      , 
                            r      = self.r         ):
          """ Marking loop for choosing good cluster centers """
    
          numPoints               = len( points )
          markers                 = [ False for i in range( numPoints ) ]
          potentialClusterCenters = [ ] # Populated in the while loop below.  
     
          # Warning: The n_neighbors=r was chosen by me arbitrarily. Without this, the default parameter chosen by sklearn is 5
          # Might have to do replace this with something else in the future me thinks.  
          #nbrs_datastructure = NearestNeighbors (n_neighbors=r, radius=2*R , algorithm='ball_tree',metric=self.dist , n_jobs=-1).fit( points ) 
          # See note above. It might be very important! 
          # The following while loop replacement to the confusing tangle spelled out in the Aggarwal 
          # paper was suggested by Jie and Jiemin in the email thread with Rik, after I cried for help. 
    
          # First get all the points within distance 2*R for EVERY point in the cloud.
          (_, idx_nbrs_2R) = self.rangeSearch( points, 2.0*R )
          while( all( markers ) !=  True ): 
               
              unmarkedIndices =  [ index for ( index,boolean ) 
                                         in zip( range( numPoints ), markers) 
                                         if boolean == False ]
           
              randomIndex          = random.choice ( unmarkedIndices ) 
              ball2R_neighbor_list = idx_nbrs_2R[randomIndex]
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
  
    print "Started filtering!"
  
    #print "The points are ", points   
    #print "Number of points are", numPoints
  
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
  
    from termcolor import colored
  
    for R in sorted(dijHalfsFiltered) : # The first R that goes through the else block is the required R
     
      print colored(str(R) + 'is being tested', 'red', 'on_white', ['underline', 'bold'])
      clusterCenters = makeClusterCenters( R )
      flowNetwork    = makeFlowNetwork( R, clusterCenters )
  
      try: # Check if a feasible flow exists in the constructed network.  
            flowDict = nx.min_cost_flow( flowNetwork )
  
      except nx.NetworkXUnfeasible:# If not, try the next R
            print Fore.RED, "Unfeasible R detected: R= ", R, Style.RESET_ALL
            continue 
      else: # Found a feasible R.  
          print "Found an feasible R! R= ", R
          print Fore.RED, " In fact, it is the best thus far ", Style.RESET_ALL 
          bestR            = R
          bestRflowNetwork = flowNetwork
          bestRflowDict    = flowDict
          bestRCenters     = clusterCenters
          break 
  
    #Use the best network to construct the needed clusters. 
    self.computedClusterings = makeClusters( bestRflowDict, bestRCenters, bestRflowNetwork, bestR)
    #====================================================================================================
    ## Check if all clusters are small enough. i.e. don't have too many points. 
    #largeclusters = [cluster for cluster in self.computedClusterings  if len(cluster) >= 2*self.r and len(cluster) < len(self.pointCloud)]
    #
    #print largeclusters
    #sys.exit()
    #if largeclusters: # A non-empty list evaluates to True. The Pythonic way.
    #         # Iterate through each cluster and run the r-Gather algorithm on it. 
    #         for cluster in largeclusters:
    #
    #               cluster_size = len(cluster)
    #
    #               if cluster_size >= 2*self.r and cluster_size != len(self.pointCloud):
    #
    #                 cluster_pointCloud = [self.pointCloud[i]  for i in cluster]  # bingo, this is why point-cloud has to be passed to refineLargeClusters !!! finally!              
    #
    #                 run = AlgoJieminDynamic(r= self.r, pointCloud= cluster_pointCloud, memoizeNbrSearch= True) 
    #                 clusterCenters = run.generateClusters() 
    #                 #print clusterCenters
    #                                              
  
    # Things to do: 
    # ---. Abstract recursion as a function with the larger clusters.
    # candidateClusters  = makeClusters (<blah-blah>)
    # largeClusters   = [cluster for cluster in candidateClusters  if len(cluster) >= 2*self.r and len(cluster) < len(self.pointCloud)] # When this is empty, function will return.
    # smallClusters   = Take complement. 
    # refinedClusters = refineLargeClusters (largeclusters, self.pointCloud) # refer to the large clusters of the pointcloud 
    # self.computedClusterings,  = smallClusters + refinedClusters # (+) is indeed the concatenation operator.
    # coordinatesActualExtractopn = using self.computedClusterings extract it. This will be necessary when passing on to run.
    # return coOrdinatesActualExtraction, centers # if you don't really want them, the callee will not store them. He will be interested  in the self.computedClusterings variable.
  
    # all the magic happens in the refineLargeClusters function, INCLUDING the list-flattening operation. Have a dedicated list-flattener also! Defined
  
    # makeClusters just returns indices. self.computedClusterings are then extracted/set as actual trajectories
    # Another option, possibly better will be for makeClusters to return both indexes and coordinates. 
    # Then we can use pattern-matching _,blah and blah,_ style sth.
    # This will support both recursive and non-recursive variants.
    # plotClusters function will then need to be changed at some points
  
    # that supports making , the clusters, and solving the neighbor problem
    # now for the most important: how do we return????, that is implement the splitting? 
    # certainly, just before returning you will have to flatten the list.
  
    #=====================================================================================================
    # Sanity check on the computed clusters. They should all be of size r and should cover the full point set
    assert( all( [ len(cluster) >= self.r for cluster in self.computedClusterings ] ) )
    assert( len( { i for cluster in self.computedClusterings for i in cluster } ) == numPoints   )
    print Fore.YELLOW, "Yay All points Covered!!", Style.RESET_ALL
  
    print "BestRCenters are ", bestRCenters 
  
    # Print the clusters along with their sizes
    print colored(str(len(self.computedClusterings)) + ' clusters have been computed on ' + \
                  str(len(self.pointCloud)) + ' elements' , 'magenta',  attrs=['bold', 'underline'] )
    for i, cluster in enumerate(self.computedClusterings):
        print "Cluster(", i+1, ") Size:", len(cluster), "  ", np.array( cluster ) 
  
    
    #raw_input('Press Enter to continue...')
  
    return  bestRCenters
   # Abstract class

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
      
      X    = np.array(pointCloud)
      nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
      distances, indices = nbrs.kneighbors(X)

      return distances, indices


   def rangeSearch(self, pointCloud, radius):
      """ A wrapper for a good neighbour search routine provided by Scipy.
          Given a point-cloud, return the neighbours within a distance of 'radius'
          for every element of the pointcloud. return the neighbour indices , sorted 
          according to distance. """
      from scipy import spatial
      
      X        = np.array( pointCloud )
      mykdtree = spatial.KDTree( X )
      nbrlists = list( mykdtree.query_ball_point( X, radius) )
     

      distances = []
      for index  in  range(len(nbrlists)):

         def fn_index( i ): # Distance function local to this iteration of the loop
            #return np.linalg.norm(  [  X[i][0] - X[index][0]   , 
            #                           X[i][1] - X[index][1]    ]    )
            return self.dist(X[i], X[index])

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
   
       # For static points in the euclidean plane with the L2 metric
class AlgoJieminDynamic( AlgoAggarwalStatic ):
     
    def __init__(self, r,  pointCloud,  memoizeNbrSearch = True, distances_and_indices_file=''):
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
       self.algoName             = '2-APX for trajectory clustering'
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
                     annotatePoints = False,
                     plot_xytspace = False):
        """ Plot the trajectory clusters computed by the algorithm."""

        import colorsys
        import itertools as it

        trajectories = self.pointCloud
        numCars      = len(trajectories)
        numClusters  = len(self.computedClusterings)

        # Generate equidistant colors
        colors       = [(x*1.0/numClusters, 0.5, 0.5) for x in range(numClusters)]
        colors       = map(lambda x: colorsys.hsv_to_rgb(*x), colors)

        # An iterator tht creates an infinite list.Ala Haskell's cycle() function.
        #marker_pool  =it.cycle (["o", "v", "s", "D", "h", "x"])
         

        for clusIdx, cluster in enumerate(self.computedClusterings):
             clusterColor = colors[clusIdx]  # np.random.rand(3,1)

             for carIdx in cluster:
                    xdata = [point[0] for point in trajectories[carIdx]]
                    ydata = [point[1] for point in trajectories[carIdx]]

                    if plot_xytspace == True:
                       timeStamps = np.linspace(0, 1, len(xdata))
                       #print "TimeStamps are : ", timeStamps
                       line, = ax.plot(xdata, ydata, timeStamps, 'o-')

                    else:
                         line, = ax.plot(xdata, ydata, 'o-')
   
                    # Every line in a cluster gets a unique color     
                    line.set_color(clusterColor)
                    line.set_markeredgecolor('k')

                    # Cluster center i.e. cluster[0] is made bolder and thicker. Think of it as a highway
                    #isClusterCenter = (carIdx == cluster[0])
                    #line.set_linewidth(trajThickness + 3*isClusterCenter)
                    #line.set_alpha(0.5 + 0.5*isClusterCenter)
 
                    # Only highways are marked with markers 
                    #if isClusterCenter:
                    #     line.set_marker( next(marker_pool) )
                    #     line.set_markersize(14)
                    #     line.set_markeredgewidth(2)
                    #     line.set_markeredgecolor('k')
                    #     line.set_markevery(3)
                    
        ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':24})
        ax.set_xlabel('Latitude', fontdict={'fontsize':23})
        ax.set_ylabel('Longitude',fontdict={'fontsize':23})
        if plot_xytspace == True:
             ax.set_zlabel('Time', fontdict={'fontsize':23})

    def animateClusters(self, ax, fig, lats, longs,
                     interval_between_frame=200,
                     lineTransparency   = 1.0,
                     markerTransparency = 1.0,
                     saveAnimation=False):
       """Instead of viewing the trajectories like a bowl of spaghetti, watch them 
       evolve in time. Each cluster gets assigned a unique color just like in plotClusters
       interval_between_frames is in milliseconds.
       """ 
     

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
           linecolor   = colors[clusIdx]
           linecolor   = ( linecolor[0], linecolor[1], linecolor[2] , lineTransparency) # Augment with a transparency
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
       #anim.save('shenzen_show_scrap.mp4', fps=5, extra_args=['-vcodec', 'libx264']) ; print "Animation saved!" # For trajectories in the euclidean plane with the linifinity-like metric 

# The abstract class has both versions of the 4-approximation
# algorithm i.e. the simple one and the improved distributed sweep 
# algorithm described by Prof. Mitchell.
class Algo_4APX_Metric:
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

    
    def generateClusters(self, config={'mis_algorithm': 'networkx_random_choose_20_iter_best'}):
      """ config : Configuration parameters which might be needed 
                   for the run. 
      Options recognized are (ALL LOWER-CASE)
      1. mis_algorithm:
           A. 'networkx_random_choose_20_iter_best', default 
           B. 'riksuggestion'
           C. 'sweep'
      """
      import pprint as pp
      
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
              distanceFromRthNearestNeighbourDict[i] = self.dist(nbdCenterCoords, nbdFarthestNeighbourCoords)
      
          nx.set_node_attributes( G, 'distanceFromRthNearestNeighbour', distanceFromRthNearestNeighbourDict )
      
          import collections
          # Generate the order to remove the vertices
          orderOfVerticesToDelete = collections.deque(sorted(  range(len(nbds)) , key = lambda x: G.node[x][ 'distanceFromRthNearestNeighbour' ]    ))
          sIndices = [ ]
      
      
          for i in orderOfVerticesToDelete:
      
            try:
               node  = orderOfVerticesToDelete[i]
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
      
      
        # Hard-hat area! Here be dragons.
        elif mis_algorithm == 'sweep':
          pass 
      
      
      
        else:
          print "Maximum independent Set Algorithm option ", mis_algorithm ," not recognized!"
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
      
        return [ nbds[s] for s in sIndices ]
      
      def extractUniqueElementsFromList( L ):
          
          uniqueElements = []
          for elt in L:
              if elt not in uniqueElements: # Just discovered a brand new element!!
                  uniqueElements.append(elt)
      
          return uniqueElements
    
    
      NrDistances, Nr = self.findNearestNeighbours( self.pointCloud, self.r )
    
      Nr              = np.array(Nr) # if Nr is a list, convert to an np.array. Note that np.array function is idempotent.
      S               = findMaximalIndependentOfNeighbourhoods( Nr.tolist( ), 
                                                                config[ 'mis_algorithm' ])
     
      #print Nr # Nr is of type [[Int]] where [Int] here refers to a neighborhood.
      #print S  # S should be a subset of Nr
      #sys.exit()
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
    
               dist = self.dist(ptIndex , ptnbIndex) 
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
    
    
    generateClustersSimple = generateClusters # Hack for backward compatibility. Had not read fact that only M2 needed to be modified.
    
       # Abstract class
class Algo_Static_4APX_R2_L2 (Algo_4APX_Metric):
    
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
    
    
    def dist(self, p,q):
          """ Euclidean distance between points p and q in R^2 """
          return np.linalg.norm( [ p[0]-q[0] , 
                                   p[1]-q[1] ]  )
    
    
    def findNearestNeighbours(self, pointCloud, k):
      """  pointCloud : 2-d numpy array. Each row is a point
           k          : The length of the neighbour list to compute. 
      """
    
      X    = np.array(pointCloud)
      nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit( X )
      distances, indices = nbrs.kneighbors(X)
    
      return distances, indices
    
    
    def rangeSearch(self, pointCloud, radius):
          """ A wrapper for a good neighbour search routine provided by Scipy.
              Given a point-cloud, return the neighbours within a distance of 'radius'
              for every element of the pointcloud. return the neighbour indices , sorted 
              according to distance. """
          
          X        = np.array( pointCloud )
          mykdtree = spatial.KDTree( X )
          nbrlists = list( mykdtree.query_ball_point( X, radius) )
         
    
          distances = []
          for index  in  range(len(nbrlists)):
    
             def fn_index( i ): # Distance function local to this iteration of the loop
                #return np.linalg.norm(  [  X[i][0] - X[index][0]   , 
                #                           X[i][1] - X[index][1]    ]    )
                return self.dist(X[i], X[index])
    
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
    
        # For points in the euclidean plane with the L2 metric
class Algo_Dynamic_4APX_R2_Linf ( Algo_4APX_Metric ):
     
    def __init__(self, r,  pointCloud,  memoizeNbrSearch = True, distances_and_indices_file=''):
       """ Initialize the AlgoJieminDynamic
 
           memoizeNbrSearch = this computes the table in the constructor itself. no need for a file. The file option below, is only useful for large runs.
           distances_and_indices_file = must be a string identifer for the file-name on disk. 
                                        containing the pairwise-distances and corresponding index numbers
                                        between points. I had to appeal to this hack, since sklearn's algorithm to search in arbitrary metric spaces does not work for my case. 
                                        Also the brute-force computation, which I initially implemented took far too long. 
                                        Since  don't know how to do the neighbor computation for arbitrary metric spaces, I just precompute 
                                        everything into a table, stored in a YAML file.
       """


       # len(trajectories) = number of cars
       # len(trajectories[i]) = number of GPS samples taken for the ith car. For shenzhen data set this is
       # constant for all cars.

       self.r                    = r     
       self.pointCloud           = pointCloud # Should be of type  [ [(Double,Double)] ] 
       self.computedClusterings  = []  
       self.algoName             = '4-APX for trajectory clustering'
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
                     markersize     = 15,
                     pointCloudInfo = ''  ,
                     annotatePoints = False,
                     plot_xytspace = False):
        """ Plot the trajectory clusters computed by the algorithm."""

        import colorsys
        import itertools as it

        trajectories = self.pointCloud
        numCars      = len(trajectories)
        numClusters  = len(self.computedClusterings)

        # Generate equidistant, hence maximally dispersed colors.
        colors       = [(x*1.0/numClusters, 0.5, 0.5) for x in range(numClusters)]
        colors       = map(lambda x: colorsys.hsv_to_rgb(*x), colors)


        print "Plot colors are", colors 

        for clusIdx, cluster in enumerate(self.computedClusterings):
             clusterColor = colors[clusIdx]  # np.random.rand(3,1)

             for carIdx in cluster:
                    xdata = [point[0] for point in trajectories[carIdx]]
                    ydata = [point[1] for point in trajectories[carIdx]]
                             
                    # if plot is three d.
                    if plot_xytspace == True:

                       timeStamps = np.linspace(0, 1, len(xdata))
                       #print "TimeStamps are : ", timeStamps
                       line, = ax.plot(xdata, ydata, timeStamps, marker='o', markersize=markersize)
                       
                    else: # else if plot is 2d
                       line, = ax.plot(xdata, ydata, 'o-', markersize=markersize)
                       #print type(ax)
                    # Every line in a cluster gets a unique color     
                    line.set_color(clusterColor)
                    line.set_markeredgecolor('k')

        ax.set_title( self.algoName + '\n r=' + str(self.r), fontdict={'fontsize':24})
        ax.set_xlabel('Latitude', fontdict={'fontsize':22})
        ax.set_ylabel('Longitude',fontdict={'fontsize':22})
        if plot_xytspace:
              ax.set_zlabel('Time', fontdict={'fontsize':22})

    def animateClusters(self, ax, fig, lats, longs,
                     interval_between_frames=500,
                     lineTransparency   = 1.0,
                     markerTransparency = 1.0,
                     markersize =15,
                     saveAnimation=False):
       """Instead of viewing the trajectories like a bowl of spaghetti, watch them 
       evolve in time. Each cluster gets assigned a unique color just like in plotClusters
       interval_between_frames is in milliseconds.
       """ 
 
       import colorsys
       import itertools as it

       # A special dumb initial function.
       # Absolutely essential if you do blitting
       # otherwise it will call the generator as an
       # initial function, leading to trouble
       def init():
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
           
           numCars      = len(self.pointCloud)
           numClusters  = len(self.computedClusterings)
           numSamples   = len(self.pointCloud[0])

           # Generate equidistant, hence maximally dispersed colors.
           colors       = [(x*1.0/numClusters, 0.5, 0.5) for x in range(numClusters)]
           colors       = map(lambda x: colorsys.hsv_to_rgb(*x), colors)



           for i in range(numSamples):
               ax.lines = [] 
               # Set trajectories to be a bundle of empty lines, with all colors set.
               trajectories = []
               for clusIdx, cluster in enumerate(self.computedClusterings):
                   clusterColor = colors[clusIdx]  # np.random.rand(3,1)
               
                   for carIdx in cluster:
                        
                        xdata = lats [0:i+1,carIdx]
                        ydata = longs[0:i+1,carIdx]
                        line, = ax.plot(xdata, ydata, 'o-', markersize=markersize)
                        line.set_color(clusterColor)
                        line.set_markeredgecolor('k')
                        line.set_alpha(lineTransparency)
                        trajectories.append(line) # Add the reference to the line object.2
       
               yield trajectories, i
       
       
       # Separating the animateData and the rGather generator function allows
       def animateData(state, fig, ax):
           """ Render the trajectories rendered by the rGather algorithms
           and add fancy effects.
           """
           trajectories = state[0] # All trajectories
           currentTime  = state[1] # The time at which to animate
           ax.set_title( self.algoName + '\n r=' + str(self.r) + '  ' +  ' Dynamic Algorithm ' + str(currentTime) + '/' + str(len(self.pointCloud[0])-1), 
                         fontdict={'fontsize':24})
           ax.set_xlabel('Latitude', fontdict={'fontsize':22})
           ax.set_ylabel('Longitude',fontdict={'fontsize':22})

           for line in ax.lines:
                line.set_markevery((currentTime, currentTime))

           print colored(str(currentTime) +  ' ' + str(len(ax.lines)), 'cyan')
           return trajectories
       
       # Call the animator.  blit=True means only re-draw the parts that have changed.
       # Ensures better speed
       
       anim = animation.FuncAnimation(fig, animateData, rGather(),
                                      init_func=init, interval=interval_between_frames, blit=False, fargs=(fig,ax))
       # The draw commands are very important for the animation to be rednered.
       fig.canvas.draw()
       plt.show()
       anim.save('dynamic_clustering.mp4', fps=5, extra_args=['-vcodec', 'libx264']) ; print "Animation saved"


    def mkClustersEveryTimeStep( self, ax, fig, lats, longs, 
                                 interval_between_frames=500,
                                 lineTransparency   = 1.0,
                                 markerTransparency = 1.0,
                                 markersize =15,
                                 saveAnimation=False):
       import colorsys
       import itertools as it

       # A special dumb initial function.
       # Absolutely essential if you do blitting
       # otherwise it will call the generator as an
       # initial function, leading to trouble
       def init():
           print "Initializing "
           return ax.lines
       
       # Update the state of rGather
       def rGather():
           """ Run the static online r-gather algorithm at each time-step as the cars
           move around. 
           """
           
           numCars      = len(self.pointCloud)
           numSamples   = len(self.pointCloud[0])

           for i in range(numSamples):
               ax.lines = [] 

               currentPositions = []
               for trajectory in self.pointCloud:
                       x, y = trajectory[i][0], trajectory[i][1]
                       currentPositions.append([x,y])                 

               #print np.array(currentPositions)
               #print np.array(currentPositions).shape
               tmprun_static = Algo_Static_4APX_R2_L2( r= self.r, pointCloud= np.array(currentPositions))
               tmprun_static.generateClusters( config ={'mis_algorithm':'networkx_random_choose_20_iter_best' }  ) 
               numClusters = len(tmprun_static.computedClusterings)

               # Generate equidistant, hence maximally dispersed colors.
               colors       = [(x*1.0/numClusters, 0.5, 0.5) for x in range(numClusters)]
               colors       = map(lambda x: colorsys.hsv_to_rgb(*x), colors)

               # Set trajectories to be a bundle of empty lines, with all colors set.
               trajectories = []
               for clusIdx, cluster in enumerate(tmprun_static.computedClusterings):
                   clusterColor = colors[clusIdx]  # np.random.rand(3,1)
               
                   for carIdx in cluster:
                        
                        xdata = lats [0:i+1,carIdx]
                        ydata = longs[0:i+1,carIdx]
                        line, = ax.plot(xdata, ydata, 'o-', markersize=markersize)
                        line.set_color(clusterColor)
                        line.set_markeredgecolor('k')
                        line.set_alpha(lineTransparency)
                        trajectories.append(line) # Add the reference to the line object.
       
               yield trajectories, i
       
       
       # Separating the animateData and the rGather generator function allows
       def animateData(state, fig, ax):
           """ Render the trajectories rendered by the rGather algorithms
           and add fancy effects.
           """
           trajectories = state[0] # All trajectories
           currentTime  = state[1] # The time at which to animate
           ax.set_title( self.algoName + '\n r=' + str(self.r) + '  ' +  str('  Static Repeat  ') + str(currentTime) + '/' + str(len(self.pointCloud[0])-1) , 
                         fontdict={'fontsize':24})
           ax.set_xlabel('Latitude', fontdict={'fontsize':22})
           ax.set_ylabel('Longitude',fontdict={'fontsize':22})

           for line in ax.lines:
                line.set_markevery((currentTime, currentTime))

           print colored(str(currentTime) + ' ' + str(len(ax.lines)), 'cyan')
           return trajectories
       
       # Call the animator.  blit=True means only re-draw the parts that have changed.
       # Ensures better speed
       
       anim = animation.FuncAnimation(fig, animateData, rGather(),
                                      init_func=init, interval=interval_between_frames, blit=False, fargs=(fig,ax))
       # The draw commands are very important for the animation to be rednered.
       fig.canvas.draw()
       plt.show()
       anim.save('static_repeat.mp4', fps=5, extra_args=['-vcodec', 'libx264']) ; print "Animation saved"
 # for trajectories in the euclidean plane with the linifinity-like metric

def flattenNestedLists ( L, E):
      """ Make sure, that inside the rGather code, you always call this with E = [].
      I am NOT using a default value of E=[] inside because of soemthing Guido warned against
      mutable arguments being used as default arguments. My impression is, it tends to behave 
      quite like the static key-word in C inside functions. 
     
      Here L is the nested list, which looks like the example above.
      """ 
      for elt in L:
                print "Looking at", elt
                if isinstance(elt,list):
                        for eltt in elt:
                              if isinstance(eltt,list):
                                   flattenNestedLists(eltt,E)
                              else:
                                   E.append(elt)
                                   break
                else:
                        E.append(L)
                        break

# Types would have been invaluable here!!
def refineLargeClusters(pointCloud, largeClusters):
    pass
