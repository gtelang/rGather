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
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



def main( argv ):

    print "Started Program"
    with open(  './4ApproxStaticOutput/cars1500_riksuggestion.yaml' , 'r') as stream:
        try:
            D4_riksuggestion = yaml.load( stream )
        except yaml.YAMLError as exc:
            print(exc)

    pointCloud = D4_riksuggestion[ 'coordinates' ]
    rArray     = sorted( D4_riksuggestion['r'].keys() )

    r = rArray[6]
    fig, ax = plt.subplots(   )


    runJie = rg.AlgoJieminDecentralizedStatic( r = r, pointCloud = pointCloud )
    runJie.computedClusterings = D4_riksuggestion['r'][ r ]
    runJie.plotClusters(ax, pointSize=5, annotatePoints=False)
    #plt.savefig('plots/visualComparison-r' + str(r) + '.pdf')
    plt.show()








if __name__=="__main__":
    main( sys.argv )

