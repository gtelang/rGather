import matplotlib.pyplot as plt
import numpy as np
import colorsys

def getRandomColor():
    """ A cute function for generating colors better than 
    matplotlib's defaults. Ripped from http://goo.gl/SMlEaU
    """
    golden_ratio_conjugate = 0.618033988749895
    h = np.random.rand()
    h += golden_ratio_conjugate
    h %= 1.0
    return colorsys.hsv_to_rgb(h, 0.7, 0.9)




def makeplot(ax):
    x = np.linspace(0,2*np.pi)
    y = 2*np.sin(x) + np.random.rand(len(x))
    ax.plot(x, y, color=getRandomColor()  )

fig, ax = plt.subplots(2,2)
makeplot(ax[0][0])
makeplot(ax[0][1])

import networkx as nx
import matplotlib.pyplot as plt
g1 = nx.petersen_graph()
nx.draw(g1, ax=ax[1][0])


fig2, ax2 = plt.subplots()
import matplotlib.pyplot as plt
import networkx as nx
G=nx.dodecahedral_graph()
nx.draw(G)  # networkx draw()







plt.show()  # pyplot draw()
