import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Important web-links: https://networkx.github.io/documentation/networkx-1.9.1/tutorial/tutorial.html
# http://networkx.github.io/documentation/networkx-1.8.1/reference/generated/networkx.algorithms.flow.min_cost_flow_cost.html#networkx.algorithms.flow.min_cost_flow_cost



# G = nx.DiGraph()

# G.add_nodes_from(['s','t', 1, 2, 3])

# G.add_edges_from( [ ('s',1), ('s',2), ('s',3),
#                     (1,'t'), (2,'t'), (3,'t')]  )

# # Give edge costs and capcacities to edges
# G.edge['s'][1]['capacity'] = 4
# G.edge['s'][2]['capacity'] = 3
# G.edge['s'][3]['capacity'] = 3
# G.edge[1]['t']['capacity'] = 3.5
# G.edge[2]['t']['capacity'] = 3.5
# G.edge[3]['t']['capacity'] = 3.5


# G.node[ 1 ]['demand'] = 0
# G.node[ 2 ]['demand'] = 0
# G.node[ 3 ]['demand'] = 0
# G.node['t']['demand'] = -3
# G.node['s']['demand'] = 3

# print G.node[1]['demand']

# # Run the flow algorithm and get the smallest cost
# # thus calculated.
# flowCost = nx.min_cost_flow_cost(G)
# print "The network is unfeasible. fuck you "
# print "The flow cost is ", flowCost

# nx.draw(G, with_labels=True)
# plt.show()



import networkx as nx
G = nx.DiGraph()
G.add_node('a', demand = -2)
G.add_node('d', demand =  2)
G.add_edge('a', 'b',  capacity = 1)
G.add_edge('a', 'c',  capacity = 1)
G.add_edge('b', 'd',  capacity = 1)
G.add_edge('c', 'd',  capacity = 1.99)

# Magic edge
G.add_edge('a','d',capacity=0.001)

#G.node['a']['center'] = 'Bulla ki Jaana main kaun' 
#print G.node['a']['center']
#print G.node['b']['center']
#print G.edge['c']['d']['capacity'] # This works!

isClusterCenterDict = {'a':True, 'b':False, 'c': True, 'd':'False'}
nx.set_node_attributes(G,'isCenter', isClusterCenterDict)

print G.node['a']['isCenter']

print G.node['b']['isCenter']

print G.node['c']['isCenter']

print G.node['d']['isCenter']

# flowDict[u][v] is the flow sent along the edge u,v
# flowdict is a dictionary of key-value pairs, where
# the values themselves are dictionaries.

try:
    flowDict = nx.min_cost_flow(G)
except nx.NetworkXUnfeasible:
    print "Sorry doll, unfeasible network"
else:
    print flowDict

print "Any rasied exceptions thus fae were handled successfully!Stay at ease!"
