
# Warning!! append, insert for numpy arrays return copies of the arrays!
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
fig = plt.figure()
ax = plt.axes()

xs = np.arange(0,2,0.1)
line = matplotlib.lines.Line2D( xs, 4*xs,
                                linewidth=2,
                                linestyle='--',
                                marker='o')

ax.add_line(line)



xt = np.arange(0,3,0.1)
line = ax.get_lines()[0]

line.set_xdata(xt)
xl   = line.get_xdata()
print (id(xt) == id(xl)) # Are these the same? Yes!!!! Looks like!1 Getting and setting is free!!!!!
                         # Another reason why you might want to do this!
print id(xt), " " , id(xl)
#plt.show()
