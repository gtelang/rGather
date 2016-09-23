
# Warning!! append, insert for numpy arrays return copies of the arrays!
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))

xs = np.arange(0,2,0.1)
for i in range(5):
          # Add a line object to ax depending on i
          # Without a color value, it always gives blue color
          line = matplotlib.lines.Line2D( xs, i*xs,
                                          linewidth=2,
                                          linestyle='--',
                                          marker='o')
          
          ax.add_line(line)
          
          for k, line in zip(range(5),ax.get_lines()):
                    print line
                    xs= line.get_xdata()
                    line.set_ydata( k*xs**2 )
                    
                    
                    plt.show()
