# This contains all the needed tricks to customize matplotlib plots.
import matplotlib.pyplot as plt
import numpy as np
import rGather as rg

fig_a, ax_a = plt.subplots(1,2)
fig_b, ax_b = plt.subplots()


x0 = np.linspace(-1,1,1000)
y0 = x0**2

x1 = np.linspace(-1,1,1000)
y1 = np.sin(x1)


x2 = np.linspace(-1,1,1000)
y2 = np.cos(x1)

##################################
ax_a[0].plot(x0,y0)
ax_a[0].set_title('Quadratic Curve')

ax_a[1].plot(x1,y1)
ax_a[1].set_title('Sine Curve')

ax_b.plot(x2,y2)
ax_b.set_title('Cosine Curve')

# The file format is automatically detected from the string name.
# You will use .eps and  .svg for high-quality plots. You can convert
# .svg to .eps files if you need with inkscape the command-line way to do this
# is `inkscape scrap.svg --export-eps=scrap.eps` the output file is your needed .eps file
fig_a.savefig('Figure_a.svg') # Note that figures can be saved independently. 
fig_b.savefig('Figure_b.svg')

# ax_b.savefig('ax_b.svg') # Unfortunately, ax object cannot be stored this way.
# but see this for a clever alternative. http://stackoverflow.com/a/4328608/505306
# plt.show()
