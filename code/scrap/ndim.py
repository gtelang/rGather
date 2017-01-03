import numpy as np

traj1 = np.rec.array([(1.1, 2.1), (2.3, 3.6), (7.8,9.2), (5.6, 8.9)], dtype=[('x', 'float64'),('y', 'float64')])
traj2 = np.rec.array([(4.5, 5.2), (2.3, 3.6), (7.8,9.2), (5.6, 8.9)], dtype=[('x', 'float64'),('y', 'float64')])

trajlist = np.array( [traj1, traj2] )
