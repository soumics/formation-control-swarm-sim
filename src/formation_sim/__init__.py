"""formation_sim -- multi-agent formation control simulator.

Modules
-------
dynamics        single/double integrator and unicycle agent models, RK4 step
graph_topology  adjacency builders, Laplacians, connectivity, agent removal
formations      formation shapes, virtual-leader trajectories, time-varying references
control_laws    consensus and leader-follower formation laws (1st and 2nd order)
avoidance       potential-field avoidance and a decentralised CBF safety filter
simulate        closed-loop simulator with dropouts
metrics         formation/tracking error, settling time, safety distances
scenarios       ready-made demonstration scenarios
visualize       trajectory/error plots and GIF animation
"""

__version__ = "0.1.0"
