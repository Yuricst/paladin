"""
UDP for transitioning to full ephemeris model
"""

import numpy as np
import pygmo as pg


class FullEphemerisTransition:
    """UDP as problem to transition to full ephemeris model
    Chromosome encoding:
        x = [et0, node_0, ..., node_{N-1}, tof_0, ..., tof_{N-1}]
    """
    def __init__(
        self,
        et0,
        nodes,
        tofs,
        et0_bounds=None,
        node0_bounds=None,
        nodef_bounds=None,
        tofs_bounds=None,
    ):
        """UDP constructor"""
        assert len(nodes) == len(tofs) + 1, "Must provide 1 extra nodes than tofs"
        # store inputs
        self.nodes = nodes
        self.N = len(nodes)
        # whether to fix et0 
        if et0_bounds is None:
            self.et0_bounds = [et0, et0]
        else:
            self.et0_bounds = et0_bounds
        # whether to fix node0_bounds
        if node0_bounds is None:
            self.node0_bounds = [nodes[0], nodes[0]]  # fix to values at initial nodes
            self.fixed_node0 = True
        else:
            self.node0_bounds = node0_bounds
            self.fixed_node0 = False
        # whether to fix nodef_bounds
        if nodef_bounds is None:
            self.nodef_bounds = [nodes[-1], nodes[-1]]  # fix to values at final nodes
            self.fixed_nodef = True
        else:
            self.nodef_bounds = nodef_bounds
            self.fixed_nodef = False
        # whether to fix tofs
        if tofs_bounds is None:
            self.tofs_bounds = [tofs, tofs]   # fix tofs
        else:
            self.tofs_bounds = tofs_bounds
        return

    def get_nec(self):
        """Number of equality constraints
        nec = 6*(len(nodes) - 1)
        """
        return 6*(self.N - 1)

    def get_nic(self):
        """Number of inequality constraints"""
        return 0
    
    def get_bounds(self):
        """Construct bounds on the decision variables"""
        lb = [self.et0_bounds[0]] + list(self.node0_bounds[0])
        ub = [self.et0_bounds[1]] + list(self.node0_bounds[1])
        for idx in range(self.N-1):
            lb += []
            ub += []
        lb += list(self.nodef_bounds[0]) + [self.tofs_bounds[0]]
        ub += list(self.nodef_bounds[1]) + [self.tofs_bounds[1]]
        return (lb, ub)
    
    # def fitness(self, x):
    #     # compute fitness 
    #     # in order: objective, equality constraints, inequality constraints
    #     return [obj, ceqs, cineqs]

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)