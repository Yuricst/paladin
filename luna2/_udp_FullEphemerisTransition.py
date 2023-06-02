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
        propagator,
        et0,
        nodes,
        tofs,
        delta_et0_bounds=None,
        nodes_bounds=None,
        node0_bounds=None,
        nodef_bounds=None,
        tofs_bounds=None,
    ):
        """UDP constructor
        
        Args:
            propagator (Propagator): propagator in the dynamics model trajectory is to be constructed
            et0 (float): initial epoch in seconds past J2000
            nodes (list): list of patch nodes in the trajectory
        """
        assert len(nodes) == len(tofs) + 1, "Must provide 1 extra nodes than tofs"
        if nodes_bounds is not None:
            assert len(nodes) == len(nodes_bounds), "nodes and nodes_bounds must be the same length"
        # store inputs
        self.nodes = nodes
        self.N = len(nodes)

        # whether to fix et0
        self.et0_ref = et0  # reference epoch
        if delta_et0_bounds is None:
            self.delta_et0_bounds = [0.0, 0.0]
        else:
            self.delta_et0_bounds = delta_et0_bounds

        # whether all nodes' bounds are provided
        if nodes_bounds is None:
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
            # create bounds on all nodes
            self.nodes_bounds = self.node0_bounds + [] + self.nodef_bounds  # FIXME
            raise NotImplementedError("Need to implement automatic nodes_bounds")
        else:
            self.nodes_bounds = nodes_bounds
            self.fixed_node0 = nodes_bounds[0][0] == nodes_bounds[0][1]
            self.fixed_nodef = nodes_bounds[-1][0] == nodes_bounds[-1][1]
            
        # whether to fix tofs
        if tofs_bounds is None:
            self.tofs_bounds = [[tof,tof] for tof in tofs]  # fix tofs
        else:
            self.tofs_bounds = tofs_bounds

        # store propagator
        self.propagator = propagator
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
        lb = [self.delta_et0_bounds[0]] #+ list(self.node0_bounds[0])
        ub = [self.delta_et0_bounds[1]] #+ list(self.node0_bounds[1])
        for idx in range(self.N):
            lb += list(self.nodes_bounds[idx][0])
            ub += list(self.nodes_bounds[idx][1])
        for idx in range(self.N-1):
            lb += [self.tofs_bounds[idx][0]]
            ub += [self.tofs_bounds[idx][1]]
        return (lb, ub)
    
    def unpack_x(self, x):
        """Unpack decision vector into `et_nodes`, `tofs`, `nodes`"""
        # unpack decision variables
        delta_et0, tofs = x[0], x[6*self.N+1:]
        _nodes = x[1:6*self.N+1]
        et0 = self.et0_ref + delta_et0*self.propagator.tstar

        nodes = []
        for i in range(self.N):
            nodes.append(_nodes[i * 6: (i + 1) * 6])
    
        # list of epochs
        if self.propagator.use_canonical:
            et_nodes = [et0] + [et0 + sum(tofs[:i+1])*self.propagator.tstar for i in range(len(tofs))]
        else:
            et_nodes = [et0] + [et0 + sum(tofs[:i+1]) for i in range(len(tofs))]
        return et_nodes, tofs, nodes
    
    def fitness(self, x, get_sols=False, verbose=False):
        """Compute fitness of decision variables"""
        # unpack decision variables
        et_nodes, tofs, nodes = self.unpack_x(x)

        # propagate first node forward only
        sol_fwd_list = [self.propagator.solve(et_nodes[0], (0,tofs[0]/2), nodes[0]),]
        sol_bck_list = []

        # propagate intermediate nodes forward and backward
        for idx in range(self.N-2):
            if verbose:
                print(f"Integrating segment {idx}")
            sol_bck_list.append(
                self.propagator.solve(et_nodes[idx+1], (0,-tofs[idx]/2), nodes[idx+1])
            )
            sol_fwd_list.append(
                self.propagator.solve(et_nodes[idx+1], (0,tofs[idx+1]/2), nodes[idx+1])
            )

        # propagate final node backward only
        sol_bck_list.append(
            self.propagator.solve(et_nodes[self.N-1], (0,-tofs[self.N-2]/2), nodes[self.N-1])
        )

        # compute residuals for each interval
        ceqs = []
        for idx in range(self.N-1):
            ceqs += list(sol_bck_list[idx].y[:,-1] - sol_fwd_list[idx].y[:,-1])
            if verbose:
                print(f"Appending inequality {idx}: {list(sol_bck_list[idx].y[:,-1] - sol_fwd_list[idx].y[:,-1])}")
        
        # in order: objective, equality constraints, inequality constraints
        if get_sols is False:
            return [1.0,] + ceqs
        else:
            return [1.0,] + ceqs, sol_fwd_list, sol_bck_list
        
    def gradient_custom(self, x):
        """Custom gradient computation"""
        # unpack decision variables
        et_nodes, tofs, nodes = self.unpack_x(x)
        return

    def gradient(self, x, dx=1e-8, use_h=False):
        """Compute gradient of decision variables
        Ref:
        * https://esa.github.io/pygmo2/gh_utils.html#pygmo.estimate_gradient
        
        Args:
            x (list): decision variables
            dx (float): step size for finite difference
            use_h (bool): whether to use higher order finite difference

        Returns:
            list: gradient of decision variables
        """
        if use_h:
            return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
        else:
            return pg.estimate_gradient(lambda x: self.fitness(x), x, dx=dx)
