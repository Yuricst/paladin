"""
UDP for transitioning to full ephemeris model
"""

import numpy as np
import pygmo as pg
from scipy.linalg import block_diag
import copy


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
        if self.propagator.use_canonical:
            et0 = self.et0_ref + delta_et0*self.propagator.tstar
        else:
            et0 = self.et0_ref + delta_et0

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
        
    def gradient_custom(self, x, dx=1e-6, return_list=True):
        """Custom gradient computation
        
        Args:
            x (list): decision variables
            dx (float): step size for finite difference
            return_list (bool): whether to return a list or a numpy array

        Returns:
            (list): gradient of the objective function
        """
        # number of constraints and decision variables
        nX = len(x)
        nF = 1 + self.get_nec() + self.get_nic()

        # unpack decision variables
        et_nodes, tofs, nodes = self.unpack_x(x)
        
        # compute sensitivity w.r.t. nodes via stms
        Bs_diag, Bs_upperdiag = [], []

        # for-loop: for each leg
        n_legs = len(tofs)
        for idx in range(n_legs):
            idx_fwd_seg = idx
            idx_back_seg = idx + 1
            
            # forward segment
            svf0, stm0, _ = self.propagator.get_stm_cdm(et_nodes[idx_fwd_seg], tofs[idx]/2, nodes[idx_fwd_seg], get_svf=True)
            
            # backward segment
            svf1, stm1, _ = self.propagator.get_stm_cdm(et_nodes[idx_back_seg], -tofs[idx]/2, nodes[idx_back_seg], get_svf=True)
            
            # construct elements of Jacobian via STM
            Bs_diag.append(-stm0)
            Bs_upperdiag.append(stm1)

        # concatenate
        B_diag = np.concatenate((block_diag(*Bs_diag), np.zeros((6*n_legs,6))), axis=1)
        B_offdiag = np.concatenate((np.zeros((6*n_legs,6)), block_diag(*Bs_upperdiag)), axis=1)
        B = B_diag + B_offdiag
        B = np.concatenate((np.zeros((1,nX-1-len(tofs))),B))

        # compute sensitivity w.r.t. et0 & tofs via central differencing
        # w.r.t. et0
        xtest_ptrb_fwd = copy.copy(x)
        xtest_ptrb_fwd[0] += dx

        xtest_ptrb_bck = copy.copy(x)
        xtest_ptrb_bck[0] -= dx

        A = (np.array(self.fitness(xtest_ptrb_fwd)) - np.array(self.fitness(xtest_ptrb_bck)))/(2*dx)
        A = A.reshape(-1,1)

        # w.r.t. TOF
        C = np.zeros((nF,len(tofs)))
        for idx in range(len(tofs)):
            xtest_ptrb_fwd = copy.copy(x)
            xtest_ptrb_fwd[1+6*len(nodes)+idx] += dx

            xtest_ptrb_bck = copy.copy(x)
            xtest_ptrb_bck[1+6*len(nodes)+idx] -= dx
            
            C[:,idx] = (np.array(self.fitness(xtest_ptrb_fwd)) - np.array(self.fitness(xtest_ptrb_bck)))/(2*dx)
        
        # concatenate A, B, C
        grad_custom = np.concatenate((A,B,C), axis=1)
        if return_list:
            return list(grad_custom.reshape(np.size(grad_custom),))
        else:
            return grad_custom
        

    def gradient(self, x, dx=1e-6, use_h=False):
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
