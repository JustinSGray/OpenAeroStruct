from collections import OrderedDict

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.api import Group, ScipyGMRES

class LUSolver(ScipyGMRES): 
    """ only works with LUGroup, assumes that an LU factorization was made during linearize""" 

    def solve(self, rhs_mat, system, mode):

        sol_buf = OrderedDict()
        self.lup = dict()

        # TODO: This solver could probably work with multiple RHS
        for voi, rhs in rhs_mat.items():
            self.voi = None

            # TODO: When to record?
            self.system = system
            self.mode = mode

            n_edge = len(rhs)
            if system.regen_lu: 
                ident = np.eye(n_edge)

                partials = np.empty((n_edge, n_edge))

                for i in range(n_edge):
                    partials[:, i] = self.mult(ident[:, i])

                self.lup[voi] = lu_factor(partials)

            # don't need to explicitly handle transpose, because mode takes care of it in mult            
            sol_buf[voi] = lu_solve(self.lup[voi], rhs) 

        system.regen_lu = False
        return sol_buf


class LUGroup(Group): 

    def __init__(self): 
        super(LUGroup, self).__init__()

        self.ln_solver = LUSolver()

    def linearize(self, params, unknowns, resids): 
        super(LUGroup, self).linearize(params, unknowns, resids)

        self.regen_lu = True



