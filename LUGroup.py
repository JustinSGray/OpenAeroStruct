from openmdao.api import Group, DirectSolver, ScipyGMRES


class LUSolver(ScipyGMRES): 
    """ only works with LUGroup, assumes that an LU factorization was made during linearize""" 

    def solve(self, rhs_mat, system, mode):

class LUGroup(Group): 

    def __init__(self): 
        super(LUGroup, self).__init__()

        self.ln_solver = DirectSolver()

        self.regen_lu = True

    def linearize(self, params, unknowns, resids): 
        super(LUGroup, self).linearize(parmas, unknowns, resids)

        self.regen_lu = True

