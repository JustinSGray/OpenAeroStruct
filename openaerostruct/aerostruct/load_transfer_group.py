from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.aerostruct.components.as_load_transfer_comp import ASLoadTransferComp


class LoadTransferGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_subsystem('as_load_transfer_comp',
            ASLoadTransferComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )