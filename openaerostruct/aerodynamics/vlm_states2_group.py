from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.common.linear_comb_comp import LinearCombComp

from openaerostruct.aerodynamics.components.velocities.vlm_eval_vectors_comp import VLMEvalVectorsComp
from openaerostruct.aerodynamics.components.velocities.vlm_eval_vel_mtx_comp import VLMEvalVelMtxComp
from openaerostruct.aerodynamics.components.velocities.vlm_eval_velocities_comp import VLMEvalVelocitiesComp

from openaerostruct.aerodynamics.components.circulations.vlm_mtx_rhs_comp import VLMMtxRHSComp
from openaerostruct.aerodynamics.components.circulations.vlm_circulations_comp import VLMCirculationsComp
from openaerostruct.aerodynamics.components.circulations.vlm_horseshoe_circulations_comp import VLMHorseshoeCirculationsComp


class VLMStates2Group(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('vlm_scaler', types=float)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        vlm_scaler = self.metadata['vlm_scaler']

        num_collocation_points = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            num_collocation_points += (num_points_x - 1) * (num_points_z - 1)

        num_force_points = num_collocation_points

        in_names = ['freestream_vel', 'vlm_ext_velocities']
        out_name = 'inflow_velocities'
        comp = LinearCombComp(shape=(num_nodes, num_force_points, 3), in_names=in_names, out_name=out_name)
        self.add_subsystem('vlm_inflow_velocities_comp', comp, promotes=['*'])

        comp = VLMEvalVectorsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='coll_pts', num_eval_points=num_collocation_points)
        self.add_subsystem('vlm_collocation_eval_vectors_comp', comp, promotes=['*'])

        comp = VLMEvalVelMtxComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='coll_pts', num_eval_points=num_collocation_points)
        self.add_subsystem('vlm_collocation_eval_vel_mtx_comp', comp, promotes=['*'])

        comp = VLMMtxRHSComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler)
        self.add_subsystem('vlm_mtx_rhs_comp', comp, promotes=['*'])

        comp = VLMCirculationsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_circulations_comp', comp, promotes=['*'])

        comp = VLMHorseshoeCirculationsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_horseshoe_circulations_comp', comp, promotes=['*'])

        comp = VLMEvalVectorsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='force_pts', num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_vectors_comp', comp, promotes=['*'])

        comp = VLMEvalVelMtxComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='force_pts', num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_vel_mtx_comp', comp, promotes=['*'])

        comp = VLMEvalVelocitiesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='force_pts', num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_velocities_comp', comp, promotes=['*'])