from __future__ import print_function, division, absolute_import

import unittest

import itertools
from six import iteritems

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver, view_model, ExecComp, SqliteRecorder
from openmdao.utils.assert_utils import assert_check_partials

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.structures.fea_bspline_group import FEABsplineGroup

from openaerostruct.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct.structures.fea_preprocess_group import FEAPreprocessGroup
from openaerostruct.structures.fea_postprocess_group import FEAPostprocessGroup

from openaerostruct.aerostruct.aerostruct_group import AerostructGroup
from openaerostruct.aerostruct.aerostruct_postprocess_group import AerostructPostprocessGroup
from openaerostruct.tests.utils import get_default_lifting_surfaces
from openaerostruct.common.lifting_surface import LiftingSurface


num_nodes = 1
num_points_x = 2
num_points_z_half = 2
num_points_z = 2 * num_points_z_half - 1
g = 9.81

wing = LiftingSurface('wing')

wing.initialize_mesh(num_points_x, num_points_z_half, airfoil_x=np.linspace(0., 1., num_points_x), airfoil_y=np.zeros(num_points_x))
wing.set_mesh_parameters(distribution='sine', section_origin=.25)
wing.set_structural_properties(E=70.e9, G=29.e9, spar_location=0.35, sigma_y=200e6, rho=2700)
wing.set_aero_properties(factor2=.119, factor4=-0.064, cl_factor=1.05)

wing.set_chord(1.)
wing.set_twist(0.)
wing.set_sweep(0.)
wing.set_dihedral(0.)
wing.set_span(5.)
wing.set_thickness(0.05)
wing.set_radius(0.1)

lifting_surfaces = [('wing', wing)]

vlm_scaler = 1e2
fea_scaler = 1e6

prob = Problem()
prob.model = Group()

indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
indep_var_comp.add_output('mu_Pa_s', shape=num_nodes, val=14.36e-6)
indep_var_comp.add_output('reference_area', shape=num_nodes, val=7.5)

prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

group = FEABsplineGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
prob.model.add_subsystem('tube_bspline_group', group, promotes=['*'])

prob.model.add_subsystem('vlm_preprocess_group',
    VLMPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('fea_preprocess_group',
    FEAPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler),
    promotes=['*'],
)

prob.model.add_subsystem('aerostruct_group',
    AerostructGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler, fea_scaler=fea_scaler),
    promotes=['*'],
)

prob.model.add_subsystem('vlm_postprocess_group',
    VLMPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('fea_postprocess_group',
    FEAPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('as_postprocess_group',
    AerostructPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)

prob.model.add_subsystem('objective',
    ExecComp('obj=10000 * sum(C_D) + structural_weight', C_D=np.zeros(num_nodes)),
    promotes=['*'],
)

prob.model.add_design_var('alpha_rad', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
prob.model.add_design_var('wing_twist_dv', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
prob.model.add_design_var('wing_tube_thickness_dv', lower=0.001, upper=0.19, scaler=1e3)
prob.model.add_objective('obj')
prob.model.add_constraint('wing_ks', upper=0.)
prob.model.add_constraint('C_L', equals=np.linspace(0.8, 0.8, num_nodes))

prob.driver = ScipyOptimizeDriver()
prob.driver.opt_settings['Major optimality tolerance'] = 1e-9
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-9

prob.driver.add_recorder(SqliteRecorder('aerostruct.hst'))
prob.driver.recording_options['includes'] = ['*']

prob.setup(force_alloc_complex=True)

prob['wing_chord_dv'] = [0.5, 1.0, 0.5]

prob.run_model()