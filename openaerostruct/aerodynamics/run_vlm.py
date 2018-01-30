from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, pyOptSparseDriver, view_model, Group, ExecComp, SqliteRecorder

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct.aerodynamics.vlm_states1_group import VLMStates1Group
from openaerostruct.aerodynamics.vlm_states2_group import VLMStates2Group
from openaerostruct.aerodynamics.vlm_states3_group import VLMStates3Group
from openaerostruct.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct.utils.plot_utils import plot_mesh_2d, scatter_2d, arrow_2d


mode = 1

check_derivs = mode == 0

num_nodes = 1 if not check_derivs else 2

num_points_x = 3
num_points_z_half = 30 if not check_derivs else 2
num_points_z = 2 * num_points_z_half - 1
lifting_surfaces = [
    ('wing', {
        'num_points_x': num_points_x, 'num_points_z_half': num_points_z_half,
        'airfoil_x': np.linspace(0., 1., num_points_x),
        'airfoil_y': np.zeros(num_points_x),
        'mac': 0.7,
        'chord': 1.,
        'chord_bspline': (2, 2),
        'twist': 0. * np.pi / 180.,
        'twist_bspline': (11, 3),
        'sweep_x': 0.,
        'dihedral_y': 0.,
        'span': 5,
        'sec_z_bspline': (num_points_z_half, 2),
        'thickness_bspline': (10, 3),
        'thickness' : .1,
        'radius' : 1.,
        'distribution': 'sine',
        'section_origin': 0.25,
        'spar_location': 0.35,
    })
]

vlm_scaler = 1e0

prob = Problem()
prob.model = Group()

indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
indep_var_comp.add_output('Re_1e6', shape=num_nodes, val=2.)
indep_var_comp.add_output('C_l_max', shape=num_nodes, val=1.5)
prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

prob.model.add_subsystem('vlm_preprocess_group',
    VLMPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('vlm_states1_group',
    VLMStates1Group(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('vlm_states2_group',
    VLMStates2Group(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler),
    promotes=['*'],
)
prob.model.add_subsystem('vlm_states3_group',
    VLMStates3Group(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('vlm_postprocess_group',
    VLMPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('objective',
    ExecComp('obj=sum(C_D)', C_D=np.zeros(num_nodes)),
    promotes=['*'],
)

prob.model.add_design_var('alpha_rad', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
prob.model.add_design_var('wing_twist_dv', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
prob.model.add_objective('obj')
prob.model.add_constraint('C_L', equals=np.linspace(0.4, 0.6, num_nodes))

prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.opt_settings['Major optimality tolerance'] = 3e-7
prob.driver.opt_settings['Major feasibility tolerance'] = 3e-7

if mode == 2:
    prob.driver.add_recorder(SqliteRecorder('aero.hst'))
    prob.driver.recording_options['includes'] = ['*']

prob.setup()

prob['wing_chord_dv'] = [0.5, 1.0, 0.5]

if mode == -1:
    view_model(prob)
    exit()

elif mode == 0:
    prob.setup(force_alloc_complex=True)
    prob['wing_chord_dv'] = [0.5, 1.0, 0.5]
    prob.run_model()
    prob.check_partials(compact_print=True)
    exit()
elif mode == 2:
    prob.run_driver()
elif mode == 1:
    print('alpha', prob['alpha_rad'])
    print(prob['C_L'])

    num = 50
    x = np.zeros(num)
    y = np.zeros(num)
    for i, alpha in enumerate(np.linspace(-3, 20, num)):
        prob['alpha_rad'] = alpha * np.pi / 180.
        prob.run_model()
        x[i] = alpha
        y[i] = prob['C_L']
        # print(np.max(prob['wing_sec_C_L']))
        # print(np.max(prob['wing_sec_C_L_capped']))
        # print(np.min(prob['sec_C_L_factor']))
        print(np.max(prob['panel_forces_rotated_capped'][0, :, 1]))
    plt.plot(x, y)
    plt.show()

if 0:
    for i in range(num_nodes):
        C_L = prob['wing_sec_C_L'].reshape((num_nodes, num_points_z - 1))[i, :] \
            * 0.5 * (prob['wing_chord'][i, 1:] + prob['wing_chord'][i, :-1])
        sec_z = 0.5 * (prob['wing_sec_z'][i, 1:] + prob['wing_sec_z'][i, :-1])
        elliptical = C_L[num_points_z_half - 1] * np.sqrt(np.abs(1 - (sec_z / sec_z[-1]) ** 2))
        plt.subplot(num_nodes + 1, 2, 2 * i + 1)
        plt.plot(prob['wing_sec_z'][i, :], prob['wing_twist'][i, :] + prob['alpha_rad'][i], 'ko-')
        plt.subplot(num_nodes + 1, 2, 2 * i + 2)
        plt.plot(sec_z, C_L, 'bo-')
        plt.plot(sec_z, elliptical, 'ro-')
    plt.subplot(num_nodes + 1, 2, 2 * num_nodes + 1)
    plt.plot(prob['wing_sec_z'][i, :], prob['wing_twist'][i, :], 'ko-')
    plt.show()

if 0:
    mesh = prob['wing_mesh'][0]
    vortex_mesh = prob['wing_vortex_mesh'][0]
    collocation_points = prob['coll_pts'][0]
    force_points = prob['force_pts'][0]
    bound_vecs = prob['bound_vecs'][0]
    wing_normals = prob['wing_normals'][0].reshape(
        (int(np.prod(prob['wing_normals'][0].shape[:2])), 3))

    fig = plt.figure()

    # plt.subplot(2, 1, 1)
    ax = fig.gca()
    plot_mesh_2d(ax, vortex_mesh, 2, 0, color='b')
    plot_mesh_2d(ax, mesh, 2, 0, color='k')
    scatter_2d(ax, collocation_points, 2, 0, color='b', markersize=3)
    scatter_2d(ax, force_points, 2, 0, color='r', markersize=3)
    arrow_2d(ax, force_points, 0.5 * bound_vecs, 2, 0, color='grey')
    plt.axis('equal')
    plt.xlabel('z')
    plt.ylabel('x')

    # print(prob['horseshoe_circulations'][0].reshape((
    #     num_points_x - 1, 2 * num_points_z_half - 2 )))

    plt.show()