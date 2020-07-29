from __future__ import print_function
import os
#set default parallel/serial flag
try:
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    use_parallel=num_proc > 1
except:
    myid = 0
    use_parallel=False

from numpy import sin
from numpy import cos
from numpy import tan
from petram.helper.variables import cosd
from petram.helper.variables import sind
from petram.helper.variables import tand
from numpy import arctan
from numpy import arctan2
from numpy import exp
from numpy import log10
from numpy import log
from numpy import log2
from numpy import sqrt
from numpy import abs
from numpy import conj
from numpy import real
from numpy import imag
from numpy import sum
from numpy import dot
from numpy import vdot
from numpy import array
from numpy import cross
from numpy import min
from numpy import sign

from petram.mfem_model import MFEM_ModelRoot
from petram.mfem_model import MFEM_GeneralRoot
from petram.mfem_model import MFEM_GeomRoot
from petram.geom.occ_geom_model import OCCGeom
from petram.geom.geom_primitives import CircleOCC
from petram.geom.geom_primitives import Rect
from petram.geom.geom_primitives import Difference
from petram.mfem_model import MFEM_MeshRoot
from petram.mesh.gmsh_mesh_model import GmshMesh
from petram.mesh.gmsh_mesh_actions import FreeFace
from petram.mesh.gmsh_mesh_actions import MergeText
from petram.mesh.mesh_model import MFEMMesh
from petram.mesh.mesh_model import MeshFile
from petram.mesh.mesh_model import UniformRefinement
from petram.mfem_model import MFEM_PhysRoot
from petram.phys.wf.wf_model import WF
from petram.phys.wf.wf_model import WF_DefDomain
from petram.phys.wf.wf_constraints import WF_WeakDomainBilinConstraint
from petram.phys.wf.wf_model import WF_DefBdry
from petram.phys.wf.wf_essential import WF_Essential
from petram.phys.wf.wf_model import WF_DefPoint
from petram.phys.wf.wf_model import WF_DefPair
from petram.phys.aux_variable import AUX_Variable
from petram.mfem_model import MFEM_InitRoot
from petram.mfem_model import MFEM_PostProcessRoot
from petram.mfem_model import MFEM_SolverRoot
from petram.solver.solver_model import SolveStep
from petram.solver.std_solver_model import StdSolver
from petram.solver.mumps_model import MUMPS
from collections import OrderedDict

def make_model():
    obj1 = MFEM_ModelRoot()
    obj1.root_path = '/tmp/piscope_shiraiwa/jiji.pid161339/.###ifigure_home_shiraiwa_fokker_planck_geom.pfz/proj/model1/mfem'
    obj1.model_path = '/home/shiraiwa'
    obj2 = obj1.add_node(name = "General", cls = MFEM_GeneralRoot)
    obj2.ns_name = "global"
    obj3 = obj1.add_node(name = "Geometry", cls = MFEM_GeomRoot)
    obj4 = obj3.add_node(name = "OCCSequence1", cls = OCCGeom)
    obj4.geom_timestamp = 'Mon Jul 27 10:40:47 2020'
    obj4.maxthreads = 2
    obj5 = obj4.add_node(name = "Circle1", cls = CircleOCC)
    obj5.center_x = '0'
    obj5.center_y = '0'
    obj5.center_z = '0'
    obj5.center_m = '[0, 0, 0]'
    obj5.ax1_x = '1'
    obj5.ax1_y = '0'
    obj5.ax1_z = '0'
    obj5.ax1_m = '[1, 0, 0]'
    obj5.ax2_x = '0'
    obj5.ax2_y = '1'
    obj5.ax2_z = '0'
    obj5.ax2_m = '[0, 1, 0]'
    obj5.radius = '1.0'
    obj5.radius_txt = '1.0'
    obj5.num_points = '4'
    obj5.num_points_txt = '4'
    obj6 = obj4.add_node(name = "Rect1", cls = Rect)
    obj6.corner_x = '-1'
    obj6.corner_x_txt = '-1'
    obj6.corner_y = '-1'
    obj6.corner_y_txt = '-1'
    obj6.corner_z = '0'
    obj6.corner_m = '[0, 0, 0]'
    obj6.edge1_x = '1'
    obj6.edge1_y = '0'
    obj6.edge1_z = '0'
    obj6.edge1_m = '[1, 0, 0]'
    obj6.edge2_x = '0'
    obj6.edge2_y = '1'
    obj6.edge2_z = '0'
    obj6.edge2_m = '[0, 1, 0]'
    obj7 = obj4.add_node(name = "Rect2", cls = Rect)
    obj7.corner_x = '0'
    obj7.corner_y = '-1'
    obj7.corner_y_txt = '-1'
    obj7.corner_z = '0'
    obj7.corner_m = '[0, 0, 0]'
    obj7.edge1_x = '1'
    obj7.edge1_y = '0'
    obj7.edge1_z = '0'
    obj7.edge1_m = '[1, 0, 0]'
    obj7.edge2_x = '0'
    obj7.edge2_y = '1'
    obj7.edge2_z = '0'
    obj7.edge2_m = '[0, 1, 0]'
    obj8 = obj4.add_node(name = "Difference1", cls = Difference)
    obj8.objplus = 'ps1'
    obj8.objplus_txt = 'ps1'
    obj8.objminus = 'rec1, rec2'
    obj8.objminus_txt = 'rec1, rec2'
    obj9 = obj1.add_node(name = "Mesh", cls = MFEM_MeshRoot)
    obj10 = obj9.add_node(name = "GmshMesh1", cls = GmshMesh)
    obj10.geom_group = 'OCCSequence1'
    obj10.geom_timestamp = 'Mon Jul 27 10:40:47 2020'
    obj11 = obj10.add_node(name = "FreeFace1", cls = FreeFace)
    obj11.clmin = '0.02'
    obj11.clmin_txt = '0.02'
    obj11.resolution = '5'
    obj11.resolution_txt = '5'
    obj11.alg_2d = 'FrrontalQuad'
    obj12 = obj10.add_node(name = "MergeText1", cls = MergeText)
    obj12.enabled = False
    obj13 = obj9.add_node(name = "MFEMMesh1", cls = MFEMMesh)
    obj14 = obj13.add_node(name = "MeshFile1", cls = MeshFile)
    obj14.path = '/home/shiraiwa/src/gFP/data/semi_circle5_quad.msh'
    obj14.refine = 1
    obj15 = obj13.add_node(name = "UniformRefinement1", cls = UniformRefinement)
    obj15.num_refine = '3'
    obj16 = obj1.add_node(name = "Phys", cls = MFEM_PhysRoot)
    obj17 = obj16.add_node(name = "WF1", cls = WF)
    obj17.sel_index = [1]
    obj17.order = 2
    obj18 = obj17.add_node(name = "Domain", cls = WF_DefDomain)
    obj19 = obj18.add_node(name = "diffsion", cls = WF_WeakDomainBilinConstraint)
    obj19.coeff_type = 'Matrix'
    obj19.integrator = 'DiffusionIntegrator'
    obj19.coeff_lambda = '=BF_term'
    obj19.coeff_lambda_txt = '=BF_term'
    obj19.sel_index = ['all']
    obj19.sel_index_txt = 'all'
    obj19.paired_var = ('WF1', 0)
    obj20 = obj18.add_node(name = "convection", cls = WF_WeakDomainBilinConstraint)
    obj20.coeff_type = 'Vector'
    obj20.integrator = 'ConvectionIntegrator'
    obj20.coeff_lambda = '=A_term'
    obj20.coeff_lambda_txt = '=A_term'
    obj20.sel_index = ['all']
    obj20.sel_index_txt = 'all'
    obj20.paired_var = ('WF1', 0)
    obj21 = obj18.add_node(name = "mass", cls = WF_WeakDomainBilinConstraint)
    obj21.coeff_type = 'Scalar'
    obj21.coeff_lambda = '=dA_term'
    obj21.coeff_lambda_txt = '=dA_term'
    obj21.sel_index = ['all']
    obj21.sel_index_txt = 'all'
    obj21.paired_var = ('WF1', 0)
    obj22 = obj17.add_node(name = "Boundary", cls = WF_DefBdry)
    obj22.sel_index = []
    obj23 = obj22.add_node(name = "Essential1", cls = WF_Essential)
    obj23.enabled = False
    obj23.sel_index = [3, 4]
    obj23.sel_index_txt = '3, 4'
    obj23.esse_value = '1'
    obj23.esse_value_txt = '1'
    obj24 = obj17.add_node(name = "Point", cls = WF_DefPoint)
    obj24.sel_index = []
    obj25 = obj17.add_node(name = "Pair", cls = WF_DefPair)
    obj26 = obj17.add_node(name = "Variable1", cls = AUX_Variable)
    obj26.variable_name = 'n_partilce'
    obj26.aux_connection = OrderedDict([(0, ('WF1', 0))])
    obj26.rhs_vec = '1'
    obj26.rhs_vec_txt = '1'
    obj26.oprt1_0 = '=integral()'
    obj26.oprt1_0_txt = '=integral()'
    obj26.oprt2_0 = '=integral()'
    obj26.oprt2_0_txt = '=integral()'
    obj27 = obj1.add_node(name = "InitialValue", cls = MFEM_InitRoot)
    obj28 = obj1.add_node(name = "PostProcess", cls = MFEM_PostProcessRoot)
    obj29 = obj1.add_node(name = "Solver", cls = MFEM_SolverRoot)
    obj30 = obj29.add_node(name = "SolveStep1", cls = SolveStep)
    obj31 = obj30.add_node(name = "StdSolver1", cls = StdSolver)
    obj31.phys_model = 'WF1'
    obj32 = obj31.add_node(name = "MUMPS1", cls = MUMPS)
    obj32.log_level = 1
    return obj1

if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description="PetraM sciprt")
    parser.add_argument("-s", "--force-serial", 
                     action = "store_true", 
                     default = False,
                     help="Use serial model even if nproc > 1.")
    parser.add_argument("-p", "--force-parallel", 
                     action = "store_true", 
                     default = False,
                     help="Use parallel model even if nproc = 1.")
    parser.add_argument("-d", "--debug-param", 
                     action = "store", 
                     default = 1, type=int) 
    args = parser.parse_args()
    if args.force_serial: use_parallel=False
    if args.force_parallel: use_parallel=True

    import  petram.mfem_config as mfem_config
    mfem_config.use_parallel = use_parallel
    if (myid == 0): parser.print_options(args)
    debug_level=args.debug_param

    import time, datetime
    stime = time.time()
    if mfem_config.use_parallel:
        from petram.engine import ParallelEngine as Eng
    else:
        from petram.engine import SerialEngine as Eng
    
    import petram.debug as debug
    debug.set_debug_level(debug_level)
    
    model = make_model()
    
    eng = Eng(model = model)
    
    solvers = eng.run_build_ns()
    
    is_first = True
    for s in solvers:
        s.run(eng, is_first=is_first)
        is_first=False
    
    if myid == 0:
        print("End Time " + 
              datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print("Total Elapsed Time: " + str(time.time()-stime) + "s")
        print("Petra-M Normal End")