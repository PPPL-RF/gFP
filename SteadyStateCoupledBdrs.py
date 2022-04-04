import numpy as np
import numba
from numba import jit, float64, farray, vectorize, cfunc, void, carray
from numpy import sin, cos, sqrt, pi, arctan2, exp
# from scipy.special import erf
from math import erf

import mfem.ser as mfem
from mfem.common.sparse_utils import sparsemat_to_scipycsr
from glvis import glvis

from scipy.sparse import lil_matrix, csr_matrix, coo_matrix

import matplotlib.pyplot as plt


CoulLog = 15.
Zb = 1.
Za = 1.
nb = 1.0e20
vb = 8.8e6
va=vb*3.0
ma = 9.1093837015e-31
mb = 9.1093837015e-31
ee = 1.602176634e-19
em = 2.718281828459045
gamma = 4.0*pi*pow(Za,4.0)*pow(ee,4.0)/ma/ma

@jit(float64(float64, float64))
def pow(x, c):
    return x**c


@jit(void(float64, float64, float64[:]))
def v_theta(x, y, E):
    y = y if y > 0 else 0.0

    theta = arctan2(y, x)
    vel = sqrt(x**2 + y**2)
    vel = vel if vel != 0 else 1e-12
    E[0] = x*va
    E[1] = y*va
    E[2] = vel*va
    E[3] = theta

@jit(float64(float64))
def calc_A(vel):
    A = (((Zb/Za)**2)*CoulLog*ma/mb*nb*
         (-vel/vb*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))+erf(vel/sqrt(2.0)/vb)))
    return A

@jit(float64(float64))
def calc_dA(vel):
    dA = pow(Zb/Za,2.0)*CoulLog*ma/mb*nb*pow(vel,2.0)/pow(vb,3.0)*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))
    return dA

@mfem.jit.scalar()
def dA_func(p):
    E = np.empty(4, dtype = float64)
    v_theta(p[0], p[1], E)
    x = E[0]
    y = E[1]
    vel = E[2]
    theta = E[3]
    dA = calc_dA(vel)
    base = dA/vel*sin(theta)
    return gamma*base

# class dA_term(mfem.PyCoefficient):
#     def EvalValue(self, v):
#         return dA_func(v[0], v[1])

@mfem.jit.vector()
def A_func(p, out):
    out_array = carray(out, (2, ))
    E = np.empty(4, dtype = float64)
    v_theta(p[0], p[1], E)
    x = E[0]
    y = E[1]
    vel = E[2]
    theta = E[3]
    
    A = calc_A(vel)
    x = A*sin(theta)*cos(theta)/vel
    y = A*sin(theta)*sin(theta)/vel
    vec = np.array([x, y])
    out_array[0] = x/va*gamma
    out_array[1] = y/va*gamma

# class A_term(mfem.VectorPyCoefficient):
#     def EvalValue(self,v):
#         return A_func(v[0], v[1])

@mfem.jit.matrix(sdim = 2)
def BF_func(ptx, out):    
    E = np.empty(4, dtype = float64)
    v_theta(ptx[0], ptx[1], E)
    x = E[0]
    y = E[1]
    vel = E[2]
    theta = E[3]
    
    mat = carray(out, (2,2))
    
    A = calc_A(vel)

    B = (vb**2)*mb/vel/ma*A
    FWoSin = (pow(Zb/Za,2.0)*CoulLog*nb/2.0/vel*(vb/vel*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))
              +(1.0-pow(vb,2.0)/pow(vel,2.0))*erf(vel/sqrt(2.0)/vb)))
    F = FWoSin*sin(theta);

    #double check matrix indices
    mat[0,0] = B*sin(theta)*cos(theta)*cos(theta)/vel+vel*F*sin(theta)*sin(theta)
    mat[1,0] = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta)
    mat[0,1] = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta)
    mat[1,1] = B*sin(theta)*sin(theta)*sin(theta)/vel+vel*F*cos(theta)*cos(theta)
    
    mat*=-1.0/va/va*gamma

@mfem.jit.matrix(sdim = 2)
def quasi_func(ptx, out):    
    mat = carray(out, (2,2))
    x = ptx[0]

    w1 = 0.15
    w2 = 0.2
    mat[0,1] = 0.0
    mat[1,0] = 0.0
    mat[1,1] = 0.0
    mat[0,0] = 0.0

    if ((x < w2) & (x > w1)):
        mat[0,0] = 0.0 #-1.0e-10


def dirichlet_constraint(bdr, fes, mesh, rev=False):
    be_ids=np.where(mesh.GetBdrAttributeArray()==bdr)[0]
    
    @mfem.jit.scalar()
    def x(p):
        return p[0]
    @mfem.jit.scalar()
    def y(p):
        return p[1]
    
    
    gfx = mfem.GridFunction(fes)
    gfx.ProjectCoefficient(x)
    
    gfy = mfem.GridFunction(fes)
    gfy.ProjectCoefficient(y)
    
    arr = np.sqrt(gfy.GetDataArray()**2 + gfx.GetDataArray()**2)
    
    if (rev):
        be_ids=be_ids[::-1]
    
    be_dof_pairs=[fes.GetBdrElementDofs(i)[:] for i in be_ids]
    be_dof=[]
    be_id=[]
    
    for i, pair in enumerate(be_dof_pairs):
        for dof in pair:
            be_dof.append(dof)
            
    bdr_dofs = np.unique(np.array(be_dof))
    num_dof = len(bdr_dofs)
    idx = np.argsort([arr[i] for i in bdr_dofs])
    bdr_dofs = bdr_dofs[idx]
    
    bdr_identity_matrix = lil_matrix(((num_dof,fes.GetNDofs())))
    
    for i,dof in enumerate(bdr_dofs):
        bdr_identity_matrix[i,dof] = 1.0
        
    bdr_identity_matrix_transpose = mfem.SparseMatrix(bdr_identity_matrix.transpose().tocsr())
    bdr_identity_matrix = mfem.SparseMatrix(bdr_identity_matrix.tocsr())
    
    return bdr_identity_matrix, bdr_identity_matrix_transpose

def mixed_mm(fesH1, fesL2, mm_co):
#     f=mfem.MixedBilinearForm(fesL2, fesH1)
    f=mfem.MixedBilinearForm(fesH1, fesL2)
    f.AddDomainIntegrator(mfem.MixedScalarMassIntegrator(mm_co))
    f.Assemble()
    f.Finalize()
    f_mat=f.LoseMat()

    return f_mat



def mixed_flux_op(bdr, fesl2, mesh1D, fesh1, mesh2D, pnt_trans, pnt_arg = 0):
    be_ids_2D = np.where(mesh2D.GetBdrAttributeArray() == bdr)[0]
    el_ids_1D = np.where(mesh1D.GetAttributeArray() == 1)[0]
    flux = lil_matrix((fesl2.GetNDofs(), fesh1.GetNDofs()))
    dof_num = 0
    for i, j in zip(el_ids_1D, be_ids_2D):
        element1D = fesl2.GetFE(i)
        ir = element1D.GetNodes()
        
        transform1D = fesl2.GetElementTransformation(i)
        transform2D = fesh1.GetBdrElementTransformation(j)
        
        face = mesh2D.GetBdrFace(j)
        face_trans = mesh2D.GetFaceElementTransformations(face)

        el_info = mesh2D.GetBdrElementAdjacentElement(j)
        el_num = el_info[0]

        el = fesh1.GetFE(el_num)
        e_flux = mfem.Vector(el.GetDof())
        el_trans = fesh1.GetElementTransformation(el_num)
        dof_ids = fesh1.GetElementDofs(el_trans.ElementNo)
        for ii in range(ir.GetNPoints()):
            ip = ir.IntPoint(ii)
            ip_2D = mfem.IntegrationPoint()
            vec_pnt = mfem.Vector(2)
            vec_pnt.Assign(0.0)
            vec_pnt_x = transform1D.Transform(ip)
            vec_pnt[0] = vec_pnt_x[0]
            vec_pnt = pnt_trans(vec_pnt, pnt_arg)
            transform2D.TransformBack(vec_pnt, ip_2D)

            nor = mfem.Vector(mesh2D.SpaceDimension())
            dshape = mfem.DenseMatrix(el.GetDof(), dim)

            dof_ids = fesh1.GetElementDofs(el_trans.ElementNo)

            el_trans.SetIntPoint(ip_2D)  
            face_trans.SetIntPoint(ip_2D)

            mfem.CalcOrtho(face_trans.Jacobian(), nor)
            nor1=nor.GetDataArray()/np.linalg.norm(nor.GetDataArray())

            el.CalcPhysDShape(el_trans, dshape)

            dshape.Mult(mfem.Vector(nor1/ir.GetNPoints()),e_flux)
            for row, dof_id in enumerate(dof_ids):
                flux[dof_num, dof_id]+=e_flux.GetDataArray()[row]
            dof_num += 1

    flux_op=mfem.SparseMatrix(flux.tocsr())
    return flux_op

def pnt_vertical_trans(vec, val):
    vec[1] = val
    return vec

def pnt_rotation_trans(vec, angle):
    x = vec[0]*cos(angle) - vec[1]*sin(angle)
    y = vec[0]*sin(angle) + vec[1]*cos(angle)
    vec[0] = x 
    vec[1] = y
    return vec

class sorter_2d(mfem.PyCoefficient):
    def EvalValue(self,v):
        return v[0]**2 + v[1]**2
    
class sorter_1d(mfem.PyCoefficient):
    def EvalValue(self,v):
        return v[0]**2

def bdr_map(mesh1d, fes1d, mesh2d, fes2d, bdr):
    be_id = np.where(mesh2d.GetBdrAttributeArray() == bdr)[0]
    el_id = np.where(mesh1d.GetAttributeArray() == 1)[0]
#     print(list(mesh2d.GetBdrAttributeArray()))
#     print(list(mesh1d.GetAttributeArray()))
#     print(len(el_id), len(be_id))
    #do it for the 2d h1 case
    dof_2d = []
    dist_2d = []
    for id in be_id:
        dof_list = fes2d.GetBdrElementDofs(id)
        for dof in dof_list:
            if (not (dof in dof_2d)):
                dof_2d.append(dof)
    gf_2d = mfem.GridFunction(fes2d)
    lin_func = sorter_2d()
    gf_2d.ProjectCoefficient(lin_func)
    for id in dof_2d:
        dist_2d.append(gf_2d[id])
    sorted_2d = [x for (_,x) in sorted(zip(dist_2d,dof_2d))]
                    
    #do it for the 1d h1 case
    gf_1d = mfem.GridFunction(fes1d)
    lin_func = sorter_1d()
    gf_1d.ProjectCoefficient(lin_func)
    dof_1d = []
    dist_1d = []
    
    for dof, dist in enumerate(gf_1d.GetDataArray()):
        dof_1d.append(dof)
        dist_1d.append(dist)
    
    sorted_1d = [x for (_, x) in sorted(zip(dist_1d, dof_1d))]
    return sorted_1d, sorted_2d

def reduced_mixed_matrix(sorted_1d, sorted_2d, reduced_mm, fes2d):
#     print(len(sorted_1d), len(sorted_2d))
    assert len(sorted_1d) == len(sorted_2d)
    num_bdr_dof = len(sorted_2d)
    num_bdr_dof_l2 = reduced_mm.Height()
    
    num_dof = fes2d.GetNDofs()
    reduced_mixed_mm = lil_matrix((num_dof, num_bdr_dof_l2))
    
    reduced_mm = sparsemat_to_scipycsr(reduced_mm, float)
    for i in range(len(sorted_1d)):
        for j in range(num_bdr_dof_l2):
            reduced_mixed_mm[sorted_2d[i], j] = reduced_mm[j, sorted_1d[i]]
    
    return mfem.SparseMatrix(reduced_mixed_mm.tocsr())
            
#inputs are mesh1d, h1 1d fes, mesh2d, h1 2d fes, bdr #, and l2 1d fes
def neumann_mixed_bc(mesh1d, fes1d, mesh2d, fes2d, bdr, fesl2):
    one = mfem.ConstantCoefficient(1.0)
    
    reduced_mm = mixed_mm(fes1d, fesl2, one)
    sort_1d, sort_2d = bdr_map(mesh1d, fes1d, mesh2d, fes2d, bdr)
    reduced_mm = reduced_mixed_matrix(sort_1d, sort_2d, reduced_mm, fes2d)
    return reduced_mm

#Get dictionary of boundary integration point locations
def getBD_coords(mesh,fec,fes,bd_num):
    bdr=np.where(mesh.GetBdrAttributeArray()==bd_num)[0] #Get the ids of the elements on bd bd_num
    pts={} #create empty dict for storing coords with by id
    
    for i in bdr:
        el=fes.GetBE(i) #gets the boundary element with id i
        ir=el.GetNodes() #get integration rule
        tr=fes.GetBdrElementTransformation(i) #For converting to real space
        dof=fes.GetBdrElementDofs(i) #Get ids of dof
        
        for j in range(ir.GetNPoints()): #loop over dofs and get coords
            pts[dof[j]]=tr.Transform(ir.IntPoint(j))
            
    return pts

#Define pie class, named because the subdomains look like slices of pie and subdomain is already a concept in mfem
class pie:
    def __init__(self,mesh,fec):
        self.mesh = mesh
        self.fec = fec
        self.fes = mfem.FiniteElementSpace(mesh,self.fec)
        self.bdr_attributes = self.mesh.bdr_attributes.ToList()
        self.getBdr_dict()
        self.NDof=self.fes.GetNDofs()

    def getBdr_coords(self,bd_num):
        bdr=np.where(self.mesh.GetBdrAttributeArray()==bd_num)[0] #Get the ids of the elements on bd bd_num
        pts={} #create empty dict for storing coords with by id

        for i in bdr:
            el=self.fes.GetBE(i) #gets the boundary element with id i
            ir=el.GetNodes() #get integration rule
            tr=self.fes.GetBdrElementTransformation(i) #For converting to real space
            dof=self.fes.GetBdrElementDofs(i) #Get ids of dof

            for j in range(ir.GetNPoints()): #loop over dofs and get coords
                pts[dof[j]]=tr.Transform(ir.IntPoint(j))
        return pts
                
        
    def getBdr_dict(self):
        self.Bdr_dict={}
        for j in self.bdr_attributes:
            self.Bdr_dict[j]=self.getBdr_coords(j)
            
    def Bdr_plot(self):
        fig,ax=plt.subplots(figsize=(10,7))
        for j in self.bdr_attributes:
            x=[]
            y=[]
            for i in self.Bdr_dict[j]:
                x.append(self.Bdr_dict[j][i][0])
                y.append(self.Bdr_dict[j][i][1])
            plt.plot(x,y,label=str(j))
        plt.legend()

num_be = 10
order = 2
dim = 2

# left_mesh=mfem.Mesh("left_slice.mesh") #left piece
# right_mesh=mfem.Mesh("right_slice.mesh") #right piece
# mid_mesh=mfem.Mesh("mid_slice.mesh") #middle piece

left_mesh=mfem.Mesh("left_fixed.mesh") #left piece
right_mesh=mfem.Mesh("right_fixed.mesh") #right piece
mid_mesh=mfem.Mesh("mid_fixed.mesh") #middle piece

print(left_mesh.SpaceDimension())

# left_mesh.Save("left_fixed.mesh")
# mid_mesh.Save("mid_fixed.mesh")
# right_mesh.Save("right_fixed.mesh")

bdr_mesh = mfem.Mesh(num_be)

#bdr domain
fec_h1_1d = mfem.H1_FECollection(order, dim-1)
fec_l2 = mfem.L2_FECollection(order-1, dim-1)

fes_h1_1d = mfem.FiniteElementSpace(bdr_mesh, fec_h1_1d)
fes_l2 = mfem.FiniteElementSpace(bdr_mesh, fec_l2)

#left domain
fec_h1_2d_left = mfem.H1_FECollection(order, dim)
fes_h1_2d_left = mfem.FiniteElementSpace(left_mesh, fec_h1_2d_left)


#right domain
fec_h1_2d_right = mfem.H1_FECollection(order, dim)
fes_h1_2d_right = mfem.FiniteElementSpace(right_mesh, fec_h1_2d_right)


#mid domain
fec_h1_2d_mid = mfem.H1_FECollection(order, dim)
fes_h1_2d_mid = mfem.FiniteElementSpace(mid_mesh, fec_h1_2d_mid)

#coefficients
zero = mfem.ConstantCoefficient(0.0)
one = mfem.ConstantCoefficient(1.0)


#left domain bilinear and linear forms
bdr_left = 2
ess_tdof_list_left = mfem.intArray()
ess_bdr_left = mfem.intArray(left_mesh.bdr_attributes.Size())
ess_bdr_left.Assign(0)
ess_bdr_left[2]=1
fes_h1_2d_left.GetEssentialTrueDofs(ess_bdr_left, ess_tdof_list_left)

a_left=mfem.BilinearForm(fes_h1_2d_left)
# a_left.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
a_left.AddDomainIntegrator(mfem.DiffusionIntegrator(BF_func))
a_left.AddDomainIntegrator(mfem.ConvectionIntegrator(A_func))
a_left.AddDomainIntegrator(mfem.MassIntegrator(dA_func))
a_left.Assemble()

b_left=mfem.LinearForm(fes_h1_2d_left)
b_left.AddDomainIntegrator(mfem.DomainLFIntegrator(zero))
# b_left.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
b_left.Assemble()

x_left=mfem.GridFunction(fes_h1_2d_left)
x_left.Assign(0.0)
x_left.ProjectCoefficient(one)

A_left = mfem.OperatorPtr()
B_left = mfem.Vector()
X_left = mfem.Vector()
a_left.FormLinearSystem(ess_tdof_list_left, x_left, b_left, A_left, X_left, B_left)

bdr_left_ident, bdr_left_ident_trans = dirichlet_constraint(bdr_left, fes_h1_2d_left, left_mesh)
reduced_mm_left = neumann_mixed_bc(bdr_mesh, fes_h1_1d, left_mesh, fes_h1_2d_left, bdr_left, fes_l2)
flux_op_left = mixed_flux_op(bdr_left, fes_l2, bdr_mesh, fes_h1_2d_left, left_mesh, pnt_rotation_trans, 3.0*pi/4.0)

#right domain bilinear and linear forms

bdr_right = 2
ess_tdof_list_right = mfem.intArray()
ess_bdr_right = mfem.intArray(right_mesh.bdr_attributes.Size())
ess_bdr_right.Assign(0)
ess_bdr_right[2]=1
fes_h1_2d_right.GetEssentialTrueDofs(ess_bdr_right, ess_tdof_list_right)

a_right=mfem.BilinearForm(fes_h1_2d_right)
a_right.AddDomainIntegrator(mfem.DiffusionIntegrator(BF_func))
# a_right.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
a_right.AddDomainIntegrator(mfem.DiffusionIntegrator(quasi_func))
a_right.AddDomainIntegrator(mfem.ConvectionIntegrator(A_func))
a_right.AddDomainIntegrator(mfem.MassIntegrator(dA_func))
a_right.Assemble()

b_right=mfem.LinearForm(fes_h1_2d_right)
b_right.AddDomainIntegrator(mfem.DomainLFIntegrator(zero))
# b_right.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
b_right.Assemble()

x_right=mfem.GridFunction(fes_h1_2d_right)
x_right.Assign(0.0)
x_right.ProjectCoefficient(one)



A_right = mfem.OperatorPtr()
B_right = mfem.Vector()
X_right = mfem.Vector()
a_right.FormLinearSystem(ess_tdof_list_right, x_right, b_right, A_right, X_right, B_right)

bdr_right_ident, bdr_right_ident_trans = dirichlet_constraint(bdr_right, fes_h1_2d_right, right_mesh)
reduced_mm_right = neumann_mixed_bc(bdr_mesh, fes_h1_1d, right_mesh, fes_h1_2d_right, bdr_right, fes_l2)
flux_op_right = mixed_flux_op(bdr_right, fes_l2, bdr_mesh, fes_h1_2d_right, right_mesh, pnt_rotation_trans, pi/4.0)

#mid domain bilinear and linear forms

bdr_mid = 3
ess_tdof_list_mid = mfem.intArray()
ess_bdr_mid = mfem.intArray(mid_mesh.bdr_attributes.Size())
ess_bdr_mid.Assign(0)
ess_bdr_mid[1]=1
fes_h1_2d_mid.GetEssentialTrueDofs(ess_bdr_mid, ess_tdof_list_mid)

a_mid=mfem.BilinearForm(fes_h1_2d_mid)
# a_mid.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
a_mid.AddDomainIntegrator(mfem.DiffusionIntegrator(BF_func))
a_mid.AddDomainIntegrator(mfem.DiffusionIntegrator(quasi_func))
a_mid.AddDomainIntegrator(mfem.ConvectionIntegrator(A_func))
a_mid.AddDomainIntegrator(mfem.MassIntegrator(dA_func))
a_mid.Assemble()

b_mid=mfem.LinearForm(fes_h1_2d_mid)
b_mid.AddDomainIntegrator(mfem.DomainLFIntegrator(zero))
# b_mid.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
b_mid.Assemble()

x_mid=mfem.GridFunction(fes_h1_2d_mid)
x_mid.Assign(0.0)
x_mid.ProjectCoefficient(one)



A_mid = mfem.OperatorPtr()
B_mid = mfem.Vector()
X_mid = mfem.Vector()
a_mid.FormLinearSystem(ess_tdof_list_mid, x_mid, b_mid, A_mid, X_mid, B_mid)

bdr_mid_ident, bdr_mid_ident_trans = dirichlet_constraint(bdr_mid, fes_h1_2d_mid, mid_mesh)
reduced_mm_mid = neumann_mixed_bc(bdr_mesh, fes_h1_1d, mid_mesh, fes_h1_2d_mid, bdr_mid, fes_l2)
flux_op_mid = mixed_flux_op(bdr_mid, fes_l2, bdr_mesh, fes_h1_2d_mid, mid_mesh, pnt_rotation_trans, pi/4.0)

#assemble block matrix and solve problem

block_offsets = mfem.intArray([0.0, a_mid.SpMat().Height(), a_left.SpMat().Height(), a_right.SpMat().Height(), flux_op_mid.Height(), bdr_left_ident.Height(), bdr_right_ident.Height()])
print(list(block_offsets))
block_offsets.PartialSum()

dif_flux_op = mfem.BlockOperator(block_offsets)
rhs = mfem.BlockVector(block_offsets)
rhs.Assign(0.0)
block_solution=mfem.BlockVector(block_offsets)
block_solution.Assign(0.0)

rhs.GetBlock(0).Set(1.0, B_mid)
rhs.GetBlock(1).Set(1.0, B_left)
rhs.GetBlock(2).Set(1.0, B_right)
rhs.GetBlock(3).Assign(0.0)
rhs.GetBlock(4).Assign(0.0)
rhs.GetBlock(5).Assign(0.0)

bdr_mid_ident *= -1.0

identity = mfem.IdentityOperator(flux_op_mid.Height())

dif_flux_op.SetBlock(0, 0, A_mid.Ptr())
dif_flux_op.SetBlock(1, 1, A_left.Ptr())
dif_flux_op.SetBlock(2, 2, A_right.Ptr())

#dirichlet
if (1):
    print("Dirichlet imposed")
    dif_flux_op.SetBlock(0, 4, bdr_mid_ident_trans)
    dif_flux_op.SetBlock(0, 5, bdr_mid_ident_trans)
    dif_flux_op.SetBlock(1, 4, bdr_left_ident_trans)
    dif_flux_op.SetBlock(2, 5, bdr_right_ident_trans)

    dif_flux_op.SetBlock(4, 0, bdr_mid_ident)
    dif_flux_op.SetBlock(4, 1, bdr_left_ident)
    dif_flux_op.SetBlock(5, 0, bdr_mid_ident)
    dif_flux_op.SetBlock(5, 2, bdr_right_ident)

#neumann
if (0):
    print("Flux Read out")
    dif_flux_op.SetBlock(3, 0, flux_op_mid)
    dif_flux_op.SetBlock(3, 3, identity)
else:
    print("Neumann BC")
    reduced_mm_mid *= -1.0
    flux_op_mid *= 2.0
    dif_flux_op.SetBlock(3, 0, flux_op_mid)
    dif_flux_op.SetBlock(3, 1, flux_op_left)
    dif_flux_op.SetBlock(3, 2, flux_op_right)
    dif_flux_op.SetBlock(0, 3, reduced_mm_mid)
    dif_flux_op.SetBlock(1, 3, reduced_mm_left)
    dif_flux_op.SetBlock(2, 3, reduced_mm_right)


# dif_flux_op.SetBlock()
# dif_flux_op.SetBlock()
# dif_flux_op.SetBlock()
# dif_flux_op.SetBlock()
# dif_flux_op.SetBlock()

atol=1e-15
rtol=1e-10
maxIter=50

psolver=mfem.GMRESSolver()
psolver.SetAbsTol(atol)
psolver.SetRelTol(rtol)
psolver.SetMaxIter(maxIter)
psolver.SetOperator(dif_flux_op)
psolver.SetPrintLevel(-1)

atol=1e-15
rtol=1e-10
maxIter=1000

solver=mfem.FGMRESSolver()
solver.SetAbsTol(atol)
solver.SetRelTol(rtol)
solver.SetMaxIter(maxIter)
solver.SetOperator(dif_flux_op)
solver.SetPreconditioner(psolver)
solver.SetPrintLevel(1)

solver.Mult(rhs, block_solution)

fp_mid = mfem.GridFunction(fes_h1_2d_mid)
fp_left = mfem.GridFunction(fes_h1_2d_left)
fp_right = mfem.GridFunction(fes_h1_2d_right)
fp_flux = mfem.GridFunction(fes_l2)

fp_mid.MakeRef(fes_h1_2d_mid, block_solution.GetBlock(0), 0)
fp_left.MakeRef(fes_h1_2d_left, block_solution.GetBlock(1), 0)
fp_right.MakeRef(fes_h1_2d_right, block_solution.GetBlock(2), 0)
fp_flux.MakeRef(fes_l2, block_solution.GetBlock(3), 0)

fp_mid.Save("mid_sol.gf")
fp_left.Save("left_sol.gf")
fp_right.Save("right_sol.gf")
fp_flux.Save("flux_sol.gf")


#
# plotting
#

import matplotlib.pyplot as plt
import matplotlib.tri as tri

@mfem.jit.scalar()
def x(p):
    return p[0]
@mfem.jit.scalar()
def y(p):
    return p[1]

def dom_plot(sol_gf, fes, axes, flip = False, return_tpc = False):
    x_gf = mfem.GridFunction(fes)
    y_gf = mfem.GridFunction(fes)
    
    x_gf.ProjectCoefficient(x)
    y_gf.ProjectCoefficient(y)
    
    if (flip):
        triang = tri.Triangulation(x_gf.GetDataArray()*-1, y_gf.GetDataArray())
    else:
        triang = tri.Triangulation(x_gf.GetDataArray(), y_gf.GetDataArray())
    if (log_scale):
        if (contour):
            tpc = axes.tricontour(triang, log(sol_gf.GetDataArray()), levels = 30) #, shading='gouraud')
        else:
            tpc = axes.tripcolor(triang, log(sol_gf.GetDataArray()))#, shading='gouraud')
    else:
        if (contour):
            tpc = axes.tricontour(triang, sol_gf.GetDataArray(), levels = 30) #, shading='gouraud')
        else:
            tpc = axes.tripcolor(triang, sol_gf.GetDataArray())#, shading='gouraud')
    
    if (return_tpc):
        return tpc
        

fig, ax = plt.subplots(figsize = (13,7))

dom_plot(fp_right, fes_h1_2d_right, ax)
dom_plot(fp_left, fes_h1_2d_left, ax)
dom_plot(fp_mid, fes_h1_2d_mid, ax)
tpc = dom_plot(fp_mid, fes_h1_2d_mid, ax, True, True)
fig.colorbar(tpc)

plt.xlim(-1.2,1.2)
plt.ylim(-0.2,1.2)
plt.show()
