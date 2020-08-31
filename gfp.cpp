#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace mfem;
using namespace std::chrono;

double w1=0.3;
double w2=0.5;
double D=1.e-3;

void matFun(const Vector &, DenseMatrix &);
void matFunSym(const Vector &, Vector&);
void vecFun(const Vector &, Vector&);
double scalFun(const Vector & x);
double testFun(const Vector & x);
vector<double> FPCo(const vector<double> & v);
vector<double> prelimFPCo(const Vector & v);
//void constMat(const Vector & x, DenseMatrix & m);
void constMat(const Vector & x, DenseMatrix & m)
{

  //double w1,w2;
  double fac,pi; 
  //w1 = 0.3;
  //w2 = 0.5;
  fac=5.;
  pi=3.141592;
 m(0,1) = 0;
 m(1,0) = 0;
 m(1,1) = 0;
 m(0,0) = 0;

 //if ((x(0) < w2) & (x(0) > w1)){
 m(0,0) =-((tanh((x(0)-w1)*2*pi*fac)+1)*(tanh(-1*(x(0)-w2)*2*pi*fac)+1))/4*D;
 //}
}

int main(int argc, char *argv[])
{

  

  

OptionsParser args(argc, argv);
 int refinement = 3; 
 int order = 2;
   bool pa = false;
   const char *device_config = "cpu";
   const char *mesh_file = "./data/semi_circle5_quad.msh"; 

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
  
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");

   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");

   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");

   args.AddOption(&refinement, "-r", "--refine",
		  "number of refinements of the mesh"
		  "increases accuracy and computation time");
   args.AddOption(&w1, "-w1", "--Bound1","LH term lower bound");
   args.AddOption(&w2, "-w2", "--Bound2","LH term upper bound");
   args.AddOption(&D, "-D", "--LHmaxValue","LH term coefficient max value");

   
     
   


   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }   
   args.PrintOptions(cout);   
   Device device(device_config);
   device.Print();

  
   //for (int j = 1; j < 6; j++) {


 auto start = high_resolution_clock::now();

   //start loop 
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  for (int i =0; i < refinement; i++) { 
  mesh->UniformRefinement();
  }
  
  FiniteElementCollection *fec;

  if (order > 0) {
  fec = new H1_FECollection(order, dim);
  }
  else if (mesh->GetNodes()) {     
  fec = mesh->GetNodes()->OwnFEC();
  cout << "Using isoparametric FEs: " << fec->Name() << endl;  
  }
  else {   
  fec = new H1_FECollection(order = 1, dim); 
  }


  //fec = new H1_FECollection(order, dim);
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
    {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr=0;
      ess_bdr[2]=1;
      ess_bdr[3]=1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      //ess_bdr.Print();
    }

 
LinearForm *b = new LinearForm(fespace);
ConstantCoefficient zero(0.0);
b->AddDomainIntegrator(new DomainLFIntegrator(zero));
b->Assemble();
GridFunction x(fespace);
x=1.0;

BilinearForm *a = new BilinearForm(fespace);
if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
 int sdim = mesh->Dimension();

 //std::cout << "mesh dim " << sdim << "\n";
 /*
double z[2];	 
 Vector v(z, 2);
 v(0)=0.1;
 v(1)=0.1;
 DenseMatrix m(2,2);
 matFun(v,m);
 */

 VectorFunctionCoefficient vecFunCo(sdim, vecFun);
 a->AddDomainIntegrator(new ConvectionIntegrator(vecFunCo)); 

MatrixFunctionCoefficient constMatFunCo(sdim,constMat);
a->AddDomainIntegrator(new DiffusionIntegrator(constMatFunCo));

 FunctionCoefficient scalFunCo(scalFun);
 a->AddDomainIntegrator(new MassIntegrator(scalFunCo));
 
if (pa) {
 MatrixFunctionCoefficient matFunCo(sdim,matFunSym);
 matFunCo.SetSymmetric(true);
 a->AddDomainIntegrator(new DiffusionIntegrator(matFunCo));
 }
 else {
MatrixFunctionCoefficient matFunCo(sdim,matFun);
 a->AddDomainIntegrator(new DiffusionIntegrator(matFunCo));
 }


 a->Assemble();
 OperatorPtr A;
 //SparseMatrix A;
 Vector B, X;
 a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

/*
 FunctionCoefficient testCo(testFun);
FiniteElementCollection *fec_2;
  fec_2 = new H1_FECollection(3, dim);
  FiniteElementSpace *fespace_2 = new FiniteElementSpace(mesh, fec_2);
GridFunction testSpace(fespace_2);
testSpace.ProjectCoefficient(testCo);
ofstream sol_ofs2("co.gf");
sol_ofs2.precision(10);
testSpace.Save(sol_ofs2);
*/

//A.Save("matrix");
/*
ofstream mat_ofs("matrix_d.mesh");
 SparseMatrix *Am = A.As<SparseMatrix>();
 Am->Print(mat_ofs);
 Operator *op = A.Ptr();
 std::cout << op->GetType();
*/

/*
if (UsesTensorBasis(*fespace))
  {
    OperatorJacobiSmoother M(*a, ess_tdof_list);
    PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
    std::cout << "using pcg->UsesTensorBasis=true";
  }
 else
   {
     CG(*A, B, X, 1, 300, 1e-12, 0.0);
   }
*/

 


 GMRESSolver *j_gmres = new GMRESSolver();
 j_gmres->iterative_mode = false; 
   j_gmres->SetRelTol(1e-8);
   j_gmres->SetAbsTol(0);
   j_gmres->SetMaxIter(300);
   j_gmres->SetPrintLevel(-1);
   if (! pa) {
     GSSmoother M((SparseMatrix&)(*A));     
     j_gmres->SetPreconditioner(M);
   }
   j_gmres->SetOperator(*A);

   FGMRESSolver *fgmres = new FGMRESSolver();
   fgmres-> SetRelTol(1e-16);
   fgmres-> SetAbsTol(0);
   fgmres-> SetMaxIter(8000);
   fgmres-> SetKDim(60);
   fgmres-> SetPrintLevel(1);
   fgmres-> SetOperator(*A);
   fgmres-> SetPreconditioner(*j_gmres);
   fgmres->Mult(B,X);

a->RecoverFEMSolution(X, *b, x);

ofstream mesh_ofs("refined.mesh");
mesh_ofs.precision(8);
mesh->Print(mesh_ofs);
ofstream sol_ofs("sol.gf");
sol_ofs.precision(8);
x.Save(sol_ofs);

delete a;
delete b;
delete fespace;
delete fec;
delete mesh;

 
 auto stop = high_resolution_clock::now();
 auto duration =duration_cast<microseconds>(stop-start);
 cout << "time is " <<duration.count() << endl;
   

return 0;
}

void matFun(const Vector & x, DenseMatrix & m)
{
  //std::cout << "begin of mat \n ";
  vector<double> velocities = prelimFPCo(x); 
  //std::cout << "after prelim \n ";
vector<double> coefficients = FPCo(velocities);
//std::cout << "after fpco \n ";

 m(0,0)=coefficients[3];
  m(0,1)=coefficients[4];
  m(1,0)=coefficients[5];
  m(1,1)=coefficients[6]; 

  //std::cout << "end of mat \n ";
}



void matFunSym(const Vector & x, Vector & K)
{
  vector<double> velocities = prelimFPCo(x); 
  vector<double> coefficients = FPCo(velocities);

  K(0)=coefficients[3];
  K(1)=coefficients[4];
  K(2)=coefficients[6];
  //std::cout << "end of mat \n ";
}

void vecFun(const Vector & x, Vector & f)
{
  vector<double> velocities = prelimFPCo(x); 
  vector<double> coefficients = FPCo(velocities);

  f(0)=coefficients[1];
  f(1)=coefficients[2];

}

double scalFun(const Vector & x)
{
  double co;
  vector<double> velocities = prelimFPCo(x); 
  vector<double> coefficients = FPCo(velocities);
  co = coefficients[0];
  return co;
 
}

double testFun(const Vector & x)
{
  double co;
  vector<double> velocities = prelimFPCo(x); 
  vector<double> coefficients = FPCo(velocities);
  co = coefficients[2];
  return co;
 
}



vector<double> prelimFPCo(const Vector & v){
  vector<double> velocities;
  double x1,x0,vel,theta;
  x1=v(1);
 if (v(1) < 0)
   {
     x1=0;
   }

theta = atan2(x1,v(0));
vel = sqrt(pow(v(0),2.0)+pow(x1,2.0));
 x0=v(0);
if (vel == 0.)
   {
     //vel = 984860.7096481468;
     vel = 1.;
     
     //std::cout << "I am here!\n" << "   " <<  x(0) << ' ' << x(1) << "\n";
   }
 velocities.reserve(10);
 velocities.push_back(x0);
 velocities.push_back(x1);
 velocities.push_back(vel);
 velocities.push_back(theta);

 return velocities;
}



vector<double> FPCo(const vector<double> & v){
const double CoulLog = 15.;
const double Zb = 1.;
const double Za = 1.;
 const double nb = 1.530e27;//1.9e+26; //1.0e20;
const double vb = 8.8e6; 
const double va=vb*5.0;
const double ma = 9.1093837015e-31;
const double mb = 9.1093837015e-31;
const double pi = 3.1415926535897932;
const double ee = 1.602176634e-19;
const double gama = 4.0*pi*pow(Za,4.0)*pow(ee,4.0)/pow(ma,2.0);
//std::cout << "first\n";
 double x0,x1,theta,vel;
 x0=v[0]*va;
 x1=v[1]*va;
 vel=v[2]*va;
 theta=v[3];
 //std::cout << x0 << " " << x1 << " " << theta << " " << vel << " \n";
double A = pow(Zb/Za,2.0)*CoulLog*ma/mb*nb*(-vel/vb*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))+erf(vel/sqrt(2.0)/vb));
double B = pow(vb,2.0)*mb/vel/ma*A;
double FWoSin = pow(Zb/Za,2.0)*CoulLog*nb/2.0/vel*(vb/vel*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))+(1.0-pow(vb,2.0)/pow(vel,2.0))*erf(vel/sqrt(2.0)/vb));
double F = FWoSin*sin(theta);
double dA = pow(Zb/Za,2.0)*CoulLog*ma/mb*nb*pow(vel,2.0)/pow(vb,3.0)*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0));
//std::cout << "dA " << dA << " \n";
 double base, x, y, xx, xy, yx, yy;
 base = dA/vel*sin(theta);

 x = A*sin(theta)*cos(theta)/vel;
 y = A*sin(theta)*sin(theta)/vel;

 xx = B*sin(theta)*cos(theta)*cos(theta)/vel+vel*F*sin(theta)*sin(theta);
 xy = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta);
 yx = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta);
 yy = B*sin(theta)*sin(theta)*sin(theta)/vel+vel*F*cos(theta)*cos(theta);

 vector<double> coefficients;
 /*
 std::cout << "xy and yx" << " "<< xy << " "<< yx << "\n"; 
std::cout << "xx and yy" << " "<< xx << " "<< yy << "\n";
std::cout << "x and y" << " "<< x << " "<< y << "\n";
 std::cout << "capacity " << coefficients.size() << "\n";
 */
coefficients.reserve(10);
coefficients.push_back(1.0*gama*base);
coefficients.push_back(1.0*gama*x/va);
coefficients.push_back(1.0*gama*y/va);
coefficients.push_back(-1.*gama*xx/va/va);
coefficients.push_back(-1.*gama*xy/va/va);
coefficients.push_back(-1.*gama*yx/va/va);
coefficients.push_back(-1.*gama*yy/va/va);
//std::cout << coefficients[0] << "\n";
 //std::cout<< "made it to the end!\n";
 return coefficients;

}


