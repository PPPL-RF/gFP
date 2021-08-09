
/*

Time-Dependent Fokker-Planck Solver for Non-Relativistic Uniform Plasma

 */

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <stdlib.h>

using namespace std;
using namespace mfem;
using namespace std::chrono;

const double pi = 3.1415926535898;
const double k = 8.617333262e-5; // boltzmann constant in eV/K

double T_0 = 100.0; //eV
double T_f = 1000.0;  //eV
double w1 = 0.3;
double w2 = 0.5;
double D = 1.e-2;
double v_max = 6.0e7;

void matFun(const Vector &, DenseMatrix &);
void matFunSym(const Vector &, Vector&);
void vecFun(const Vector &, Vector&);
double scalFun(const Vector & x);
double testFun(const Vector & x);
vector<double> FPCo(const vector<double> & v);
vector<double> prelimFPCo(const Vector & v);
double InitialDist(const Vector & v);
double Jacobian(const Vector & v);


class FokkerPlanckOperator : public TimeDependentOperator {
  
protected:
  int dim = 2;
  FiniteElementSpace &fespace;
  Array<int> ess_tdof_list;
  
  BilinearForm *M;
  BilinearForm *K;
  
  SparseMatrix Mmat, Kmat;
  SparseMatrix *T;
  double current_dt;

  CGSolver M_solver;
  DSmoother M_prec;

  CGSolver T_solver;
  //FGMRESSolver T_solver;
  DSmoother T_prec;
  // GMRESSolver T_prec;

  mutable Vector z;

public:
  double particles; //tracking # of particles for error checking

  bool particles_defined = false;
  
  FokkerPlanckOperator(FiniteElementSpace &f, const Vector &u);
  
  virtual void Mult(const Vector &u, Vector &du_dt) const;

  virtual void ImplicitSolve(const double dt, const Vector &u, Vector &du_dt);
  
  void SetParameters(const Vector & u);

  void ParticleConservation(const GridFunction & gf);

  virtual ~FokkerPlanckOperator();
 

};


void constMat(const Vector & x, DenseMatrix & m)
{
 
  double fac = 5.0;
  m(0,1) = 0;
  m(1,0) = 0;
  m(1,1) = 0;
  m(0,0) = 0;

  m(0,0) = -((tanh((x(0)-w1)*2*pi*fac)+1)*(tanh(-1*(x(0)-w2)*2*pi*fac)+1))/4*D;
 }


int main(int argc, char *argv[])
{
 
 OptionsParser args(argc, argv);
 int refinement = 3; 
 int order = 2;
 bool pa = false;
 const char *device_config = "cpu";
 const char *mesh_file = "./data/semi_circle5_quad.msh";
 int ode_solver_type = 11;
 double t_final = 0.005;
 double dt = t_final/1000;
 bool visualization = true;
 bool visit = false;
 int vis_steps = 5;

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
   args.AddOption(&t_final, "-tf", "--t_final", "Final Time");
   args.AddOption(&dt,"-dt", "--time-step", "Time step");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
		  "Enable or disable GLVIS visualization");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit", "--no-visit-datafiles",
		  "Save data files for VisIt visualization");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps", "Visualize every n-th timestep");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }   
   args.PrintOptions(cout);   
   Device device(device_config);
   device.Print();


   auto start = high_resolution_clock::now();

  ODESolver *ode_solver;

  switch (ode_solver_type){

  case 1:  ode_solver = new BackwardEulerSolver; break;
  case 11: ode_solver= new ForwardEulerSolver; break;
  case 12: ode_solver = new RK2Solver(0.5); break;

  }

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  
  for (int i =0; i < refinement; i++)
  { 
      mesh->UniformRefinement();
  }
  
  FiniteElementCollection *fec;

  if (order > 0)
  {
    fec = new H1_FECollection(order, dim);
  }
  else if (mesh->GetNodes())
  {
    fec = mesh->GetNodes()->OwnFEC();
    cout << "Using isoparametric FEs: " << fec->Name() << endl;  
  }
  else
  {   
    fec = new H1_FECollection(order = 1, dim); 
  }


  //FiniteElementSpace fespace(mesh, fec);
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

  
  GridFunction x(fespace);
  x=1.0;

  
  BilinearForm *a = new BilinearForm(fespace);
  if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
  //int sdim = mesh->Dimension();
  
  
  
  //Set up initial condition 
  FunctionCoefficient x_0(InitialDist);
  x.ProjectCoefficient(x_0);
  Vector x_vec;
  x.GetTrueDofs(x_vec);

  //Create FP Time Dependent Operator
  FokkerPlanckOperator fp_oper(*fespace, x_vec);

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh->Print(mesh_ofs);
  ofstream init("FP_init.gf");
  init.precision(8);
  x.Save(init);


  //setting up continuous visualization
  socketstream sout;
  if (visualization)
    {
      char vishost[]="localhost";
      int visport = 19916;
      sout.open(vishost, visport);
      if (!sout)
	{
	  cout << "Unable to connect to GLVis server" << endl;
	  visualization = false;
	}
      else {
	sout.precision(8);
	 sout << "solution\n" << *mesh << x;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
    }

  ode_solver->Init(fp_oper); 
  double t = 0.0;
  bool last_step = false;
  
  for (int step = 1; !last_step; step++){

    if (t + dt >= t_final-dt/2)
      {
	
	last_step = true;
      
      }
    fp_oper.ParticleConservation(x);
    ode_solver->Step(x_vec, t, dt); 
    
    if(last_step || (step % vis_steps) == 0)
      {
	
	cout << "t = " << t << endl;

       
	if (visualization)
	{
	  x.SetFromTrueDofs(x_vec);	  
	  sout << "solution \n" << *mesh << x << flush;
	
	}
      }
    fp_oper.SetParameters(x_vec);
  }





 FunctionCoefficient testCo(testFun);
 FiniteElementCollection *fec_2;
 fec_2 = new H1_FECollection(3, dim);
 FiniteElementSpace *fespace_2 = new FiniteElementSpace(mesh, fec_2);
 GridFunction testSpace(fespace_2);
 testSpace.ProjectCoefficient(testCo);
 ofstream sol_ofs2("co.gf");
 sol_ofs2.precision(10);
 testSpace.Save(sol_ofs2);


  
  
  delete a;
  delete fespace;
  delete fec;
  delete mesh;
  delete ode_solver;

 
  auto stop = high_resolution_clock::now();
  auto duration =duration_cast<microseconds>(stop-start);
  cout << "time is " <<duration.count() << endl;
   

  return 0;
}

FokkerPlanckOperator::FokkerPlanckOperator(FiniteElementSpace &f, const Vector &u) :
  TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL), T(NULL), current_dt(0.0), z(height)
{
  const double rel_tol = 1.e-8;

Mesh * mesh = fespace.GetMesh();

  if (mesh->bdr_attributes.Size())
    {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      
      ess_bdr=0;
      ess_bdr[2]=1;
      ess_bdr[3]=1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

  
  M = new BilinearForm(&fespace);
  M->AddDomainIntegrator(new MassIntegrator());
  M->Assemble();
  M->FormSystemMatrix(ess_tdof_list, Mmat);

  M_solver.iterative_mode = false;
  M_solver.SetRelTol(rel_tol);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(30);
  M_solver.SetPrintLevel(0);
  M_solver.SetPreconditioner(M_prec);
  M_solver.SetOperator(Mmat);
  /*
  T_prec.iterative_mode = false; 
   T_prec.SetRelTol(1e-8);
   T_prec.SetAbsTol(0);
   T_prec.SetMaxIter(50);
   T_prec.SetPrintLevel(-1);
   
   if (! pa) {
     GSSmoother M((SparseMatrix&)(*A));     
     j_gmres->SetPreconditioner(M);
   }
   
   

   
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0);
   T_solver.SetMaxIter(8000);
   T_solver.SetKDim(60);
   T_solver.SetPrintLevel(1);
   T_solver.SetPreconditioner(T_prec);
   
*/
  
  T_solver.iterative_mode = false;
  //T_solver.SetKDim(50);
  T_solver.SetRelTol(rel_tol);
  T_solver.SetAbsTol(0.0);
  T_solver.SetMaxIter(1000);
  T_solver.SetPrintLevel(1);
  T_solver.SetPreconditioner(T_prec);
  

  SetParameters(u);

  cout << "Fokker-Planck Operator Initialized" << endl;

  //ofstream ess_Out("ess_out.txt");
  //ess_tdof_list.Print(ess_Out);

  //ofstream mOut("Mmat_data.txt");
  //  Mmat.Print(mOut);
  
}

void FokkerPlanckOperator::ParticleConservation(const GridFunction & gf)
{
  double updated_particles;
  LinearForm lin(&fespace);
  FunctionCoefficient jac_fp(Jacobian);
  ConstantCoefficient part_scale(pow(v_max,3.0));
  //lin.AddDomainIntegrator(new DomainLFIntegrator(part_scale));
  lin.AddDomainIntegrator(new DomainLFIntegrator(jac_fp));
  lin.Assemble();

  updated_particles = lin(gf);//lin.operator() (gf);

  if (!particles_defined)
    {
      particles_defined = true;
      particles = updated_particles;

    }
  else
    {
      double del_particles = (particles-updated_particles)/particles;
      
      if (abs(del_particles) > 0.01)
	{
	  cout << "\n% Change above 0.01" << endl;
	  cout << "Particles changed by " << del_particles * 100. << "%" << endl;
	}
      
      particles = updated_particles;
      cout << "# particles is " << particles <<  endl;
    }
 
  
}

void FokkerPlanckOperator::SetParameters(const Vector &u)
{
  K = new BilinearForm(&fespace);

  VectorFunctionCoefficient vecFunCo(dim, vecFun);
  K->AddDomainIntegrator(new ConvectionIntegrator(vecFunCo));
  
  FunctionCoefficient scalFunCo(scalFun);
  K->AddDomainIntegrator(new MassIntegrator(scalFunCo));

  MatrixFunctionCoefficient matFunCo(dim ,matFun);
  K->AddDomainIntegrator(new DiffusionIntegrator(matFunCo));

  

  MatrixFunctionCoefficient constMatFunCo(dim, constMat);                //For LH Term
  K->AddDomainIntegrator(new DiffusionIntegrator(constMatFunCo));
  
  /*
  if (pa)
    {
      MatrixFunctionCoefficient matFunCo(dim, matFunSym);
      K->AddDomainIntegrator(new DiffusionIntegrator(matFunCo));
    }
  else
    {
      MatrixFunctionCoefficient matFunCo(dim, matFun);
      K->AddDomainIntegrator(new DiffusionIntegrator(matFunCo));
    }
  */

  
  
  K->Assemble();
  K->FormSystemMatrix(ess_tdof_list, Kmat);

  //ofstream kOut("Kmat_data.txt");
  //Kmat.Print(kOut);

  delete T;
  T = NULL;
      
}

void FokkerPlanckOperator::Mult(const Vector &u, Vector &du_dt) const
{
  Kmat.Mult(u,z);
  z.Neg();
  M_solver.Mult(z,du_dt);
  //cout << "being used" << endl;
}

void FokkerPlanckOperator::ImplicitSolve(const double dt, const Vector & u, Vector & du_dt)
{

  if (!T)
    {
      T = Add(1.0, Mmat, -1.0*dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
      T_prec.SetOperator(*T);

      //if (! pa) {
      //DSmoother M((SparseMatrix&)(Mmat));     
      //T_prec.SetPreconditioner(M);
     //}
    }
  MFEM_VERIFY(dt == current_dt, "");
  Kmat.Mult(u,z);
  //z.Neg();
  T_solver.Mult(z, du_dt);
  //du_dt.Print("du_dt.gf");
  //cout << "calling T_solver"<< endl;
}

FokkerPlanckOperator::~FokkerPlanckOperator()
{

  delete T;
  delete M;
  delete K;
  
}

void matFun(const Vector & x, DenseMatrix & m)
{
  vector<double> velocities = prelimFPCo(x); 
  vector<double> coefficients = FPCo(velocities);


  m(0,0)=coefficients[3];
  m(0,1)=coefficients[4];
  m(1,0)=coefficients[5];
  m(1,1)=coefficients[6];

  //cout << m(0,0) << endl;
  //  cout << m(1,1) << endl;

}



void matFunSym(const Vector & x, Vector & K)
{
  vector<double> velocities = prelimFPCo(x); 
  vector<double> coefficients = FPCo(velocities);

  K(0)=coefficients[3];
  K(1)=coefficients[4];
  K(2)=coefficients[6];
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
  co = coefficients[5];
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
     vel = 1.;
   }
 
 velocities.reserve(10);
 velocities.push_back(x0);
 velocities.push_back(x1);
 velocities.push_back(vel);
 velocities.push_back(theta);

 return velocities;
}



vector<double> FPCo(const vector<double> & v){
  
 const double CoulLog = 10.;
 const double Zb = 1.;
 const double Za = 1.;
 const double nb = 1.530e27;
 const double ee = 1.602176634e-19;
 const double ma = 9.1093837015e-31;
 const double mb = 9.1093837015e-31;
 const double vb = sqrt(2*T_f*ee/mb); //plasma thermal velocity given by temperature in eV // ee to convert
 const double va = v_max;          
 const double gama = 4.0*pi*pow(Za,4.0)*pow(ee,4.0)/pow(ma,2.0);
 double x0,x1,theta,vel;
 x0=v[0]*va;
 x1=v[1]*va;
 vel=v[2]*va;
 theta=v[3];
 double A = pow(Zb/Za,2.0)*CoulLog*ma/mb*nb*(-vel/vb*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))+erf(vel/sqrt(2.0)/vb));
 double B = pow(vb,2.0)*mb/vel/ma*A;
 double FWoSin = pow(Zb/Za,2.0)*CoulLog*nb/2.0/vel*(vb/vel*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))+(1.0-pow(vb,2.0)/pow(vel,2.0))*erf(vel/sqrt(2.0)/vb));
 double F = FWoSin*sin(theta);
 double dA = pow(Zb/Za,2.0)*CoulLog*ma/mb*nb*pow(vel,2.0)/pow(vb,3.0)*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0));
 double base, x, y, xx, xy, yx, yy;
 base = dA/vel*sin(theta);

 x = A*sin(theta)*cos(theta)/vel;
 y = A*sin(theta)*sin(theta)/vel;

 xx = B*sin(theta)*cos(theta)*cos(theta)/vel+vel*F*sin(theta)*sin(theta);
 xy = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta);
 yx = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta);
 yy = B*sin(theta)*sin(theta)*sin(theta)/vel+vel*F*cos(theta)*cos(theta);

 vector<double> coefficients;

 coefficients.reserve(10);
 coefficients.push_back(1.0*gama*base);
 coefficients.push_back(1.0*gama*x/va);
 coefficients.push_back(1.0*gama*y/va);
 coefficients.push_back(-1.*gama*xx/va/va);
 coefficients.push_back(-1.*gama*xy/va/va);
 coefficients.push_back(-1.*gama*yx/va/va);
 coefficients.push_back(-1.*gama*yy/va/va);


 return coefficients;

}

double InitialDist(const Vector & v)
{
  
  double vel = sqrt(pow(v[0],2.0)+pow(v[1],2.0))*v_max;
  const double ee = 1.602176634e-19; //J/eV
  const double mass = 9.1093837015e-31; //electron mass in kg
  const double T = T_0/k; // initial temp in Kelvin 
  double f = pow(mass/pi/2.0/k/T/ee,3.0/2.0)*exp(-1.0*mass*vel*vel/2.0/k/T/ee);
  return f;
  
  
}

double Jacobian(const Vector & v)
{
  return 2.0*pi*v[1]*pow(v_max,3.0);
}
