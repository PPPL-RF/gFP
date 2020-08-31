#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
  const char *mesh_file = "refined.mesh";
  const char *gf_file = "sol.gf";

  OptionsParser args(argc,argv);

  args.AddOption(&mesh_file, "-m", "--mesh",
		 "Mesh file to use.");
  args.AddOption(&gf_file, "-gf", "--gridfunction",
		 "Grid function to use");

  Mesh *mesh = new Mesh(mesh_file,1,1);
  int dim = mesh->Dimension();
  //cout << dim << endl;

  ifstream sol_ofs;
  sol_ofs.open(gf_file);
  GridFunction gf = GridFunction(mesh, sol_ofs);
  Array<int> elem_ids;
  Array<IntegrationPoint> ips;
  int xDeg=500;
  int yDeg=250;
  int numPoints = xDeg*yDeg;
  DenseMatrix point_mat(2,numPoints);
  DenseMatrix points(2,numPoints);

  double vx,vy,v,theta;
  int row = 0;
  for (double j = 0; j < xDeg; j++){
    for (double i = 0; i < yDeg; i++){
      vx = (j-xDeg/2)/xDeg*2;
      vy = (i+1)/yDeg;
      theta=atan2(vy,vx);
      v=sqrt(vx*vx+vy*vy);
      //if (v<0.98){
	point_mat(0,row)=vx;
	point_mat(1,row)=vy;
	points(0,row)=v;
	points(1,row)=theta;
	row+=1;
	//}
    }
  }

  mesh->FindPoints(point_mat,elem_ids,ips);

  DenseMatrix val(1,numPoints);
  VisItDataCollection dc("gFP");
  typedef DataCollection::FieldMapType fields_t;
  const fields_t &fields = dc.GetFieldMap();


  Vector value(numPoints);
  for (int it=0; it<numPoints;it++){
    if (! (elem_ids[it]<0)){
    value(it)=gf.GetValue(elem_ids[it],ips[it]);
    }
    else {
      value[it]=0.;
    }
  }

  ofstream file ("gFPOut.txt", ios::trunc);
  file << "f v theta vx vy \n";
  int num_points=0;
    for (int it = 0; it < numPoints; it++){
      //if (0<elem_ids[it]){
      file << value(it) << " " << points(0,it) << " " << points(1,it) << " " 
	   << point_mat(0,it) << " " << point_mat(1,it) << " \n"; 
      num_points+=1;
      //}
    }

  file.close();
  cout << "number of points is " << num_points << "\n";
  
  return 0;
}
