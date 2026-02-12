#include <curve-energy/bending_energy.h>
#include <curve-energy/membrane_energy.h>
#include <curve-energy/curve_energy.h>

#include <curve-energy/auxiliary.h>

int main(  ) {
  int numVertices = 5;
  int numEdges = 4;
  // Data storage of meshes
  Eigen::MatrixXd V1;
  V1 = Eigen::MatrixXd::Zero(numVertices,3);
  V1(1,0) =  0.8723510161150685;
  V1(1,1) = 0.5118696929498748;
  V1(2,0) = 1.9817743198576525;
  V1(2,1) =  0.833924420257607;
  V1(3,0) = 3.3013734912641786;
  V1(3,1) = 0.42312004657397906;
  V1(4,0) = 4.;

  Eigen::MatrixXd V2;
  V2 = Eigen::MatrixXd::Zero(numVertices,3);
  V2(1,0) =  1.;
  V2(2,0) = 2;
  V2(3,0) = 3.;
  V2(4,0) = 4.;

  Eigen::MatrixXi F1;
  F1 = Eigen::MatrixXi::Zero(numEdges,2);
  F1(0,0) = 0;
  F1(0,1) = 1;
  F1(1,0) = 1;
  F1(1,1) = 2;
  F1(2,0) = 2;
  F1(2,1) = 3;
  F1(3,0) = 3;
  F1(3,1) = 4;

  // normal of planar curve plane 
  Eigen::Vector3d normal;
  normal(0) = 0;
  normal(1) = 0;
  normal(2) = 1; 

  Eigen::VectorXd out;

  // Output number and dimension of vertices
  std::cout << " .. V1 = " << V1.rows() << " x " << V1.cols() << std::endl;
  std::cout << " .. V2 = " << V2.rows() << " x " << V2.cols() << std::endl;

  std::cout << " --- MEMBRANE --- " << std::endl;
  std::cout << " .. membrane_energy = " << curve::membrane_energy( V1, V2, F1 ) << std::endl;
  std::cout << " --- BENDING --- " << std::endl;

  std::cout << " .. bending_energy = " << curve::bending_energy( V1, V2, F1, normal ) << std::endl;
 
  std::cout << " --- BENDING GRADIENT --- " << std::endl; 

  //3n vector, first all x coords, then y, then z
  curve::bending_undeformed_gradient( V1, V2, F1, normal, out );
  std::cout << " .. bending_undeformed_gradient = " << out << std::endl;
  
  out.setZero();
  curve::bending_deformed_gradient( V1, V2, F1, normal, out );
  std::cout << " .. bending_deformed_gradient = " << out << std::endl;
  double bendingWeight = 0.001;
  double membraneWeight = 2;
  std::cout << "--- FULL ENERGY ---" << std::endl;
  std::cout << ".. " << curve::curve_energy(V1, V2, F1, normal, bendingWeight, membraneWeight) << std::endl;

  Eigen::Matrix3d  test;
  test.setZero();
  test.diagonal().array() += 5;
  std::cout << test << std::endl;
  

}
