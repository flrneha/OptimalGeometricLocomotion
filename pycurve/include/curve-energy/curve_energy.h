#ifndef CURVE_CURVE_ENERGY_H
#define CURVE_CURVE_ENERGY_H

#include "bending_energy.h"
#include "membrane_energy.h"

namespace curve{
    double curve_energy(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, const Eigen::MatrixXi &Edge, 
                          const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight);

    void curve_deformed_gradient(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                   const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight, Eigen::VectorXd &grad);
    
    void curve_undeformed_gradient(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                     const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight,double membraneWeight, Eigen::VectorXd &grad);

    void curve_deformed_hessian(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                  const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, 
                                  double bendingWeight, double membraneWeight, Eigen::SparseMatrix<double> &Hess);
    
    void curve_undeformed_hessian(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                    const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight,
                                    Eigen::SparseMatrix<double> &Hess);   

    void curve_mixed_hessian(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, 
                                double bendingWeight,double membraneWeight, Eigen::SparseMatrix<double> &Hess, bool FirstDerivWRTDef=true);
                               
}
#endif //CURVE_CURVE_ENERGY_H