#ifndef CURVE_BENDING_ENERGY_H
#define CURVE_BENDING_ENERGY_H

#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace curve{
    double bending_energy(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                            const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal );

    double reg_bending_energy( const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                              const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal );



    void bending_undeformed_gradient(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
                                        const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, Eigen::VectorXd &grad );

    void bending_deformed_gradient(   const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, Eigen::VectorXd &grad );
    void reg_bending_deformed_gradient(   const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, Eigen::VectorXd &grad );

    void bending_deformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                        const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, Eigen::SparseMatrix<double> &Hess );
                                        
    void reg_bending_deformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                       const Eigen::MatrixXi &Edge,const Eigen::Vector3d &normal, 
                                       Eigen::SparseMatrix<double> &Hess )  ;  
 
    void bending_undeformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                        const Eigen::MatrixXi &Edge,const Eigen::Vector3d &normal, Eigen::SparseMatrix<double> &Hess );

    void bending_mixed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                         const Eigen::MatrixXi &Edge,const Eigen::Vector3d &normal, 
                                         Eigen::SparseMatrix<double> &Hess,bool FirstDerivWRTDef=true );

}

#endif //CURVE_BENDING_ENERGY_H