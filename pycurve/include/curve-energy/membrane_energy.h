#ifndef CURVE_MEMBRANE_ENERGY_H
#define CURVE_MEMBRANE_ENERGY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace curve{

    double membrane_energy( const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                            const Eigen::MatrixXi &Edge );

    void membrane_undeformed_gradient(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
                                        const Eigen::MatrixXi &Edge, Eigen::VectorXd &grad );
    void membrane_deformed_gradient(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
                                        const Eigen::MatrixXi &Edge, Eigen::VectorXd &grad );
    
    void membrane_deformed_hessian( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
                                        const Eigen::MatrixXi &Edge, Eigen::SparseMatrix<double> &Hess );
    void membrane_undeformed_hessian( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
                                        const Eigen::MatrixXi &Edge, Eigen::SparseMatrix<double> &Hess );
    void membrane_mixed_hessian( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
                                    const Eigen::MatrixXi &Edge, Eigen::SparseMatrix<double> &Hess, 
                                    bool FirstDerivativeWRTDef = true );                                    
}
#endif //CURVE_MEMBRANE_ENERGY_H