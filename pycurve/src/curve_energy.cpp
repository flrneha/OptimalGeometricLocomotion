#include "curve-energy/curve_energy.h"
#include "curve-energy/bending_energy.h"
#include "curve-energy/membrane_energy.h"

namespace curve{
    double curve_energy(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, const Eigen::MatrixXi &Edge, 
                          const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight) {
        return membraneWeight*membrane_energy(UndeformedGeom, DeformedGeom, Edge) + bendingWeight*bending_energy(UndeformedGeom, DeformedGeom, Edge, normal);
    }

    void curve_deformed_gradient(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                   const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight, Eigen::VectorXd &grad){
        membrane_deformed_gradient( UndeformedGeom, DeformedGeom, Edge, grad);
        grad*= membraneWeight;
        Eigen::VectorXd tmp;
        bending_deformed_gradient( UndeformedGeom, DeformedGeom, Edge, normal, tmp );
        grad += bendingWeight * tmp;
    }
    
    void curve_undeformed_gradient(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                     const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight,double membraneWeight, Eigen::VectorXd &grad){
        membrane_undeformed_gradient( UndeformedGeom, DeformedGeom, Edge, grad);
        grad *= membraneWeight;
        Eigen::VectorXd tmp;
        bending_undeformed_gradient( UndeformedGeom, DeformedGeom, Edge, normal, tmp );
        grad += bendingWeight * tmp;
    }
    
    void curve_deformed_hessian(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                    const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, 
                                    double bendingWeight, double membraneWeight, Eigen::SparseMatrix<double> &Hess){
        
        membrane_deformed_hessian(UndeformedGeom, DeformedGeom, Edge, Hess);
        Eigen::SparseMatrix<double> tmp;
        Hess *= membraneWeight;
        bending_deformed_hessian(UndeformedGeom, DeformedGeom, Edge, normal, tmp);
        Hess +=bendingWeight * tmp;
    }

    void curve_undeformed_hessian(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                    const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight,
                                    Eigen::SparseMatrix<double> &Hess){
        
        membrane_undeformed_hessian(UndeformedGeom, DeformedGeom, Edge, Hess);
        Hess *= membraneWeight;
        Eigen::SparseMatrix<double> tmp;
        bending_undeformed_hessian(UndeformedGeom, DeformedGeom, Edge, normal, tmp);
        Hess +=bendingWeight * tmp;
    }


    void curve_mixed_hessian(  const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                                    const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, 
                                    double bendingWeight, double membraneWeight, Eigen::SparseMatrix<double> &Hess, bool FirstDerivWRTDef){
        
        membrane_mixed_hessian(UndeformedGeom, DeformedGeom, Edge, Hess, FirstDerivWRTDef);
        Hess *= membraneWeight;
        Eigen::SparseMatrix<double> tmp;
        bending_mixed_hessian(UndeformedGeom, DeformedGeom, Edge, normal, tmp, FirstDerivWRTDef);
        Hess +=bendingWeight * tmp;
    }
    
    
}