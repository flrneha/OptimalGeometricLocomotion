#ifndef CURVE_AUXILIARY_H
#define CURVE_AUXILIARY_H
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace curve{
    double reg_angle_weight( double angle );
    double Dreg_angle_weight( double angle );
    double D2reg_angle_weight( double angle );
    Eigen::VectorXd curveAngles(  const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, 
                              const std::vector<std::vector<int>>&adjacentEdges, const std::vector<int>&interiorVertices, const Eigen::Vector3d &normal);

    void getAdjacentEdges(int numVertices, const Eigen::MatrixXi &Edge, std::vector<std::vector<int>>&adjacentEdges, 
                     std::vector<std::vector<int>>&adjacentEdgesOther, std::vector<int> &interiorVertices);

    Eigen::SparseMatrix<double> DcurveAngles(  const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                                    const std::vector<std::vector<int>> &adjacentEdgesOther, const std::vector<int> &interiorVertices, const Eigen::Vector3d &normal);

    void D2curveAngles(const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                                 const std::vector<std::vector<int>>  &adjacentEdgesOther ,const std::vector<int> &interiorVertices, const Eigen::Vector3d &normal,      
                                 std::vector<std::vector<Eigen::Triplet<double>>> &tripletList );

    Eigen::SparseMatrix<double> DcurveLength(  const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                        const std::vector<std::vector<int>>  &adjacentEdgesOther ,const std::vector<int> &interiorVertices, 
                        const Eigen::Vector3d &normal );
    
    void D2curveLength( const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                        const std::vector<std::vector<int>>  &adjacentEdgesOther ,const std::vector<int> &interiorVertices, 
                        const Eigen::Vector3d &normal,std::vector<std::vector<Eigen::Triplet<double>>> &tripletList );

    void getCrossOp( const Eigen::Vector3d &a, Eigen::Matrix3d &matrix );
    
    void localToGlobal( int targetIdx, int domainIdx, int numVertices, const Eigen::Matrix3d &localDerivative,
                         std::vector<std::vector<Eigen::Triplet<double>>> &tripletList );
    
    void localToGlobal( int targetIdx, int domainIdx, int numVertices, const Eigen::Vector3d &localDerivative,
                        std::vector<Eigen::Triplet<double>> &tripletList );
    Eigen::Matrix3d tensorProduct( const Eigen::Vector3d &a, const Eigen::Vector3d &b);
}
#endif //CURVE_AUXILIARY_H
