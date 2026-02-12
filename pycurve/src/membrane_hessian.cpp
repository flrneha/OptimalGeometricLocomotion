
#include "curve-energy/membrane_energy.h"

void curve::membrane_deformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, Eigen::SparseMatrix<double> &Hess ){
    
    int numVertices = UndeformedGeom.rows();  
    int rowOffset, colOffset = 0;                            
    int factor = 1;
    
    Hess.resize( numVertices * UndeformedGeom.cols(),numVertices * UndeformedGeom.cols());
    Hess.setZero();

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve( 6 * 6 * Edge.rows());

    for ( int edgeIdx=0; edgeIdx < Edge.rows(); ++edgeIdx ) {
        // Undeformed Length
        Eigen::Vector3d p0, p1;
        p0 = UndeformedGeom.row( Edge( edgeIdx,0 ) );
        p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );
        
        double undeformedLength = ( p0 - p1 ).norm();

        // Deformed Length
        p0 = DeformedGeom.row( Edge( edgeIdx,0 )  );
        p1 = DeformedGeom.row( Edge( edgeIdx,1 ) );
        
        Eigen::Vector3d localGradient = p0 - p1;
        double deformedLength = localGradient.norm();

        localGradient /= deformedLength;

        Eigen::Matrix3d localHessian, tensorProduct;

        localHessian = localGradient * localGradient.transpose();

        tensorProduct = (p0 - p1) * ( p0 - p1 ).transpose();

        tensorProduct *= -1. / ( deformedLength * deformedLength * deformedLength );
        tensorProduct.diagonal().array() += ( 1 / deformedLength );
        tensorProduct *= undeformedLength - deformedLength ;

        localHessian -= tensorProduct;

        localHessian *= 2. / ( undeformedLength * undeformedLength );

        for ( int i : { 0, 1, 2 } ) {
        for ( int j : { 0, 1, 2 } ) {
            tripletList.emplace_back( i * numVertices +  Edge( edgeIdx,0 )+ rowOffset,
                                    j * numVertices +  Edge( edgeIdx,0 ) + colOffset,
                                    factor * localHessian( i, j ));
            tripletList.emplace_back( i * numVertices + Edge( edgeIdx,1 )+ rowOffset,
                                    j * numVertices + Edge( edgeIdx,1 ) + colOffset,
                                    factor * localHessian( i, j ));
            tripletList.emplace_back( i * numVertices +  Edge( edgeIdx,0 )+ rowOffset,
                                    j * numVertices +Edge( edgeIdx,1 ) + colOffset,
                                    -factor * localHessian( i, j ));
            tripletList.emplace_back( i * numVertices + Edge( edgeIdx,1 ) + rowOffset,
                                    j * numVertices +  Edge( edgeIdx,0 ) + colOffset,
                                    -factor * localHessian( i, j ));
        }
        }
    }
    // fill matrix from triplets
     Hess.setFromTriplets( tripletList.begin(), tripletList.end());
}
    
void curve::membrane_undeformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, Eigen::SparseMatrix<double> &Hess ){
    
    int numVertices = UndeformedGeom.rows();  
    int rowOffset, colOffset = 0;                              
    int factor = 1;
    
    Hess.resize( numVertices * UndeformedGeom.cols(),numVertices * UndeformedGeom.cols());
    Hess.setZero();

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve( 6 * 6 * Edge.rows());

    for ( int edgeIdx=0; edgeIdx < Edge.rows(); ++edgeIdx ) {
        // Undeformed Length
        Eigen::Vector3d p0, p1;
        Eigen::Matrix3d localHessian, tensorProduct;

        p0 = UndeformedGeom.row( Edge( edgeIdx, 0 ) );
        p1 = UndeformedGeom.row( Edge( edgeIdx, 1 ) );
        
        Eigen::Vector3d localGradient = p0 - p1;
        double undeformedLength = localGradient.norm();
        localGradient /= undeformedLength;
        tensorProduct = (p0 - p1) * ( p0 - p1 ).transpose();
        tensorProduct *= -1. / ( undeformedLength * undeformedLength * undeformedLength );
        tensorProduct.diagonal().array() += ( 1 / undeformedLength );

        // Deformed Length
        p0 = DeformedGeom.row( Edge( edgeIdx, 0 )  );
        p1 = DeformedGeom.row( Edge( edgeIdx, 1 ) );
        
        double deformedLength = ( p0 -p1 ).norm();

        localHessian = localGradient * localGradient.transpose();

        tensorProduct *= 2 * deformedLength / ( undeformedLength * undeformedLength ) -
                        2 * ( deformedLength * deformedLength ) /
                        ( undeformedLength * undeformedLength * undeformedLength );

        localHessian *= 6 * ( deformedLength * deformedLength ) /
                        ( undeformedLength * undeformedLength * undeformedLength * undeformedLength ) -
                        4 * deformedLength / ( undeformedLength * undeformedLength * undeformedLength );

        
        localHessian += tensorProduct;

        for ( int i : { 0, 1, 2 } ) {
            for ( int j : { 0, 1, 2 } ) {
                tripletList.emplace_back( i * numVertices +  Edge( edgeIdx,0 )+ rowOffset,
                                        j * numVertices +  Edge( edgeIdx,0 ) + colOffset,
                                        factor * localHessian( i, j ));
                tripletList.emplace_back( i * numVertices + Edge( edgeIdx,1 )+ rowOffset,
                                        j * numVertices + Edge( edgeIdx,1 ) + colOffset,
                                        factor * localHessian( i, j ));
                tripletList.emplace_back( i * numVertices +  Edge( edgeIdx,0 )+ rowOffset,
                                        j * numVertices +Edge( edgeIdx,1 ) + colOffset,
                                        -factor * localHessian( i, j ));
                tripletList.emplace_back( i * numVertices + Edge( edgeIdx,1 ) + rowOffset,
                                        j * numVertices +  Edge( edgeIdx,0 ) + colOffset,
                                        -factor * localHessian( i, j ));
            }
        }
    }
    // fill matrix from triplets
     Hess.setFromTriplets( tripletList.begin(), tripletList.end());
}

void curve::membrane_mixed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, Eigen::SparseMatrix<double> &Hess, bool FirstDerivWRTDef ){
    
    int numVertices = UndeformedGeom.rows();  
    int rowOffset, colOffset = 0;                                        
    int factor = 1;
    
    Hess.resize( numVertices * UndeformedGeom.cols(),numVertices * UndeformedGeom.cols());
    Hess.setZero();

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve( 6 * 6 * Edge.rows());

    for ( int edgeIdx=0; edgeIdx < Edge.rows(); ++edgeIdx ) {
        // Undeformed Length
        Eigen::Vector3d p0, p1;
        Eigen::Matrix3d localHessian, tensorProduct;

        p0 = UndeformedGeom.row( Edge( edgeIdx, 0 ) );
        p1 = UndeformedGeom.row( Edge( edgeIdx, 1 ) );
        
        Eigen::Vector3d localUndefGradient = p0 - p1;
        double undeformedLength = localUndefGradient.norm();
        localUndefGradient /= undeformedLength;

        // Deformed Length
        p0 = DeformedGeom.row( Edge( edgeIdx, 0 )  );
        p1 = DeformedGeom.row( Edge( edgeIdx, 1 ) );
        
        Eigen::Vector3d localDefGradient = p0 - p1;
        double deformedLength = localDefGradient.norm();
        localDefGradient /=deformedLength;
        
        if( !FirstDerivWRTDef)
            localHessian = localUndefGradient * localDefGradient.transpose();
        else
            localHessian = localDefGradient * localUndefGradient.transpose();

        localHessian *= 2 * ( undeformedLength - 2 * deformedLength ) /
                      ( undeformedLength * undeformedLength * undeformedLength );

        for ( int i : { 0, 1, 2 } ) {
            for ( int j : { 0, 1, 2 } ) {
                tripletList.emplace_back( i * numVertices +  Edge( edgeIdx,0 )+ rowOffset,
                                        j * numVertices +  Edge( edgeIdx,0 ) + colOffset,
                                        factor * localHessian( i, j ));
                tripletList.emplace_back( i * numVertices + Edge( edgeIdx,1 )+ rowOffset,
                                        j * numVertices + Edge( edgeIdx,1 ) + colOffset,
                                        factor * localHessian( i, j ));
                tripletList.emplace_back( i * numVertices +  Edge( edgeIdx,0 )+ rowOffset,
                                        j * numVertices +Edge( edgeIdx,1 ) + colOffset,
                                        -factor * localHessian( i, j ));
                tripletList.emplace_back( i * numVertices + Edge( edgeIdx,1 ) + rowOffset,
                                        j * numVertices +  Edge( edgeIdx,0 ) + colOffset,
                                        -factor * localHessian( i, j ));
            }
        }
    }
    // fill matrix from triplets
     Hess.setFromTriplets( tripletList.begin(), tripletList.end());
}