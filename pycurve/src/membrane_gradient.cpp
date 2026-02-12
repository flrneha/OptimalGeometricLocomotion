
#include "curve-energy/membrane_energy.h"

void curve::membrane_undeformed_gradient(   const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, Eigen::VectorXd &grad ){
    
    int numVertices = UndeformedGeom.rows();                                              
    grad.resize( 3 * numVertices );
    grad.setZero();

    for ( int edgeIdx=0; edgeIdx < Edge.rows(); ++edgeIdx ) {
        // Undeformed Length
        Eigen::Vector3d p0, p1;
        p0 = UndeformedGeom.row( Edge( edgeIdx,0 ) );
        p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );

        Eigen::Vector3d localGradient = p0 - p1;
        double undeformedLength = localGradient.norm();
        localGradient /= undeformedLength;

        // Deformed Length
        p0 = DeformedGeom.row( Edge( edgeIdx,0 )  );
        p1 = DeformedGeom.row( Edge( edgeIdx,1 ) );

        double deformedLength = ( p0 - p1 ).norm();

        localGradient *= -2 * deformedLength * ( undeformedLength - deformedLength ) /
                        ( undeformedLength * undeformedLength * undeformedLength );

        for ( int j : { 0, 1, 2 } ) {
            grad(j * numVertices +  Edge( edgeIdx,0 )) -= localGradient(j);
            grad(j * numVertices +  Edge( edgeIdx,1 )) += localGradient(j);
        }
    }
}

void curve::membrane_deformed_gradient( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                        const Eigen::MatrixXi &Edge, Eigen::VectorXd &grad ){
    int numVertices = UndeformedGeom.rows();                                              
    grad.resize( 3 * numVertices );
    grad.setZero();

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

        localGradient *= 2 * ( undeformedLength - deformedLength ) / ( undeformedLength * undeformedLength );

        for ( int j : { 0, 1, 2 } ) {
            grad(j * numVertices +  Edge( edgeIdx,0 )) -= localGradient(j);
            grad(j * numVertices +  Edge( edgeIdx,1 )) += localGradient(j);
        }
    }
}

