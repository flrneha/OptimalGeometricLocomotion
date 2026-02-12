
#include "curve-energy/membrane_energy.h"

double curve::membrane_energy(const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, const Eigen::MatrixXi &Edge ){

    double Dest = 0.;
    
    for ( int edgeIdx=0; edgeIdx < Edge.rows(); ++edgeIdx ) {

      // Undeformed Length
      Eigen::Vector3d p0, p1;
      p0 =  UndeformedGeom.row( Edge( edgeIdx,0 ) );
      p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );

      double undeformedLength = ( p0 - p1 ).norm();

      // Deformed Length
      p0 = DeformedGeom.row( Edge( edgeIdx,0 )  );
      p1 = DeformedGeom.row( Edge( edgeIdx,1 ) );

      double deformedLength = ( p0 - p1 ).norm();

      Dest += ( undeformedLength - deformedLength ) * ( undeformedLength - deformedLength ) /
              ( undeformedLength * undeformedLength );
    }   
    return Dest;
}
