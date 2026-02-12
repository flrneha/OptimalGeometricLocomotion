#include "curve-energy/bending_energy.h"
#include "curve-energy/auxiliary.h"



double curve::bending_energy( const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                              const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal ){
    
    int numEdges = Edge.rows();
    int numVertices = UndeformedGeom.rows();
    Eigen::VectorXd nodalLengths = Eigen::VectorXd::Zero( numVertices );

    std::vector<std::vector<int>> adjacentEdges, adjacentEdgesOther;
    std::vector<int> interiorVertices;
    getAdjacentEdges(numVertices, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices);

    for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {

      // Undeformed Length
      Eigen::Vector3d p0, p1;
      p0 = UndeformedGeom.row( Edge( edgeIdx,0 ) );
      p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );

      Eigen::Vector3d edgeVec = p0 - p1;
      double undefEdgeLength = edgeVec.norm();

      nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
      nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
    }

    double Dest = 0.;
    Eigen::VectorXd angleDiff = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal ) 
                                  - curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices,normal );

    // Iterate over interior vertices, to be replaced by proper topology
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
      Dest += angleDiff( innerVertexIdx ) * angleDiff( innerVertexIdx ) / nodalLengths( interiorVertices[ innerVertexIdx ] );
    }
    return Dest;
}

double curve::reg_bending_energy( const Eigen::MatrixXd &UndeformedGeom,const Eigen::MatrixXd &DeformedGeom, 
                              const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal ){
    
    int numEdges = Edge.rows();
    int numVertices = UndeformedGeom.rows();
    Eigen::VectorXd nodalLengths = Eigen::VectorXd::Zero( numVertices );

    std::vector<std::vector<int>> adjacentEdges, adjacentEdgesOther;
    std::vector<int> interiorVertices;
    getAdjacentEdges(numVertices, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices);

    for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {

      // Undeformed Length
      Eigen::Vector3d p0, p1;
      p0 = UndeformedGeom.row( Edge( edgeIdx,0 ) );
      p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );

      Eigen::Vector3d edgeVec = p0 - p1;
      double undefEdgeLength = edgeVec.norm();

      nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
      nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
    }

    double Dest = 0.;
    Eigen::VectorXd UndeformedAngle = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal );
    Eigen::VectorXd DeformedAngle = curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal );

    Eigen::VectorXd angleDiff = UndeformedAngle - DeformedAngle;
                                  

    // Iterate over interior vertices, to be replaced by proper topology
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
       Dest += reg_angle_weight( DeformedAngle( innerVertexIdx )  ) * angleDiff( innerVertexIdx ) * angleDiff( innerVertexIdx ) / nodalLengths( interiorVertices[ innerVertexIdx ] );
    }
    return Dest;
}