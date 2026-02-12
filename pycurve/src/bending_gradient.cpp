  
#include "curve-energy/bending_energy.h"
#include "curve-energy/auxiliary.h"

  
void curve::bending_undeformed_gradient(   const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, Eigen::VectorXd &grad ){

    int numVertices = UndeformedGeom.rows();  
    int numEdges = Edge.rows();                                            
    grad.resize( 3 * numVertices );
    grad.setZero();

    std::vector<Eigen::Vector3d> undefEdgeDirections;
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
        
        undefEdgeDirections.emplace_back( edgeVec.normalized() ) ;

        nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
        nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
    }

    Eigen::VectorXd innerNodalLengths = Eigen::VectorXd::Zero( interiorVertices.size() );
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ )
        innerNodalLengths( innerVertexIdx ) = nodalLengths( interiorVertices[ innerVertexIdx ] );

    Eigen::VectorXd angleDiff = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal ) 
                                    - curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal );
    Eigen::VectorXd pointwiseValue = angleDiff.array() / innerNodalLengths.array();

    // Derivative of angles
    Eigen::MatrixXd dTheta = DcurveAngles( UndeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );

    grad = 2 * dTheta.transpose() * pointwiseValue;

    // Derivative of Lengths
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
        const int vertexIdx = interiorVertices[innerVertexIdx];
        const int leftEdgeIdx = adjacentEdges[vertexIdx][0];
        const int leftEdgeIdxOther = adjacentEdgesOther[vertexIdx][0];
        double leftEdgeSign = (leftEdgeIdxOther == 0) ? 1 : -1;

        const int rightEdgeIdx = adjacentEdges[vertexIdx][1];
        const int rightEdgeIdxOther = adjacentEdgesOther[vertexIdx][1];
        double rightEdgeSign = (rightEdgeIdxOther == 1) ? 1 : -1;

        // Left
        Eigen::Vector3d localGradient =
                undefEdgeDirections[leftEdgeIdx] * pointwiseValue[innerVertexIdx] * pointwiseValue[innerVertexIdx] * rightEdgeSign;
        for ( int i : { 0, 1, 2 } )
            grad[i * numVertices + Edge( leftEdgeIdx , leftEdgeIdxOther )] -= localGradient[i];

        // Middle
        localGradient -=
                undefEdgeDirections[rightEdgeIdx] * pointwiseValue[innerVertexIdx] * pointwiseValue[innerVertexIdx] * leftEdgeSign;
        for ( int i : { 0, 1, 2 } )
            grad[i *numVertices + vertexIdx] += localGradient[i];


        // Right
        localGradient =
                undefEdgeDirections[rightEdgeIdx] * pointwiseValue[innerVertexIdx] * pointwiseValue[innerVertexIdx] * leftEdgeSign;
        for ( int i : { 0, 1, 2 } )
            grad[i * numVertices + Edge( rightEdgeIdx, rightEdgeIdxOther ) ] += localGradient[i];
    }
}

void curve::bending_deformed_gradient(   const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, Eigen::VectorXd &grad ){

    int numVertices = UndeformedGeom.rows();  
    int numEdges = Edge.rows();                                            
    grad.resize( 3 * numVertices );
    grad.setZero();

    std::vector<Eigen::Vector3d> undefEdgeDirections;
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
        
        undefEdgeDirections.emplace_back( edgeVec.normalized() ) ;

        nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
        nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
    }

    Eigen::VectorXd innerNodalLengths = Eigen::VectorXd::Zero( interiorVertices.size() );
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ )
        innerNodalLengths( innerVertexIdx ) = nodalLengths( interiorVertices[ innerVertexIdx ] );

    Eigen::VectorXd angleDiff = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal ) 
                                    - curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal );
    Eigen::VectorXd pointwiseValue = -2. * angleDiff.array() / innerNodalLengths.array();

    // Derivative of angles
    Eigen::MatrixXd dTheta = DcurveAngles( DeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );

    grad = dTheta.transpose() * pointwiseValue;

}

void curve::reg_bending_deformed_gradient(   const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                            const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, Eigen::VectorXd &grad ){

    int numVertices = UndeformedGeom.rows();  
    int numEdges = Edge.rows();                                            
    grad.resize( 3 * numVertices );
    grad.setZero();

    std::vector<Eigen::Vector3d> undefEdgeDirections;
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
        
        undefEdgeDirections.emplace_back( edgeVec.normalized() ) ;

        nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
        nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
    }
    Eigen::VectorXd UndeformedAngle = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal );
    Eigen::VectorXd DeformedAngle = curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal );

    Eigen::VectorXd innerNodalLengths = Eigen::VectorXd::Zero( interiorVertices.size() );
    Eigen::VectorXd weights = Eigen::VectorXd::Zero( interiorVertices.size() );
    Eigen::VectorXd Dweights = Eigen::VectorXd::Zero( interiorVertices.size() );
    Eigen::VectorXd angleDiff = UndeformedAngle - DeformedAngle;

    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ){
        innerNodalLengths( innerVertexIdx ) = nodalLengths( interiorVertices[ innerVertexIdx ] );
        weights( innerVertexIdx ) = reg_angle_weight(DeformedAngle( innerVertexIdx )  );
        Dweights( innerVertexIdx ) = Dreg_angle_weight( DeformedAngle( innerVertexIdx )  );
    }                                 

    Eigen::VectorXd pointwiseValue = -2. * weights.array() * angleDiff.array() / innerNodalLengths.array();
    Eigen::VectorXd DpointwiseValue = Dweights.array() * angleDiff.array() * angleDiff.array() / (innerNodalLengths.array());

    // Derivative of angles
    Eigen::MatrixXd dTheta = DcurveAngles( DeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );
    
    //Derivative of regularizing weights

    grad = dTheta.transpose() * (DpointwiseValue + pointwiseValue);

}