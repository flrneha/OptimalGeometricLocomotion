#include "curve-energy/bending_energy.h"
#include "curve-energy/auxiliary.h"


void curve::bending_deformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                       const Eigen::MatrixXi &Edge,const Eigen::Vector3d &normal, 
                                       Eigen::SparseMatrix<double> &Hess ){
    
    int numVertices = UndeformedGeom.rows();  
    int numEdges = Edge.rows();   

    std::vector<Eigen::Triplet<double>> triplets;

    Hess.resize( numVertices *3,numVertices *3);
    Hess.setZero();

    std::vector<std::vector<int>> adjacentEdges, adjacentEdgesOther;
    std::vector<int> interiorVertices;
    getAdjacentEdges(numVertices, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices);
    
    Eigen::VectorXd nodalLengths = Eigen::VectorXd::Zero( numVertices );

    for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {
        // Undeformed Length
        Eigen::Vector3d p0, p1;
        p0 = UndeformedGeom.row( Edge( edgeIdx,0 ) );
        p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );
    
        double undefEdgeLength = (p0-p1).norm();
        nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
        nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
    }       
    
    std::vector<Eigen::Triplet<double>> diagonalTriplets;
    Eigen::VectorXd innerNodalLengths = Eigen::VectorXd::Zero( interiorVertices.size() );
    
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ){
        innerNodalLengths( innerVertexIdx ) = nodalLengths( interiorVertices[ innerVertexIdx ] );   
        diagonalTriplets.emplace_back( innerVertexIdx, innerVertexIdx,
                                     1. / nodalLengths( interiorVertices[innerVertexIdx] ) );
    }

    Eigen::VectorXd angleDiff = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal ) 
                                    - curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal );    
    
    Eigen::VectorXd pointwiseValue = -2. * angleDiff.array() / innerNodalLengths.array();

    Eigen::SparseMatrix<double> dTheta = DcurveAngles( DeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );

    Eigen::SparseMatrix<double> dM(interiorVertices.size(), interiorVertices.size());
    dM.setFromTriplets(diagonalTriplets.begin(), diagonalTriplets.end());
    
    Eigen::SparseMatrix<double> dTheta_squared = dTheta.transpose() * dM * dTheta;

    for ( int k = 0; k < dTheta_squared.outerSize(); ++k ) {
      for ( typename Eigen::SparseMatrix<double>::InnerIterator it( dTheta_squared, k ); it; ++it ) {
        triplets.emplace_back( it.row(), it.col(), 2. * it.value());
      }
    }  
    
    std::vector<std::vector<Eigen::Triplet<double>>> inputTriplets;
    if ( inputTriplets.size() != interiorVertices.size() )
    inputTriplets.resize( interiorVertices.size() );  

    D2curveAngles( DeformedGeom, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices, normal, inputTriplets );

    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
      double localFactor = pointwiseValue[innerVertexIdx];
      for ( const auto &triplet : inputTriplets[innerVertexIdx] ) {
        triplets.emplace_back( triplet.row(), triplet.col(), localFactor* triplet.value());
      }
    }   
    Hess.setFromTriplets(triplets.begin(), triplets.end());

}

void curve::reg_bending_deformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                       const Eigen::MatrixXi &Edge,const Eigen::Vector3d &normal, 
                                       Eigen::SparseMatrix<double> &Hess ){
    
    int numVertices = UndeformedGeom.rows();  
    int numEdges = Edge.rows();   

    std::vector<Eigen::Triplet<double>> triplets;

    Hess.resize( numVertices *3,numVertices *3);
    Hess.setZero();

    std::vector<std::vector<int>> adjacentEdges, adjacentEdgesOther;
    std::vector<int> interiorVertices;
    getAdjacentEdges(numVertices, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices);
    
    Eigen::VectorXd nodalLengths = Eigen::VectorXd::Zero( numVertices );

    for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {
        // Undeformed Length
        Eigen::Vector3d p0, p1;
        p0 = UndeformedGeom.row( Edge( edgeIdx,0 ) );
        p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );
    
        double undefEdgeLength = (p0-p1).norm();
        nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
        nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
    }       
    
    std::vector<Eigen::Triplet<double>> diagonalTriplets;
    std::vector<Eigen::Triplet<double>> diagonalTripletsDweights;
    std::vector<Eigen::Triplet<double>> diagonalTripletsD2weights;
    Eigen::VectorXd innerNodalLengths = Eigen::VectorXd::Zero( interiorVertices.size() );
      
    Eigen::VectorXd UndeformedAngle = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal );
    Eigen::VectorXd DeformedAngle = curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal );

    Eigen::VectorXd weights = Eigen::VectorXd::Zero( interiorVertices.size() );
    Eigen::VectorXd Dweights = Eigen::VectorXd::Zero( interiorVertices.size() );
    Eigen::VectorXd D2weights = Eigen::VectorXd::Zero( interiorVertices.size() );

    Eigen::VectorXd angleDiff = UndeformedAngle - DeformedAngle;
 
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ){
        innerNodalLengths( innerVertexIdx ) = nodalLengths( interiorVertices[ innerVertexIdx ] );
        weights( innerVertexIdx ) = reg_angle_weight(  DeformedAngle( innerVertexIdx )  );
        Dweights( innerVertexIdx ) = Dreg_angle_weight(  DeformedAngle( innerVertexIdx )  );   
        D2weights( innerVertexIdx) = D2reg_angle_weight(  DeformedAngle( innerVertexIdx )  );

        diagonalTriplets.emplace_back( innerVertexIdx, innerVertexIdx,
                                     weights( innerVertexIdx ) / nodalLengths( interiorVertices[innerVertexIdx] ) ); //term 5 
        diagonalTripletsDweights.emplace_back( innerVertexIdx, innerVertexIdx,
                                     - Dweights( innerVertexIdx ) * angleDiff( innerVertexIdx ) 
                                     / (nodalLengths( interiorVertices[ innerVertexIdx ] ))); //term 3 and 4
        diagonalTripletsD2weights.emplace_back( innerVertexIdx, innerVertexIdx,
                                     D2weights( innerVertexIdx ) * angleDiff( innerVertexIdx ) * angleDiff( innerVertexIdx ) 
                                     / (nodalLengths( interiorVertices[ innerVertexIdx ] ))); //term 1
    }
 
    Eigen::VectorXd pointwiseValue = -2. * weights.array() * angleDiff.array() / innerNodalLengths.array(); //term6
    Eigen::VectorXd pointwiseValueDWeight = Dweights.array() * angleDiff.array() * angleDiff.array() / ( innerNodalLengths.array()); //term2

    Eigen::SparseMatrix<double> dTheta = DcurveAngles( DeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );

    Eigen::SparseMatrix<double> dM(interiorVertices.size(), interiorVertices.size()); //term 5
    Eigen::SparseMatrix<double> dMD2(interiorVertices.size(), interiorVertices.size()); //term1
    Eigen::SparseMatrix<double> dMD(interiorVertices.size(), interiorVertices.size()); //term4 and 3

    dM.setFromTriplets(diagonalTriplets.begin(), diagonalTriplets.end()); //term5
    dMD.setFromTriplets(diagonalTripletsDweights.begin(), diagonalTripletsDweights.end()); //term4 and 3

    dMD2.setFromTriplets(diagonalTripletsD2weights.begin(), diagonalTripletsD2weights.end()); //term1

    Eigen::SparseMatrix<double> dTheta_squared = dTheta.transpose() * dM * dTheta; //term 5
    Eigen::SparseMatrix<double> dTheta_Dweight = dTheta.transpose() * dMD * dTheta; //term  4 and 3
    Eigen::SparseMatrix<double> dTheta_D2Weight = dTheta.transpose() * dMD2 * dTheta; //term 1

    for ( int k = 0; k < dTheta_squared.outerSize(); ++k ) {
      for ( typename Eigen::SparseMatrix<double>::InnerIterator it( dTheta_squared, k ); it; ++it ) {
        triplets.emplace_back( it.row(), it.col(), 2. * it.value()); //term 5
      }
    }  
    for ( int k = 0; k < dTheta_Dweight.outerSize(); ++k ) {
      for ( typename Eigen::SparseMatrix<double>::InnerIterator it( dTheta_D2Weight, k ); it; ++it ) {
        triplets.emplace_back( it.row(), it.col(),  it.value()); //term 1
      }
    }  
    for ( int k = 0; k < dTheta_D2Weight.outerSize(); ++k ) {
      for ( typename Eigen::SparseMatrix<double>::InnerIterator it( dTheta_Dweight, k ); it; ++it ) {
        triplets.emplace_back( it.row(), it.col(), 4. * it.value()); //term 4 and 3
      }
    }  
    std::vector<std::vector<Eigen::Triplet<double>>> inputTriplets;
    if ( inputTriplets.size() != interiorVertices.size() )
    inputTriplets.resize( interiorVertices.size() );  

    D2curveAngles( DeformedGeom, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices, normal, inputTriplets );

    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
      double localFactor = pointwiseValue[innerVertexIdx];
      double localFactorWeight = pointwiseValueDWeight[innerVertexIdx];

      for ( const auto &triplet : inputTriplets[innerVertexIdx] ) {
        triplets.emplace_back( triplet.row(), triplet.col(), localFactor* triplet.value()); //term6
        triplets.emplace_back( triplet.row(), triplet.col(), localFactorWeight* triplet.value()); //term2
      }
    }   
    Hess.setFromTriplets(triplets.begin(), triplets.end());

}

void curve::bending_undeformed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                         const Eigen::MatrixXi &Edge,const Eigen::Vector3d &normal, 
                                         Eigen::SparseMatrix<double> &Hess ){
                            
    
    int numVertices = UndeformedGeom.rows();  
    int numEdges = Edge.rows();   

    std::vector<Eigen::Triplet<double>> triplets;

    Hess.resize( numVertices *3,numVertices *3);
    Hess.setZero();

    std::vector<std::vector<int>> adjacentEdges, adjacentEdgesOther;
    std::vector<int> interiorVertices;
    getAdjacentEdges(numVertices, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices);
    
    Eigen::VectorXd nodalLengths = Eigen::VectorXd::Zero( numVertices );    

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

    Eigen::VectorXd innerNodalLengths = Eigen::VectorXd::Zero( interiorVertices.size() );
    
    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ){
        innerNodalLengths( innerVertexIdx ) = nodalLengths( interiorVertices[ innerVertexIdx ] );   
    }

    Eigen::VectorXd angleDiff = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal ) 
                                    - curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal ); 

    Eigen::VectorXd pointwiseValue = angleDiff.array() / innerNodalLengths.array();
    
    // Mixed first derivatives of angles and lengths
    std::vector<Eigen::Triplet<double>> AngleLengthTriplets, AngleAngleTriplets, LengthLengthTriplets;

    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {

      AngleLengthTriplets.emplace_back( innerVertexIdx, adjacentEdges[interiorVertices[innerVertexIdx]][0],
                                          -2. * pointwiseValue[innerVertexIdx] / innerNodalLengths( innerVertexIdx ) );

      AngleLengthTriplets.emplace_back( innerVertexIdx, adjacentEdges[interiorVertices[innerVertexIdx]][1],
                                          -2. *  pointwiseValue[innerVertexIdx] / innerNodalLengths( innerVertexIdx ) );

      AngleAngleTriplets.emplace_back( innerVertexIdx, innerVertexIdx,
                                         2. / nodalLengths[interiorVertices[innerVertexIdx]] );
      for ( int i : { 0, 1 } )
        for ( int j : { 0, 1 } )
          LengthLengthTriplets.emplace_back( adjacentEdges[interiorVertices[innerVertexIdx]][i],
                                             adjacentEdges[interiorVertices[innerVertexIdx]][j],
                                             2. * pointwiseValue[innerVertexIdx] * pointwiseValue[innerVertexIdx] /
                                             innerNodalLengths[innerVertexIdx] );
    }

    Eigen::SparseMatrix<double> AngleLengthMatrix( interiorVertices.size(), numEdges );
    AngleLengthMatrix.setFromTriplets( AngleLengthTriplets.begin(), AngleLengthTriplets.end());

    Eigen::SparseMatrix<double> AngleAngleMatrix( interiorVertices.size(), interiorVertices.size());
    AngleAngleMatrix.setFromTriplets( AngleAngleTriplets.begin(), AngleAngleTriplets.end());

    Eigen::SparseMatrix<double>LengthLengthMatrix( numEdges, numEdges);
    LengthLengthMatrix.setFromTriplets( LengthLengthTriplets.begin(), LengthLengthTriplets.end());

    Eigen::SparseMatrix<double>  dTheta = DcurveAngles( UndeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );
    Eigen::SparseMatrix<double> dLengths = DcurveLength( UndeformedGeom, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices, normal );

    Eigen::SparseMatrix<double>  mixedTerms(3 * numVertices, 3 * numVertices);
    mixedTerms = dTheta.transpose() * AngleAngleMatrix * dTheta;
    mixedTerms += dLengths.transpose() * LengthLengthMatrix * dLengths;
    mixedTerms += dTheta.transpose() * AngleLengthMatrix * dLengths;
    mixedTerms += dLengths.transpose() * AngleLengthMatrix.transpose() * dTheta;

    for ( int k = 0; k < mixedTerms.outerSize(); ++k ) {
      for ( typename Eigen::SparseMatrix<double>::InnerIterator it( mixedTerms, k ); it; ++it ) {
        triplets.emplace_back( it.row(), it.col(), it.value());
      }
    }

    // Second derivative of lengths
    std::vector<std::vector<Eigen::Triplet<double>>> inputTriplets;
    if ( inputTriplets.size() != numEdges)
      inputTriplets.resize( numEdges );  
    D2curveLength( UndeformedGeom, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices, normal, inputTriplets );

    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
      double localFactor = -pointwiseValue[innerVertexIdx] * pointwiseValue[innerVertexIdx];
      for ( const auto &triplet : inputTriplets[adjacentEdges[interiorVertices[innerVertexIdx]][0]] ) {
        triplets.emplace_back( triplet.row() , triplet.col(), localFactor * triplet.value());
      }
      for ( const auto &triplet : inputTriplets[adjacentEdges[interiorVertices[innerVertexIdx]][1]] ) {
        triplets.emplace_back( triplet.row(), triplet.col(), localFactor * triplet.value());
      }
    }

    // Second derivative of angles
    inputTriplets.clear();
    D2curveAngles( UndeformedGeom, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices,normal, inputTriplets );

    for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
      double localFactor = 2. * pointwiseValue[innerVertexIdx];
      for ( const auto &triplet : inputTriplets[innerVertexIdx] ) {
        triplets.emplace_back( triplet.row(), triplet.col(), localFactor * triplet.value());
      }
    }
    Hess.setFromTriplets(triplets.begin(), triplets.end());

}


void curve::bending_mixed_hessian(  const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom, 
                                         const Eigen::MatrixXi &Edge,const Eigen::Vector3d &normal,
                                         Eigen::SparseMatrix<double> &Hess, bool FirstDerivWRTDef ){

  int numVertices = UndeformedGeom.rows();  
  int numEdges = Edge.rows();   

  std::vector<Eigen::Triplet<double>> triplets;

  Hess.resize( numVertices *3,numVertices *3);
  Hess.setZero();

  std::vector<std::vector<int>> adjacentEdges, adjacentEdgesOther;
  std::vector<int> interiorVertices;
  getAdjacentEdges(numVertices, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices);
  
  Eigen::VectorXd nodalLengths = Eigen::VectorXd::Zero( numVertices );    

  for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {
      // Undeformed Length
      Eigen::Vector3d p0, p1;
      p0 = UndeformedGeom.row( Edge( edgeIdx,0 ) );
      p1 = UndeformedGeom.row( Edge( edgeIdx,1 ) );

      double undefEdgeLength = (p0 -p1).norm();
  
      nodalLengths( Edge( edgeIdx,0 ) ) += undefEdgeLength;
      nodalLengths( Edge( edgeIdx,1 ) ) += undefEdgeLength;
  }   
                      
  Eigen::VectorXd angleDiff = curveAngles( UndeformedGeom, Edge, adjacentEdges, interiorVertices, normal ) 
                                    - curveAngles( DeformedGeom, Edge, adjacentEdges, interiorVertices, normal ); 

  std::vector<Eigen::Triplet<double>>  mixedDiagonalTriplets, sameDiagonalTriplets;
  Eigen::VectorXd innerNodalLengths = Eigen::VectorXd::Zero( interiorVertices.size());
  for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
    innerNodalLengths[innerVertexIdx] = nodalLengths[interiorVertices[innerVertexIdx]];
    mixedDiagonalTriplets.emplace_back( innerVertexIdx, adjacentEdges[interiorVertices[innerVertexIdx]][0],
                                        2. * angleDiff[innerVertexIdx] /
                                        ( nodalLengths[interiorVertices[innerVertexIdx]] *
                                          nodalLengths[interiorVertices[innerVertexIdx]] ));

    mixedDiagonalTriplets.emplace_back( innerVertexIdx, adjacentEdges[interiorVertices[innerVertexIdx]][1],
                                        2. * angleDiff[innerVertexIdx] /
                                        ( nodalLengths[interiorVertices[innerVertexIdx]] *
                                          nodalLengths[interiorVertices[innerVertexIdx]] ));

    sameDiagonalTriplets.emplace_back( innerVertexIdx, innerVertexIdx,
                                        2. / nodalLengths[interiorVertices[innerVertexIdx]] );
  }

  Eigen::SparseMatrix<double> dTheta_def = DcurveAngles(DeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );
  Eigen::SparseMatrix<double> dTheta_undef = DcurveAngles( UndeformedGeom, Edge, adjacentEdges,  adjacentEdgesOther, interiorVertices, normal );
  Eigen::SparseMatrix<double> dLengths = DcurveLength( UndeformedGeom, Edge, adjacentEdges, adjacentEdgesOther, interiorVertices, normal );

  Eigen::SparseMatrix<double> mixedDiag( interiorVertices.size(), numEdges );
  mixedDiag.setFromTriplets( mixedDiagonalTriplets.begin(), mixedDiagonalTriplets.end());
  Eigen::SparseMatrix<double> sameDiag( interiorVertices.size(), interiorVertices.size());
  sameDiag.setFromTriplets( sameDiagonalTriplets.begin(), sameDiagonalTriplets.end());

  Eigen::SparseMatrix<double> mixedTerms( 3 * numVertices, 3 * numVertices );
  mixedTerms.setZero();
  if ( FirstDerivWRTDef )
    mixedTerms -= dTheta_def.transpose() * sameDiag * dTheta_undef;
  else
    mixedTerms -= dTheta_undef.transpose() * sameDiag * dTheta_def;

  if ( FirstDerivWRTDef )
    mixedTerms += dTheta_def.transpose() * mixedDiag * dLengths;
  else
    mixedTerms += dLengths.transpose() * mixedDiag.transpose() * dTheta_def;

  for ( int k = 0; k < mixedTerms.outerSize(); ++k ) {
    for ( typename Eigen::SparseMatrix<double>::InnerIterator it( mixedTerms, k ); it; ++it ) {
      triplets.emplace_back( it.row(), it.col(),  it.value());
    }
  }

  Hess.setFromTriplets(triplets.begin(), triplets.end());


}