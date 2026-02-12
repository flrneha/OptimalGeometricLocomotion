//#include <vector>
#include "curve-energy/auxiliary.h"

const double PI  =3.141592653589793238463;

namespace curve{

    double reg_angle_weight(double angle){
        //shift angles from [-PI, PI ] to [0, 2PI]

        double weight  = - std::log(angle + PI ) - std::log(2*PI - ( angle + PI)  ) + 2*std::log(PI) +1 ;

        return weight;
    }
    
    double Dreg_angle_weight(double angle){
    //shift angles from [-PI, PI ] to [0, 2PI]

        //double weight  = 1./( 2*PI - ( angle + PI ) ) - 1./( angle + PI );
        double weight = 2*angle / (PI*PI - angle*angle);

        return weight;
    }

    double D2reg_angle_weight(double angle){
    //shift angles from [-PI, PI ] to [0, 2PI]

        double weight  = 1./( ( 2*PI - ( angle + PI ) ) * ( 2*PI - ( angle + PI ) ) )  + 1./( ( angle + PI )*( angle + PI ) )  ;
        return weight;
    }

    Eigen::Matrix3d tensorProduct( const Eigen::Vector3d &a, const Eigen::Vector3d &b  ){
        return a*b.transpose();
    }

    void getCrossOp( const Eigen::Vector3d &a, Eigen::Matrix3d &matrix ) {
        matrix.setZero();
        matrix( 0, 1 ) = -a[2];
        matrix( 0, 2 ) = a[1];
        matrix( 1, 0 ) = a[2];
        matrix( 1, 2 ) = -a[0];
        matrix( 2, 0 ) = -a[1];
        matrix( 2, 1 ) = a[0];
    }

    void localToGlobal( int edgeIdx, int firstVertexIdx, int secondVertexIdx, int numVertices, const Eigen::Matrix3d &localDerivative,
                        std::vector<std::vector<Eigen::Triplet<double>>> &tripletList ) {
        for ( int i : { 0, 1, 2 } ){
            for ( int j : { 0, 1, 2 } ){
                //Eigen::Triplet<double> val( i * numVertices + firstVertexIdx,  j * numVertices + secondVertexIdx, localDerivative( i, j ));
                tripletList[edgeIdx].emplace_back( i * numVertices + firstVertexIdx,  j * numVertices + secondVertexIdx, localDerivative( i, j ));
            }
        }
    }   

    void localToGlobal( int targetIdx, int domainIdx, int numVertices, const Eigen::Vector3d &localDerivative,
                        std::vector<Eigen::Triplet<double>> &tripletList ) {
        for ( int i : { 0, 1, 2 } )
            tripletList.emplace_back( targetIdx, i * numVertices + domainIdx, localDerivative[i] );
    }  

    Eigen::VectorXd curveAngles(    const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                                    const std::vector<int> &interiorVertices, const Eigen::Vector3d &normal ){

        int numVertices = Geom.rows();
        int numEdges = Edge.rows();
        
        Eigen::VectorXd angles = Eigen::VectorXd::Zero( interiorVertices.size() );
        std::vector<Eigen::Vector3d> EdgeVecs;

        // compute edge-based properties
        for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {

            Eigen::Vector3d p0, p1;
            p0 = Geom.row( Edge( edgeIdx,0 ) );
            p1 = Geom.row( Edge( edgeIdx,1 ) );

            EdgeVecs.emplace_back(p0 - p1);
        }

        // Iterate over interior vertices, to be replaced by proper topology
        for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
            const int vertexIdx = interiorVertices[ innerVertexIdx ];
            const int leftEdgeIdx = adjacentEdges[  vertexIdx ][0];
            const int rightEdgeIdx = adjacentEdges[ vertexIdx ][1];

            Eigen::Vector3d vertexNormal = EdgeVecs[ leftEdgeIdx ].cross( EdgeVecs[ rightEdgeIdx ] );
            angles( innerVertexIdx ) = std::atan2( normal.dot(vertexNormal), EdgeVecs[ leftEdgeIdx ].dot(EdgeVecs[ rightEdgeIdx ]));
        }

        return angles;
    }
    
    void getAdjacentEdges(  int numVertices, const Eigen::MatrixXi &Edge, std::vector<std::vector<int>> &adjacentEdges, 
                            std::vector<std::vector<int>> &adjacentEdgesOther, std::vector<int> &interiorVertices ){

        adjacentEdges.resize( numVertices );
        adjacentEdgesOther.resize( numVertices );

        for ( int edgeIdx=0; edgeIdx < Edge.rows(); ++edgeIdx ) {
            adjacentEdges[ Edge( edgeIdx, 0 ) ].emplace_back( edgeIdx );
            adjacentEdgesOther[ Edge( edgeIdx, 0)].emplace_back( 1 );
            adjacentEdges[ Edge( edgeIdx, 1 ) ].emplace_back( edgeIdx );
            adjacentEdgesOther[ Edge( edgeIdx, 1) ].emplace_back( 0 );
        }

        for ( int vertexIdx = 0; vertexIdx < numVertices; vertexIdx++ )
            if ( adjacentEdges[ vertexIdx ].size() > 1 )
                interiorVertices.emplace_back( vertexIdx );

    }

    Eigen::SparseMatrix<double> DcurveAngles(  const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                                             const std::vector<std::vector<int>>  &adjacentEdgesOther ,const std::vector<int> &interiorVertices, 
                                             const Eigen::Vector3d &normal  ){
        int numVertices = Geom.rows();
        int numEdges = Edge.rows();


        std::vector<Eigen::Triplet<double>> tripletList;
        Eigen::SparseMatrix<double> Dest( interiorVertices.size(), 3*numVertices );
        Dest.setZero();
        std::vector<Eigen::Vector3d> EdgeVectors;
        std::vector<double> EdgeLengths( numEdges );
         // compute edge-based properties
        for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {

            Eigen::Vector3d p0, p1;
            p0 = Geom.row( Edge( edgeIdx,0 ) );
            p1 = Geom.row( Edge( edgeIdx,1 ) );

            EdgeVectors.emplace_back(p0 - p1);
            EdgeLengths[edgeIdx] = EdgeVectors[edgeIdx].norm();
        }

        // Iterate over interior vertices, to be replaced by proper topology
        for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size(); innerVertexIdx++ ) {
            
            const int vertexIdx = interiorVertices[innerVertexIdx];
            const int leftEdgeIdx = adjacentEdges[vertexIdx][0];
            const int leftEdgeIdxOther = adjacentEdgesOther[vertexIdx][0];
            double leftEdgeSign = (leftEdgeIdxOther == 0) ? 1 : -1;

            const int rightEdgeIdx = adjacentEdges[vertexIdx][1];
            const int rightEdgeIdxOther = adjacentEdgesOther[vertexIdx][1];
            double rightEdgeSign = (rightEdgeIdxOther == 1) ? 1 : -1;

            Eigen::Vector3d vertexNormal = EdgeVectors[leftEdgeIdx].cross( EdgeVectors[rightEdgeIdx] );

            double normalProduct = normal.dot(vertexNormal);
            double edgeDotProduct = EdgeVectors[leftEdgeIdx].dot(EdgeVectors[rightEdgeIdx]);

            double Denominator = EdgeLengths[leftEdgeIdx] * EdgeLengths[leftEdgeIdx] * EdgeLengths[rightEdgeIdx] *
                                    EdgeLengths[rightEdgeIdx];

            // Left Vertex
            Eigen::Vector3d CrossDeriv_Left = (EdgeVectors[rightEdgeIdx] * leftEdgeSign).cross( normal );
            Eigen::Vector3d LocalDeriv_Left = CrossDeriv_Left * edgeDotProduct - EdgeVectors[rightEdgeIdx] * normalProduct * leftEdgeSign;
            LocalDeriv_Left /= Denominator;
            localToGlobal( innerVertexIdx, Edge( leftEdgeIdx, leftEdgeIdxOther),numVertices, LocalDeriv_Left, tripletList );

            // Middle Vertex
            Eigen::Vector3d CrossDeriv_Mid = normal.cross( EdgeVectors[rightEdgeIdx] * leftEdgeSign +
                                                            EdgeVectors[leftEdgeIdx] * rightEdgeSign );
            Eigen::Vector3d LocalDeriv_Mid = CrossDeriv_Mid * edgeDotProduct -
                                    ( EdgeVectors[leftEdgeIdx] * rightEdgeSign - EdgeVectors[rightEdgeIdx] * leftEdgeSign ) * normalProduct;
            LocalDeriv_Mid /= Denominator;
            localToGlobal( innerVertexIdx, vertexIdx, numVertices, LocalDeriv_Mid, tripletList );

            // Right Vertex
            Eigen::Vector3d CrossDeriv_Right = (EdgeVectors[leftEdgeIdx]  * rightEdgeSign).cross( normal );
            Eigen::Vector3d LocalDeriv_Right = CrossDeriv_Right * edgeDotProduct + EdgeVectors[leftEdgeIdx] * normalProduct * rightEdgeSign;
            LocalDeriv_Right /= Denominator;
            localToGlobal( innerVertexIdx, Edge( rightEdgeIdx, rightEdgeIdxOther),numVertices, LocalDeriv_Right, tripletList );
        }


        Dest.setFromTriplets( tripletList.begin(), tripletList.end());
        return Dest;
    }
    
    void D2curveAngles( const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                        const std::vector<std::vector<int>>  &adjacentEdgesOther ,const std::vector<int> &interiorVertices, const Eigen::Vector3d &normal,      
                        std::vector<std::vector<Eigen::Triplet<double>>> &tripletList ){
        int numVertices = Geom.rows();
        int numEdges = Edge.rows();
        if ( tripletList.size() != interiorVertices.size() )
            tripletList.resize( interiorVertices.size() );
        
        std::vector<Eigen::Vector3d> EdgeVectors;

        std::vector<double> EdgeLengths( numEdges );

        for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {

            Eigen::Vector3d p0, p1;
            p0 = Geom.row( Edge( edgeIdx,0 ) );
            p1 = Geom.row( Edge( edgeIdx,1 ) );

            EdgeVectors.emplace_back(p0 - p1);
            EdgeLengths[edgeIdx] = EdgeVectors[edgeIdx].norm();
        }
        // Iterate over interior vertices, to be replaced by proper topology
        for ( int innerVertexIdx = 0; innerVertexIdx < interiorVertices.size() ; innerVertexIdx++ ) {
            const int vertexIdx = interiorVertices[innerVertexIdx];
            const int leftEdgeIdx = adjacentEdges[vertexIdx][0];
            const int leftEdgeIdxOther = adjacentEdgesOther[vertexIdx][0];
            double leftEdgeSign = (leftEdgeIdxOther == 0) ? 1 : -1;

            const int rightEdgeIdx = adjacentEdges[vertexIdx][1];
            const int rightEdgeIdxOther = adjacentEdgesOther[vertexIdx][1];
            double rightEdgeSign = (rightEdgeIdxOther == 1) ? 1 : -1;  

            Eigen::Vector3d vertexNormal = EdgeVectors[leftEdgeIdx].cross( EdgeVectors[rightEdgeIdx] );

            double normalProduct = normal.dot(vertexNormal);
            double edgeDotProduct = EdgeVectors[leftEdgeIdx].dot(EdgeVectors[rightEdgeIdx]);

            double Denominator = EdgeLengths[leftEdgeIdx] * EdgeLengths[leftEdgeIdx] * EdgeLengths[rightEdgeIdx] *
                                    EdgeLengths[rightEdgeIdx];        
            
            Eigen::Vector3d dL_Denominator = EdgeVectors[leftEdgeIdx] * 2 * EdgeLengths[rightEdgeIdx] *  EdgeLengths[rightEdgeIdx] * leftEdgeSign;
            Eigen::Vector3d dR_Denominator = EdgeVectors[rightEdgeIdx] * 2 *  EdgeLengths[leftEdgeIdx] * EdgeLengths[leftEdgeIdx] * rightEdgeSign;

            Eigen::Vector3d dM_Denominator = (dR_Denominator - dL_Denominator) * -1.;    
        
        
            Eigen::Vector3d CrossDeriv_Right = (EdgeVectors[leftEdgeIdx]  * rightEdgeSign).cross( normal );
            Eigen::Vector3d CrossDeriv_Mid = normal.cross( EdgeVectors[rightEdgeIdx] * leftEdgeSign +
                                                            EdgeVectors[leftEdgeIdx] * rightEdgeSign );        
        
            // Left Vertex
            Eigen::Vector3d CrossDeriv_Left = (EdgeVectors[rightEdgeIdx] * leftEdgeSign).cross( normal );

            Eigen::Vector3d LocalDeriv_Left = CrossDeriv_Left * edgeDotProduct - EdgeVectors[rightEdgeIdx] * normalProduct * leftEdgeSign;
            LocalDeriv_Left /= Denominator;     

            Eigen::Matrix3d LocalDeriv_Left_Left;
            LocalDeriv_Left_Left.setZero();
            LocalDeriv_Left_Left -= tensorProduct(EdgeVectors[rightEdgeIdx]*leftEdgeSign , CrossDeriv_Left) - tensorProduct(CrossDeriv_Left, EdgeVectors[rightEdgeIdx] *leftEdgeSign);
            LocalDeriv_Left_Left -= tensorProduct(LocalDeriv_Left, dL_Denominator);
            LocalDeriv_Left_Left /= Denominator;
            

            localToGlobal( innerVertexIdx, Edge( leftEdgeIdx ,leftEdgeIdxOther ), Edge(leftEdgeIdx, leftEdgeIdxOther ),numVertices,
                            LocalDeriv_Left_Left, tripletList );
            
            Eigen::Matrix3d  LocalDeriv_Left_Right;

            getCrossOp( normal, LocalDeriv_Left_Right );
            LocalDeriv_Left_Right *= rightEdgeSign * leftEdgeSign * edgeDotProduct;
            LocalDeriv_Left_Right -= tensorProduct( EdgeVectors[leftEdgeIdx] * rightEdgeSign, CrossDeriv_Left );
            LocalDeriv_Left_Right.diagonal().array()+=( leftEdgeSign * rightEdgeSign * normalProduct );
            LocalDeriv_Left_Right -= tensorProduct( CrossDeriv_Right, EdgeVectors[rightEdgeIdx] * leftEdgeSign );
            LocalDeriv_Left_Right += tensorProduct( LocalDeriv_Left, dR_Denominator );

            LocalDeriv_Left_Right /= Denominator;

            localToGlobal( innerVertexIdx, Edge( leftEdgeIdx ,leftEdgeIdxOther ), Edge(rightEdgeIdx, rightEdgeIdxOther ), numVertices,
                            LocalDeriv_Left_Right, tripletList );  
            
            Eigen::Matrix3d LocalDeriv_Left_Mid;
            getCrossOp( normal, LocalDeriv_Left_Mid );
            LocalDeriv_Left_Mid *= rightEdgeSign * leftEdgeSign * edgeDotProduct;
            LocalDeriv_Left_Mid += tensorProduct( EdgeVectors[leftEdgeIdx] * rightEdgeSign -
                                                    EdgeVectors[rightEdgeIdx] * leftEdgeSign, CrossDeriv_Left );

            LocalDeriv_Left_Mid.diagonal().array()+=( -leftEdgeSign * rightEdgeSign * normalProduct );
            LocalDeriv_Left_Mid -= tensorProduct( CrossDeriv_Mid, EdgeVectors[rightEdgeIdx] * leftEdgeSign );

            LocalDeriv_Left_Mid += tensorProduct( dM_Denominator, LocalDeriv_Left  );

            LocalDeriv_Left_Mid /=  Denominator;


            localToGlobal( innerVertexIdx, Edge( leftEdgeIdx ,leftEdgeIdxOther ), vertexIdx,numVertices, LocalDeriv_Left_Mid, 
                            tripletList );
            localToGlobal( innerVertexIdx, vertexIdx,Edge( leftEdgeIdx ,leftEdgeIdxOther ), numVertices, LocalDeriv_Left_Mid,
                            tripletList );
            
            Eigen::Vector3d LocalDeriv_Mid = CrossDeriv_Mid * edgeDotProduct -
                               ( EdgeVectors[leftEdgeIdx] * rightEdgeSign - EdgeVectors[rightEdgeIdx] * leftEdgeSign ) * normalProduct;
            LocalDeriv_Mid /= Denominator;

            Eigen::Matrix3d LocalDeriv_Mid_Mid;
            LocalDeriv_Mid_Mid.setZero();
            LocalDeriv_Mid_Mid -= tensorProduct( EdgeVectors[leftEdgeIdx] * rightEdgeSign -
                                                EdgeVectors[rightEdgeIdx] * leftEdgeSign, CrossDeriv_Mid );

            LocalDeriv_Mid_Mid.diagonal().array()+=( 2 * leftEdgeSign * rightEdgeSign * normalProduct );
            LocalDeriv_Mid_Mid += tensorProduct( CrossDeriv_Mid,  EdgeVectors[leftEdgeIdx] * rightEdgeSign - EdgeVectors[rightEdgeIdx] * leftEdgeSign );

            LocalDeriv_Mid_Mid += tensorProduct( LocalDeriv_Mid, dM_Denominator );


            LocalDeriv_Mid_Mid /= Denominator;

            localToGlobal( innerVertexIdx, vertexIdx, vertexIdx, numVertices, LocalDeriv_Mid_Mid, tripletList );

            Eigen::Vector3d LocalDeriv_Right = CrossDeriv_Right * edgeDotProduct + EdgeVectors[leftEdgeIdx] * normalProduct * rightEdgeSign;
            LocalDeriv_Right /= Denominator;

            Eigen::Matrix3d LocalDeriv_Right_Right;
            LocalDeriv_Right_Right.setZero();
            LocalDeriv_Right_Right += tensorProduct( EdgeVectors[leftEdgeIdx] * rightEdgeSign, CrossDeriv_Right ) -
                                        tensorProduct( CrossDeriv_Right, EdgeVectors[leftEdgeIdx] * rightEdgeSign );
            LocalDeriv_Right_Right += tensorProduct( LocalDeriv_Right, dR_Denominator );
            LocalDeriv_Right_Right /= Denominator;

            localToGlobal( innerVertexIdx,Edge( rightEdgeIdx, rightEdgeIdxOther ), Edge( rightEdgeIdx, rightEdgeIdxOther),numVertices,
                            LocalDeriv_Right_Right, tripletList );

            Eigen::Matrix3d LocalDeriv_Right_Left;
            LocalDeriv_Right_Left.setZero();
            getCrossOp(normal, LocalDeriv_Right_Left);
            LocalDeriv_Right_Left *= -leftEdgeSign * rightEdgeSign * edgeDotProduct;
            LocalDeriv_Right_Left += tensorProduct( EdgeVectors[rightEdgeIdx] * leftEdgeSign, CrossDeriv_Right );

            LocalDeriv_Right_Left.diagonal().array()+=(leftEdgeSign * normalProduct * rightEdgeSign);
            LocalDeriv_Right_Left += tensorProduct( CrossDeriv_Left, EdgeVectors[leftEdgeIdx] * rightEdgeSign );

            LocalDeriv_Right_Left -= tensorProduct( LocalDeriv_Right, dL_Denominator );
            LocalDeriv_Right_Left /= Denominator;

            localToGlobal( innerVertexIdx, Edge( rightEdgeIdx, rightEdgeIdxOther ), Edge( leftEdgeIdx, leftEdgeIdxOther ),numVertices,
                            LocalDeriv_Right_Left, tripletList );

            Eigen::Matrix3d LocalDeriv_Right_Mid;
            getCrossOp( normal, LocalDeriv_Right_Mid );
            LocalDeriv_Right_Mid *= -rightEdgeSign * leftEdgeSign * edgeDotProduct;
            LocalDeriv_Right_Mid += tensorProduct( EdgeVectors[leftEdgeIdx] * rightEdgeSign -
                                                    EdgeVectors[rightEdgeIdx] * leftEdgeSign, CrossDeriv_Right );
            LocalDeriv_Right_Mid.diagonal().array() +=( -leftEdgeSign * rightEdgeSign * normalProduct );
            LocalDeriv_Right_Mid += tensorProduct( CrossDeriv_Mid,  EdgeVectors[leftEdgeIdx] * rightEdgeSign );
            LocalDeriv_Right_Mid += tensorProduct( dM_Denominator, LocalDeriv_Right );

            LocalDeriv_Right_Mid /= Denominator;


            localToGlobal( innerVertexIdx, Edge( rightEdgeIdx, rightEdgeIdxOther ), vertexIdx, numVertices,
                            LocalDeriv_Right_Mid, tripletList );
            localToGlobal( innerVertexIdx, vertexIdx,Edge( rightEdgeIdx, rightEdgeIdxOther ),  numVertices,
                            LocalDeriv_Right_Mid, tripletList );
    
        }
    }



    Eigen::SparseMatrix<double> DcurveLength(  const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                                            const std::vector<std::vector<int>>  &adjacentEdgesOther ,const std::vector<int> &interiorVertices, 
                                               const Eigen::Vector3d &normal ){
        
        int numVertices = Geom.rows();
        int numEdges = Edge.rows();

        Eigen::SparseMatrix<double> Dest;
        Dest.resize(numEdges, 3*numVertices);
        Dest.setZero();
        std::vector<Eigen::Triplet<double>> tripletList;
        for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {

            Eigen::Vector3d p0, p1;
            p0 = Geom.row( Edge( edgeIdx,0 ) );
            p1 = Geom.row( Edge( edgeIdx,1 ) );

            Eigen::Vector3d localGradient = p0 - p1;
            double edgeLength = localGradient.norm();
            localGradient /= edgeLength;

            for ( int d : { 0, 1, 2 } ) {
                tripletList.emplace_back( edgeIdx, Edge( edgeIdx,0 ) + d * numVertices, localGradient[d] );
                tripletList.emplace_back( edgeIdx, Edge( edgeIdx,1 ) + d * numVertices, -localGradient[d] );
            }
        }

        Dest.setFromTriplets( tripletList.begin(), tripletList.end());
        return Dest;
    }
    
    
    void D2curveLength( const Eigen::MatrixXd &Geom, const Eigen::MatrixXi &Edge, const std::vector<std::vector<int>> &adjacentEdges,
                        const std::vector<std::vector<int>>  &adjacentEdgesOther ,const std::vector<int> &interiorVertices, 
                        const Eigen::Vector3d &normal,std::vector<std::vector<Eigen::Triplet<double>>> &tripletList ){
        int numVertices = Geom.rows();
        int numEdges = Edge.rows(); 
        if ( tripletList.size() != numEdges )
            tripletList.resize( numEdges );

        for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {
            // Undeformed Length
            Eigen::Vector3d p0, p1;
            p0 = Geom.row( Edge( edgeIdx,0 ) );
            p1 = Geom.row( Edge( edgeIdx,1 ) );
            
            Eigen::Vector3d edgeVec = p0 -p1;
            double edgeLength = edgeVec.norm();

            Eigen::Matrix3d tp;
            tp.setZero();
            tp  = tensorProduct(edgeVec, edgeVec);
            tp *= -1. / ( edgeLength * edgeLength * edgeLength );
            tp.diagonal().array() += ( 1 / edgeLength ); 

            localToGlobal( edgeIdx, Edge( edgeIdx,0 ), Edge( edgeIdx,0 ), numVertices, tp, tripletList );
            localToGlobal( edgeIdx, Edge( edgeIdx,0 ), Edge( edgeIdx,1 ), numVertices, -tp, tripletList );
            localToGlobal( edgeIdx, Edge( edgeIdx,1 ), Edge( edgeIdx,1 ), numVertices, tp, tripletList );
            localToGlobal( edgeIdx, Edge( edgeIdx,1 ), Edge( edgeIdx,0 ), numVertices, -tp, tripletList );
        }  
    
    
    }
}




