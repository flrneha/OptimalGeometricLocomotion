#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

#include <curve-energy/curve_energy.h>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(pycurve, m ) {
  // Energies
  m.def( "membrane_energy", &curve::membrane_energy, "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a );
  m.def( "bending_energy", &curve::bending_energy, "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a , "normal"_a);
  m.def( "reg_bending_energy", &curve::reg_bending_energy, "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a , "normal"_a);

  m.def( "curve_energy", &curve::curve_energy, "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a , "normal"_a, "bendingWeight"_a, "membraneWeight"_a );

  // Gradients
  m.def( "membrane_undeformed_gradient",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge) -> Eigen::VectorXd {
           Eigen::VectorXd out;
           curve::membrane_undeformed_gradient( UndeformedGeom, DeformedGeom, Edge, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a
  );

  m.def( "membrane_deformed_gradient",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge) -> Eigen::VectorXd {
           Eigen::VectorXd out;
           curve::membrane_deformed_gradient( UndeformedGeom, DeformedGeom, Edge, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a
  );

  m.def( "bending_undeformed_gradient",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal) -> Eigen::VectorXd {
           Eigen::VectorXd out;
           curve::bending_undeformed_gradient( UndeformedGeom, DeformedGeom, Edge, normal, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a
  );
  
  m.def( "bending_deformed_gradient",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal) -> Eigen::VectorXd {
           Eigen::VectorXd out;
           curve::bending_deformed_gradient( UndeformedGeom, DeformedGeom, Edge, normal, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a
  );
  m.def( "reg_bending_deformed_gradient",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal) -> Eigen::VectorXd {
           Eigen::VectorXd out;
           curve::reg_bending_deformed_gradient( UndeformedGeom, DeformedGeom, Edge, normal, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a
  );
  m.def( "curve_deformed_gradient",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight) -> Eigen::VectorXd {
           Eigen::VectorXd out;
           curve::curve_deformed_gradient( UndeformedGeom, DeformedGeom, Edge, normal, bendingWeight, membraneWeight, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a, "bendingWeight"_a, "membraneWeight"_a
  );

  m.def( "curve_undeformed_gradient",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight) -> Eigen::VectorXd {
           Eigen::VectorXd out;
           curve::curve_undeformed_gradient( UndeformedGeom, DeformedGeom, Edge, normal,bendingWeight, membraneWeight, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a, "bendingWeight"_a, "membraneWeight"_a
  );
  // Hessians
  m.def( "membrane_deformed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::membrane_deformed_hessian( UndeformedGeom, DeformedGeom, Edge, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a
  );

  m.def( "membrane_undeformed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::membrane_undeformed_hessian( UndeformedGeom, DeformedGeom, Edge, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a
  );

  m.def( "membrane_mixed_hessian",
        []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
            const Eigen::MatrixXi &Edge, bool FirstDerivWRTDef) -> Eigen::SparseMatrix<double> {
          Eigen::SparseMatrix<double> out;
          curve::membrane_mixed_hessian( UndeformedGeom, DeformedGeom, Edge, out, FirstDerivWRTDef );
          return out;
        },
        "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "first_deriv_wrt_def"_a = true
  );

  m.def( "bending_deformed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::bending_deformed_hessian( UndeformedGeom, DeformedGeom, Edge, normal, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a
  );

  m.def( "reg_bending_deformed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::reg_bending_deformed_hessian( UndeformedGeom, DeformedGeom, Edge, normal, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a
  );
  m.def( "bending_undeformed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::bending_undeformed_hessian( UndeformedGeom, DeformedGeom, Edge, normal, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a
  );

  m.def( "bending_mixed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, bool FirstDerivWRTDef) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::bending_mixed_hessian( UndeformedGeom, DeformedGeom, Edge, normal, out,FirstDerivWRTDef );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a, "first_deriv_wrt_def"_a = true
  );

  m.def( "curve_undeformed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::curve_undeformed_hessian( UndeformedGeom, DeformedGeom, Edge, normal, bendingWeight, membraneWeight, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a, "bendingWeight"_a, "membraneWeight"_a
  );

  m.def( "curve_deformed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight) -> Eigen::SparseMatrix<double> {
           Eigen::SparseMatrix<double> out;
           curve::curve_deformed_hessian( UndeformedGeom, DeformedGeom, Edge, normal,bendingWeight, membraneWeight, out );
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a, "bendingWeight"_a, "membraneWeight"_a
  );

  m.def( "curve_mixed_hessian",
         []( const Eigen::MatrixXd &UndeformedGeom, const Eigen::MatrixXd &DeformedGeom,
             const Eigen::MatrixXi &Edge, const Eigen::Vector3d &normal, double bendingWeight, double membraneWeight, bool FirstDerivWRTDef) -> Eigen::SparseMatrix<double>{
           Eigen::SparseMatrix<double> out;
           curve::curve_mixed_hessian( UndeformedGeom, DeformedGeom, Edge, normal,bendingWeight, membraneWeight, out ,FirstDerivWRTDef);
           return out;
         },
         "UndeformedGeom"_a, "DeformedGeom"_a, "Edge"_a, "normal"_a, "bendingWeight"_a, "membraneWeight"_a, "first_deriv_wrt_def"_a = true
  );
}
