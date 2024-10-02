#![feature(let_chains)]
pub mod vector;
pub mod curve;
pub mod lineboundary;
pub mod partial_curve;
pub mod partial_curve_graph_exponential;
pub mod partial_curve_graph_linear;
pub mod fsd;

// Exporting functionality.
pub use vector::Vector;
pub use lineboundary::{LineBoundary, OptLineBoundary};
pub use curve::Curve;
pub use fsd::FSD;
pub use partial_curve::*;
pub use partial_curve_graph_exponential::*;
pub use partial_curve_graph_linear::*;
pub use partial_curve_graph_linear::partial_curve_graph_linear as partial_curve_graph;
pub use partial_curve_graph_linear::make_linear_graph as make_graph;

#[allow(non_upper_case_globals)]
const sanity_check: bool = true;
pub const EPS: f64 = 0.00001;