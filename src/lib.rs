#![feature(let_chains)]
pub mod vector;
pub mod curve;
pub mod lineboundary;
pub mod partial_curve;
pub mod partial_curve_graph_linear;
pub mod fsd;

// Exporting functionality.
pub use partial_curve::*;
pub use partial_curve_graph_linear::*;
pub use vector::Vector;
pub use lineboundary::{LineBoundary, OptLineBoundary};
pub use curve::Curve;
pub use fsd::FSD;

#[allow(non_upper_case_globals)]
const sanity_check: bool = true;
pub const EPS: f64 = 0.00001;