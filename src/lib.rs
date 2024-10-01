#![feature(let_chains)]
pub mod vector;
pub mod curve;
pub mod lineboundary;
pub mod curve_graph_linear;
pub mod fsd;

pub use curve_graph_linear::{partial_curve_graph_linear, LinearGraph, make_linear_graph};
pub use vector::Vector;
pub use lineboundary::{LineBoundary, OptLineBoundary};
pub use curve::Curve;
pub use fsd::FSD;

#[allow(non_upper_case_globals)]
const sanity_check: bool = true;
pub const EPS: f64 = 0.00001;

pub mod prelude {   
    pub use crate::{OptLineBoundary, LineBoundary, Vector, FSD, EPS, Curve, LinearGraph, partial_curve_graph_linear, make_linear_graph};
}
