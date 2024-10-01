use pyo3::pyfunction;
use crate::{curve::Curve, fsd::FSD};

/// Returns any subcurve of qs (if it exists) with Fréchet distance to ps below threshold epsilon.
#[pyfunction]
pub fn partial_curve(ps: Curve, qs: Curve, eps: f64) -> Option<(f64, f64)> {
    let fsd = FSD::new(ps, qs, eps);
    let rsd = fsd.to_rsd();
    let opt_steps = rsd.pcm_steps();
    if opt_steps.is_none() { 
       None
    } else {
        let steps = opt_steps.unwrap();
        let start= steps[0].1;
        let end = steps.last().unwrap().1;
        Some((start, end))
    }
}