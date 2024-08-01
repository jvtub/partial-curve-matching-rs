use pyo3::prelude::*;
use pcm::prelude::*;

#[pymodule]
fn partial_curve_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector>()?;
    m.add_function(wrap_pyfunction!(partial_curve, m)?)?;
    Ok(())
}
