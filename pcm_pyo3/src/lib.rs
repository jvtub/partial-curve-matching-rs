use pyo3::prelude::*;
use pcm::prelude::*;

#[pymodule]
fn partial_curve_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector>()?;
    m.add_class::<Graph>()?;
    m.add_function(wrap_pyfunction!(partial_curve, m)?)?;
    m.add_function(wrap_pyfunction!(partial_curve_graph, m)?)?;
    m.add_function(wrap_pyfunction!(make_graph, m)?)?;
    Ok(())
}
