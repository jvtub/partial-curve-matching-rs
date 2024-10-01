use pyo3::prelude::*;
use pcm::prelude::*;

#[pymodule]
fn partial_curve_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector>()?;
    m.add_class::<LinearGraph>()?;
    m.add_function(wrap_pyfunction!(make_linear_graph, m)?)?;
    m.add_function(wrap_pyfunction!(partial_curve_graph_linear, m)?)?;
    Ok(())
}
