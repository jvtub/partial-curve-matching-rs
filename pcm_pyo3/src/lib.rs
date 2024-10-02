use pyo3::prelude::*;
use pcm::*;

#[pymodule]
fn partial_curve_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector>()?;
    m.add_function(wrap_pyfunction!(pcm::partial_curve::partial_curve, m)?)?;
    m.add_class::<LinearGraph>()?;
    m.add_function(wrap_pyfunction!(make_exponential_graph, m)?)?;
    m.add_function(wrap_pyfunction!(pcm::partial_curve_graph_exponential::partial_curve_graph_exponential, m)?)?;
    m.add_function(wrap_pyfunction!(make_linear_graph, m)?)?;
    m.add_function(wrap_pyfunction!(pcm::partial_curve_graph_linear::partial_curve_graph_linear, m)?)?;
    Ok(())
}
