use pyo3::prelude::*;
use pcm::*;

#[pymodule]
fn partial_curve_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Python functions for partial curve matching.
    m.add_class::<Vector>()?;
    m.add_function(wrap_pyfunction!(pcm::partial_curve::partial_curve, m)?)?;

    // Python functions for partial curve to graph matching (defaulting to the linear time complexity implementation).
    m.add_class::<Graph>()?;
    m.add_function(wrap_pyfunction!(make_graph, m)?)?;
    m.add_function(wrap_pyfunction!(partial_curve_graph, m)?)?;

    // Python functions for partial curve to graph matching (exponential complexity).
    m.add_class::<ExponentialGraph>()?;
    m.add_function(wrap_pyfunction!(pcm::partial_curve_graph_exponential::partial_curve_graph_exponential, m)?)?;
    m.add_function(wrap_pyfunction!(pcm::partial_curve_graph_exponential::make_exponential_graph, m)?)?;

    Ok(())
}
