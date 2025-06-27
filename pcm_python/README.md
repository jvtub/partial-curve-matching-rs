# PCM Python

Python bindings for the partial curve matching library.

## Installation

Install using hatch:

```bash
pip install -e .
```

Or from another project using hatch:

```toml
[project]
dependencies = [
    "pcm-python @ file:///path/to/pcm_python",
]
```

## Usage

```python
import pcm_python

# Use the partial curve matching functions
```

## Development

This package requires the Rust library to be compiled first. Make sure to build the `pcm_pyo3` crate before using this Python package.
