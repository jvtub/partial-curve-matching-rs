"""Python bindings for partial curve matching library."""

try:
    from partial_curve_matching import *  # Import from the compiled Rust module
except ImportError as e:
    raise ImportError(
        "Could not import the compiled Rust module 'partial_curve_matching'. "
        "Make sure the Rust library is built and the .so file is in your Python path."
    ) from e

__version__ = "0.1.0"
__all__ = []  # Will be populated by the imported Rust module
