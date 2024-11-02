# pcm-rs: Partial Curve Matching 
This repo provides an implementation to compute **partial curve matching** (against a **curve** _and_ against a **graph**). 

Partial curve matching **against a curve** takes two curves $P$ and $Q$ and a distance threshold $\epsilon$, and checks whether there exists _some_ subcurve $Q'$ in $Q$ in such that the Fréchet distance between $P$ and $Q'$ is within the Fréchet distance threshold of $\epsilon$.

Partial curve matching **against a graph** takes a graph G (consisting of vertices and edges), a curve $P$, and a distance threshold $\epsilon$, and checks whether there exists _some_ path $Q$ in $G$ in such that there is a partial curve match of $P$ and $Q$ for the Fréchet distance threshold of $\epsilon$.
 
### Constraints
* **Curve constraints**: Curves are polygonal chains (piecewise linear) and use the continuous strong Fréchet distance as a (sub)curve similarity/distance measure.
* **Graph constraints**: Graphs are undirected (edge can be walked in both directions) and vectorized (every edge is a straight line segment without curvature).
* Furthermore do we assume all points (curve points and graph vertices) to be in **general position** (thus all point coordinates are unique and therefore all points have non-zero pairwise distance).

## Getting started
To use this code with Python:
1. First build the shared library (`partial_curve_matching.so` file, check out the `build_python_module.sh` script).
2. Then import the shared library as a module into your Python code (using `import partial_curve_matching`, check out the `example_usage.py` script).

To use this code with Rust, check out `pcm_vis/src/main.rs` for example usage.

## Notes

### Performance considerations
This code is definitely _not_ optimized (both in terms of complexity and implementation), see [Future work](#future-work-optimizations) for obvious points of improvements.
E.g. all the boundaries of the FSD and RSD is computed, no heuristics are applied, it is single-threaded, vectors are unnecessarily copied all over the place.
The use-case of this code repo is running the BundlePatcher<a href="#bundlepatcher" id="bundlepatcherref"><sup>4</sup></a>, and I only intend to improve in order to make those algorithms run sufficiently fast.


### Partial Curve-Graph Matching variants.
Partial curve matching against a curve has two implementations, one with linear time complexity and another with exponential time complexity. It defaults to the linear variant, but for reference and as a manner for sanity checking I have left in the exponential variant.

## Future work (optimizations)
Points of interest to work on:
* [x] Partial curve to graph matching linear implementation <a href="#altefrat" id="altefratref"><sup>5</sup></a>
* [ ] Lazily FSD and RSD construction
* [ ] PCM specific extention to FSD<a href="#maheshwari" id="maheshwariref"><sup>2</sup></a>
* [ ] Multi-threaded
* [ ] SIMD

## Acknowledgement
For detailed information on the FSD and RSD we refer to the paper by Alt & Godau <a href="#altgodau" id="altgodauref"><sup>1</sup></a>. \
Work taken from pyfrechet<a href="#pyfrechet" id="pyfrechetref"><sup>3</sup></a> for computing cell boundary and the idea to separate array for horizontal and vertical cell boundaries. \
The concept of constructing the Free-Space Surface by the paper of Alt, Efrat, Rote, Wenk <a href="#altefrat" id="altefratref"><sup>4</sup></a> as inspiration for the simple algorithm with exponential complexity, and as the description for the implementation with linear complexity (Shortcut pointers, Partial Maximum Stack, event construction and sweep line).

## References
<a id="altgodau" href="#altgodauref"><sup>1</sup></a> H. Alt and M. Godau, “Computing the Fréchet Distance between Two Polygonal Curves,” Int. J. Comput. Geom. Appl., vol. 05, no. 01n02, pp. 75–91, Mar. 1995, doi: 10.1142/S0218195995000064.\
<a id="maheshwari" href="#maheshwariref"><sup>2</sup></a> A. Maheshwari, J.-R. Sack, K. Shahbaz, and H. Zarrabi-Zadeh, “Improved Algorithms for Partial Curve Matching,” Algorithmica, vol. 69, no. 3, pp. 641–657, Jul. 2014, doi: 10.1007/s00453-013-9758-3.\
<a id="pyfrechet" href="#pyfrechetref"><sup>3</sup></a> https://github.com/compgeomTU/frechetForCurves \
<a id="bundlepatcher" href="#bundlepatcher"><sup>4</sup></a> https://github.com/jvtubergen/geoalg.  \
<a id="altefrat" href="#altefrat"><sup>5</sup></a> H. Alt, A. Efrat, G. Rote, and C. Wenk, “Matching planar maps,” Journal of Algorithms, vol. 49, no. 2, pp. 262–283, Nov. 2003, doi: 10.1016/S0196-6774(03)00085-3. \
