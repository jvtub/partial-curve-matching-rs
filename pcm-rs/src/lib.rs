#![feature(let_chains)]
use std::ops::{Add, Mul, Sub};
use ndarray::{prelude::*, OwnedRepr};
use pyo3::{exceptions::PyTypeError, prelude::*};
const EPS: f64 = 0.00001;

// ==============
// === Vector ===
// ==============

#[derive(Debug, Clone, Copy)]
#[pyclass]
struct Vector {
    x: f64,
    y: f64
}
impl Vector {
    fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }
    fn distance(self, rhs: Self) -> f64 {
        (rhs - self).dot(rhs - self).sqrt()
    }
}
#[pymethods]
impl Vector {
    #[new]
    fn new(x: f64, y: f64) -> Self {
        Vector { x, y }
    }
}
impl Mul for Vector {
    type Output = Vector;
    fn mul(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x * rhs.x,
            y: self.y * rhs.y
        }
    }
}
impl Add for Vector {
    type Output = Vector;
    fn add(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl Sub for Vector {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector { 
            x: self.x - rhs.x,
            y: self.y - rhs.y
        }
    }
}
impl std::ops::Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Vector {
        Vector {
            x: self * rhs.x, 
            y: self * rhs.y 
        }
    }
}

type Curve = Vec<Vector>;


// =====================
// === Line boundary ===
// =====================

/// LineBoundaries are ponentially empty. Zero lines we denote with the None type.
type OptLineBoundary = Option<LineBoundary>;

/// Unit-distance one-dimensional boundary
#[derive(Debug, Clone, Copy)]
struct LineBoundary {
    /// Starting point, somewhere on the unit interval, but must be smaller than b.
    a: f64,
    /// End point, somewhere on the unit interval, but must be larger than a.
    b: f64
}
impl LineBoundary {
    fn new(a: f64, b: f64) -> OptLineBoundary {
        if a < b {
            Some(LineBoundary { a, b })
        } else {
            None
        }
    }
}

/// Union two line boundaries.
fn union(opt_b1: OptLineBoundary, opt_b2: OptLineBoundary) -> OptLineBoundary {
    if opt_b1.is_none() { opt_b2 } 
    else if opt_b2.is_none() { opt_b1 }
    else {
        let b1 = opt_b1.unwrap();
        let b2 = opt_b2.unwrap();

        let a = b1.a.min(b2.a); 
        let b = b1.b.max(b2.b);

        LineBoundary::new(a, b)
    }
}

/// Compute unit-distance free space between line segment p1p2 and point q.
fn compute_cell_boundary(p: Vector, q1: Vector, q2: Vector, eps: f64 ) -> OptLineBoundary {
    let dq = q2 - q1;
    // let dq = q2 - q1;
    let divisor = dq.dot(dq);

    let b = dq.dot(p - q1);
    let c = divisor * (q1.dot(q1) + p.dot(p) - 2. * q1.dot(p) - eps * eps);
    let root = b * b - c;

    if root < 0. {
        return None;
    }
    let root = root.sqrt();

    // Possible intersection points.
    let t1 = (b - root) / divisor;
    let t2 = (b + root) / divisor;

    // Bound to interval [0,1];
    let t1 = t1.min(1.).max(0.);  
    let t2 = t2.min(1.).max(0.);  

    if t2 - t1 < EPS {
        None
    } else {
        Some(
            LineBoundary {
                a: t1,
                b: t2
            }
        )
    }

}


// ==========================
// === Free-Space Diagram ===
// ==========================

/// Free-Space Diagram.
#[derive(Debug)]
struct FSD {
    n: usize,
    m: usize,
    /// Axis-specific dimensions.
    dims: [(usize, usize); 2], 
    /// Cell boundaries for both axii.
    segs : ArrayBase<OwnedRepr<OptLineBoundary>, Dim<[usize; 3]>>,
    /// Cornerpoints either true or not. 
    /// (Used for debugging purposes, the consistency in segment computations).
    corners: ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>>,
}
impl FSD {
    /// Construct empty FSD.
    fn empty(n: usize, m: usize) -> FSD {
        let dims = [(n,m-1), (m,n-1)];
        let segs = Array3::from_shape_simple_fn((2,n.max(m),m.max(n)), || None);
        // let verticals = Array2::from_shape_simple_fn([n,m-1], || None); // Contains n cols, with m-1 intervals.
        // let horizontals = Array2::from_shape_simple_fn([m,n-1], || None);  // Contains m rows, with n-1 intervals.
        let corners = Array2::from_shape_simple_fn([n,m], || false);
        FSD { n, m, dims, segs, corners }
    }


    /// Compute the free-space diagram between curve P (points ps) and curve Q (points qs).
    /// Placing P point indices on the horizontal axis and Q on the vertical axis.
    fn new(ps: Curve, qs: Curve, eps: f64) -> FSD {

        let n = ps.len();
        let m = qs.len();
        let mut fsd = FSD::empty(n, m);

        // Constructing cell boundaries.
        for axis in 0..2 {
            let dims = fsd.dims[axis];
            let (c1, c2) = [(&ps, &qs), (&qs, &ps)][axis];
            for x in 0..dims.0 {
                for y in 0..dims.1 {
                    fsd.segs[(axis,x,y)] = compute_cell_boundary(c1[x], c2[y], c2[y+1], eps);
                }
            }
        }

        // Constructing corners.
        for i in 0..n {
            for j in 0..m {
                fsd.corners[(i,j)] = ps[i].distance(qs[j]) < eps + 0.5 * EPS;
            }
        }

        fsd
    }
}


/// Compute reachable space diagram out of a free space diagram.
fn compute_rsd(fsd: FSD) -> FSD {
    let n = fsd.n;
    let m = fsd.m;
    let mut rsd = FSD::empty(n, m);

    // Initiate whole left FSD border (we seek partial curve).
    rsd.segs.slice_mut(s![0, 0, ..]).assign(&fsd.segs.slice(s![0, 0, ..]));

    // Walk all cells left to right, bottom to top.
    for j in 0..m {
        for i in 0..n {
            for axis in 0..2 {
                let (x,y) = [(i,j),(j,i)][axis];
                let curr = (axis, x, y); // current.
                let opt_prev = if y > 0 { Some((axis  , x  , y-1)) } else { None }; // previous.
                let opt_para = if x > 0 { Some((axis  , x-1, y  )) } else { None }; // parallel.
                let opt_orth = if x > 0 { Some((1-axis, y  , x-1)) } else { None }; // orthogonal.
                if let Some(orth) = opt_orth {
                    if rsd.segs[orth].is_some() {
                        rsd.segs[curr] = fsd.segs[curr];
                    }
                } 
                if let Some(para) = opt_para {
                    // Custom intersect.
                    if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[para] {
                        if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                            rsd.segs[curr] = union(rsd.segs[curr], LineBoundary::new(a.max(a_), b));
                        }
                    }
                } 
                if let Some(prev) = opt_prev { 
                    if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[prev] {
                        if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                            if b_ == 1. && a == 0. {
                                rsd.segs[curr] = union(rsd.segs[curr], LineBoundary::new(0., b));
                            }
                        }
                    }
                }
            }
        }
    }

    rsd
}


// ==============================
// === Partial Curve Matching ===
// ==============================

/// Check for a partial curve match of a reachability-space diagram.
fn rsd_pcm(rsd: &FSD) -> bool {
    rsd.segs.slice(s![0,rsd.n-1,..]).iter().any(|b| b.is_some())
}

/// Compute steps to walk along curves for partial matching solution.
fn rsd_pcm_steps(rsd: &FSD) -> Result<Option<Vec<(f64,f64)>>, String> {

    let n = rsd.n;
    let m = rsd.m;

    let mut steps = vec![];
    let mut found = false;

    let mut i = n - 1;
    let mut i_off = 0.;
    let mut j = 0;
    let mut j_off = 0.;

    // Seek lowest non-empty boundary on right side of the RSD.
    // (Basically performs PCM existence check as well.)
    for (_j, lb) in rsd.segs.slice(s![0,i,..]).iter().enumerate() {
        if let Some(LineBoundary { a, b }) = lb {
            j = _j;
            j_off = *a;
            found = true;
            break;
        }
    }
    if !found { return Ok(None) }

    // Final step is found.
    steps.push((i as f64 + i_off, j as f64 + j_off));

    // Walk backwards. (Walk greedily, it should not matter).
    while !(i == 0 && i_off == 0.) {

        let position = (i as f64 + i_off, j as f64 + j_off);
        println!("position: {position:?}");
        if !(i_off == 0. || j_off == 0.) { return Err(format!("Not at any RSD boundary while traversing ({position:?}).")); }


        // Figure out axis.
        let a = if i_off == 0. { 0 } else { 1 };

        // Convert i, i_off, j, j_off into x and y.
        let (mut x, mut x_off, mut y, mut y_off) = [(i, i_off, j, j_off), (j, j_off, i, i_off)][a];

        let curr = (a, x, y);
        if rsd.segs[curr].is_none() {
            return Err(format!("Current segment while walking ({curr:?}) should not be empty."));
        }

        // Try to walk backwards.
        let opt_prev = if y > 0 { Some((a  , x, y-1)) } else { None }; // previous.
        let opt_para = if x > 0 { Some((a  , x-1, y)) } else { None }; // parallel.
        let opt_orth = if y > 0 { Some((1-a, y-1, x)) } else { None }; // orthogonal.

        let mut next = None;
        if let Some(prev) = opt_prev {
            if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[prev] {
                if y_off == 0. && b_ == 1. {
                    next = Some(prev);
                    x_off = 0.;
                    y_off = a_;
                    y -= 1;
                }
            }
        }

        if next.is_none() && let Some(orth) = opt_orth {
            if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[orth] {
                if (y_off == 0. && b_ == 1.) || y_off > 0. {
                    next = Some(orth);
                    x_off = a_;
                    y_off = 0.;
                    x -= 1;
                } 
            }
        }

        if next.is_none() && let Some(para) = opt_para {
            if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[para] {
                if y_off >= a_ {
                    next = Some(para);
                    x_off = 0.;
                    y_off = a_;
                    x -= 1;
                }
            }
        }

        if next.is_none() {
            return Err(format!("Should find next step in backwards walk at {curr:?}."));
        }
        
        // Convert back x and y into i, i_off, j, j_off.
        (i, i_off, j, j_off) = [(x, x_off, y, y_off), (y, y_off, x, x_off)][a];

        let position = (i as f64 + i_off, j as f64 + j_off);
        steps.push(position);
    }

    steps.reverse();
    Ok(Some(steps))
}

/// Returns any subcurve of qs (if it exists) with Fréchet distance to ps below threshold epsilon.
#[pyfunction]
fn partial_curve(ps: Curve, qs: Curve, eps: f64) -> Result<Option<(f64, f64)>, PyErr> {
    let fsd = FSD::new(ps, qs, eps);
    let rsd = compute_rsd(fsd);
    let opt_steps = rsd_pcm_steps(&rsd).map_err(|str| PyErr::new::<PyTypeError, _>(str))?;
    if opt_steps.is_none() { 
        Ok(None) 
    } else {
        let steps = opt_steps.unwrap();
        let start= steps[0].1;
        let end = steps.last().unwrap().1;
        Ok(Some((start, end)))
    }
}


#[pymodule]
fn partial_curve_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector>()?;
    m.add_function(wrap_pyfunction!(partial_curve, m)?)?;
    Ok(())
}