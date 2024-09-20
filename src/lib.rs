#![feature(let_chains)]
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::ops::{Add, Div, Mul, Sub};
use ndarray::{prelude::*, OwnedRepr};
use pyo3::{exceptions::PyTypeError, prelude::*};
use serde_derive::{Deserialize, Serialize};
use std::collections::BTreeMap as Map;
use std::collections::BTreeSet as Set;
use std::iter::zip;
pub const EPS: f64 = 0.00001;
const DEBUG: bool = false;

// ==============
// === Vector ===
// ==============

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[pyclass]
pub struct Vector {
    pub x: f64,
    pub y: f64
}
impl Vector {
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }
    pub fn distance(self, rhs: Self) -> f64 {
        (rhs - self).dot(rhs - self).sqrt()
    }
    pub fn min(&self, rhs: &Self) -> Self {
        Self {
            x: self.x.min(rhs.x),
            y: self.y.min(rhs.y),
        }
    }
    pub fn max(&self, rhs: &Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
        }
    }
}
#[pymethods]
impl Vector {
    #[new]
    pub fn new(x: f64, y: f64) -> Self {
        Vector { x, y }
    }
}
impl Div for Vector {
    type Output = Vector;
    fn div(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x / rhs.x,
            y: self.y / rhs.y
        }
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

pub type Curve = Vec<Vector>;

// =====================
// === Line boundary ===
// =====================

/// LineBoundaries are ponentially empty. Zero lines we denote with the None type.
pub type OptLineBoundary = Option<LineBoundary>;

/// Unit-distance one-dimensional boundary
#[derive(Debug, Clone, Copy)]
pub struct LineBoundary {
    /// Starting point, somewhere on the unit interval, but must be smaller than b.
    pub a: f64,
    /// End point, somewhere on the unit interval, but must be larger than a.
    pub b: f64
}
impl LineBoundary {
    
    pub fn new(a: f64, b: f64) -> OptLineBoundary {
        if a < b {
            Some(LineBoundary { a, b })
        } else {
            None
        }
    }

    /// Union two line boundaries.
    pub fn union(opt_b1: OptLineBoundary, opt_b2: OptLineBoundary) -> OptLineBoundary {
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

    /// Compute unit-distance free space line boundary between point p and line segment q.
    pub fn compute(p: Vector, q0: Vector, q1: Vector, eps: f64 ) -> OptLineBoundary {
        let v = q1 - q0;
        let vli = 1. / v.dot(v).sqrt();
        let vn = vli * v;

        let l = p - q0;
        let tca = l.dot(vn);
        let d2 = l.dot(l) - tca * tca;

        let e2 = eps * eps;
        if d2 > e2 { return None; }

        let thc = (e2 - d2).sqrt();
        let t0 = vli * (tca - thc);
        let t1 = vli * (tca + thc);
        let p0 = q0 + t0 * v; // First point of intersection.
        let p1 = q0 + t1 * v; // Second point of intersection.

        if t1 < 0. || t0 > 1. || t1 - t0 < 0.0001 { return None; }
        Some(
            LineBoundary {
                a: t0.min(1.).max(0.),
                b: t1.min(1.).max(0.)
            }
        )
    }

    /// Check the other is a subset of self.
    pub fn has_subset(&self, other: Self) -> bool {
        self.b >= other.b && self.a <= other.a
    }

}


// ==========================
// === Free-Space Diagram ===
// ==========================

/// Position on the FSD considering axis.
type FSDPosition = (usize, usize, usize, f64);

/// Convert a FSD position into curve positions.
fn position_to_ij((axis, x, y, off): FSDPosition) -> (f64, f64) {
    [(x as f64, y as f64 + off), (y as f64 + off, x as f64)][axis]
}

/// Convert position into a FSD horizontal/vertical cell.
fn position_to_seg((axis, x, y, off): FSDPosition) -> (usize, usize, usize) {
    (axis, x, y)
}

/// Check whether position is on the left boundary of the FSD.
fn position_on_left_boundary((axis, x, y, off): (usize, usize, usize, f64)) -> bool {
    (axis == 0 && x == 0) ||
    (axis == 1 && y == 0)
}

/// Free-Space Diagram.
#[derive(Debug, Clone)]
pub struct FSD {
    /// Width (number of points on ps).
    pub n: usize,
    /// Height (number of points on qs).
    pub m: usize,
    /// Axis-specific dimensions (basically `[(n,m-1), (m,n-1)]`).
    pub dims: [(usize, usize); 2], 
    /// Cell boundaries for both axii. Format is (axis, x, y).
    pub segs : ArrayBase<OwnedRepr<OptLineBoundary>, Dim<[usize; 3]>>,
    /// Cornerpoints either true or not. (Used for debugging purposes, the consistency in segment computations).
    pub corners: ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>>,
    /// FSD and RSD are the same struct, use this boolean to ensure being in the correct space.
    pub is_rsd: bool
}
impl FSD {
    /// Construct empty FSD.
    fn empty(n: usize, m: usize) -> FSD {
        let dims = [(n,m-1), (m,n-1)];
        let segs = Array3::from_shape_simple_fn((2,n.max(m),m.max(n)), || None);
        // let verticals = Array2::from_shape_simple_fn([n,m-1], || None); // Contains n cols, with m-1 intervals.
        // let horizontals = Array2::from_shape_simple_fn([m,n-1], || None);  // Contains m rows, with n-1 intervals.
        let corners = Array2::from_shape_simple_fn([n,m], || false);
        let is_rsd = false;
        FSD { n, m, dims, segs, corners, is_rsd }
    }


    /// Compute the free-space diagram between curve P (points ps) and curve Q (points qs).
    /// Placing P point indices on the horizontal axis and Q on the vertical axis.
    pub fn new(ps: Curve, qs: Curve, eps: f64) -> FSD {

        let n = ps.len();
        let m = qs.len();
        let mut fsd = FSD::empty(n, m);

        // Constructing cell boundaries.
        for axis in 0..2 {
            let dims = fsd.dims[axis];
            let (c1, c2) = [(&ps, &qs), (&qs, &ps)][axis];
            for x in 0..dims.0 {
                for y in 0..dims.1 {
                    fsd.segs[(axis,x,y)] = LineBoundary::compute(c1[x], c2[y], c2[y+1], eps);
                    // Sanity check by the relation on the existence of a cornerpoint in the FSD and the curve points being within eps distance.
                    if let Some(LineBoundary { a, b }) = fsd.segs[(axis,x,y)] {
                        if a == 0. {
                            assert!(c1[x].distance(c2[y]) < eps);
                        }
                        if c1[x].distance(c2[y]) < eps {
                            assert!(a == 0.);
                        }
                        if b == 1. {
                            assert!(c1[x].distance(c2[y+1]) < eps);
                        }
                        if c1[x].distance(c2[y+1]) < eps {
                            assert!(b == 1.);
                        }
                    } else {
                        assert!(c1[x].distance(c2[y])   >= eps - 5.*0.0001);
                        assert!(c1[x].distance(c2[y+1]) >= eps - 5.*0.0001);
                    }
                }
            }
        }

        // Constructing corners.
        for i in 0..n {
            for j in 0..m {
                fsd.corners[(i,j)] = ps[i].distance(qs[j]) < eps;
            }
        }

        fsd
    }

    /// Compute reachable space diagram out of a free space diagram.
    pub fn to_rsd(&self) -> Self {
        let fsd = self;
        let n = fsd.n;
        let m = fsd.m;
        assert!(!fsd.is_rsd); // Sanity check: Check it is already an RSD.
        let mut rsd = FSD::empty(n, m);
        rsd.is_rsd = true;

        // Initiate whole left FSD border (we seek partial curve).
        rsd.segs.slice_mut(s![0, 0, ..]).assign(&fsd.segs.slice(s![0, 0, ..]));

        // Initiate first lower-left horizontal RSD border (since neither prev, para, nor orth exists).
        if let Some(LineBoundary { a, b }) = fsd.segs[(1, 0, 0)] {
            if a == 0. {
                rsd.segs[(1, 0, 0)] = Some(LineBoundary { a, b });
            }
        }

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
                                rsd.segs[curr] = LineBoundary::union(rsd.segs[curr], LineBoundary::new(a.max(a_), b));
                            }
                        }
                    } 
                    if let Some(prev) = opt_prev { 
                        if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[prev] {
                            if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                                if b_ == 1. && a == 0. {
                                    rsd.segs[curr] = LineBoundary::union(rsd.segs[curr], LineBoundary::new(0., b));
                                }
                            }
                        }
                    }
                }
            }
        }

        rsd
    }

    /// Check for a partial curve match. 
    /// 
    /// Note: Should be appied to a reachability-space diagram.
    pub fn check_pcm(&self) -> bool {
        let rsd = if self.is_rsd { self } else { &self.to_rsd() };
        rsd.segs.slice(s![0,rsd.n-1,..]).iter().any(|b| b.is_some())
    }

    /// Compute steps to walk along curves for partial matching solution.
    /// 
    /// Note: Should be appied to a reachability-space diagram.
    pub fn pcm_steps(&self) -> Result<Option<Vec<(f64,f64)>>, String> {

        let rsd = if self.is_rsd { self } else { &self.to_rsd() };
        let n = rsd.n;
        let m = rsd.m;

        let mut steps = vec![];
        let mut curr = (0, 0, 0, 0.);

        // Seek lowest non-empty boundary on right side of the RSD.
        // (Basically performs PCM existence check as well.)
        let mut found = false;
        for (_j, lb) in rsd.segs.slice(s![0,n-1,..]).iter().enumerate() {
            if let Some(LineBoundary { a, b }) = lb {
                let x = n - 1;
                let y = _j;
                let off = *a;
                curr = (0, x, y, off);
                found = true;
                break;
            }
        }
        if !found { return Ok(None) }

        // Final step is found.
        steps.push(position_to_ij(curr));

        // Walk backwards. (Walk greedily, it should not matter).
        while !(position_on_left_boundary(curr)) { // Walk while we're not at the start position of P.

            // println!("curr: {curr:?}");
            if rsd.segs[position_to_seg(curr)].is_none() {
                return Err(format!("Current segment while walking ({curr:?}) should not be empty."));
            }

            // Try to walk backwards.
            let (axis, x, y, off) = curr;
            let mut states = vec![curr];
            if off == 0. { // In this case we can decide to walk both directions.
                states.push((1-axis, y, x, off))
            }

            let mut next = None;
            for (axis, x, y, off) in states {
                if next.is_none() {
                    let opt_prev = if off == 0. && y > 0 { 
                        Some((axis, x, y-1)) 
                    } else { None }; // previous.
                    let opt_para = if off >  0. && x > 0 { 
                        Some((axis, x-1, y)) 
                    } else { None }; // parallel.
                    let opt_orth = if x > 0 { 
                        Some((1-axis, y, x-1))
                    } else { None }; // orthogonal.

                    // Attempt to walk to previous.
                    if let Some(prev) = opt_prev {
                        if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[prev] {
                            if off == 0. && b_ == 1. {
                                // println!("prev");
                                let (axis, x, y) = prev;
                                let off = a_;
                                next = Some((axis, x, y, off));
                            }
                        }
                    }

                    // Attempt to walk to orthogonal.
                    if next.is_none() && let Some(orth) = opt_orth {
                        if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[orth] {
                            if (off == 0. && b_ == 1.) || off > 0. {
                                // println!("orth");
                                let (axis, x, y) = orth;
                                let off = a_;
                                next = Some((axis, x, y, off));
                            } 
                        }
                    } 

                    // Attempt to walk to parallel.
                    if next.is_none() && let Some(para) = opt_para {
                        if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[para] {
                            if off >= a_ {
                                // println!("para");
                                let (axis, x, y) = para;
                                let off = a_;
                                next = Some((axis, x, y, off));
                            }
                        }
                    }
                }
            }

            if next.is_none() {
                return Err(format!("Should find next step in backwards walk at {curr:?}.\n{rsd:?}"));
            }
            // println!("next: {next:?}");
            curr = next.unwrap();
            steps.push(position_to_ij(curr));

        }

        steps.reverse();
        Ok(Some(steps))
    }

}


// ==============================
// === Partial Curve Matching ===
// ==============================

/// Returns any subcurve of qs (if it exists) with Fréchet distance to ps below threshold epsilon.
#[pyfunction]
pub fn partial_curve(ps: Curve, qs: Curve, eps: f64) -> Result<Option<(f64, f64)>, PyErr> {
    let fsd = FSD::new(ps, qs, eps);
    let rsd = fsd.to_rsd();
    let opt_steps = rsd.pcm_steps().map_err(|str| PyErr::new::<PyTypeError, _>(str))?;
    if opt_steps.is_none() { 
        Ok(None) 
    } else {
        let steps = opt_steps.unwrap();
        let start= steps[0].1;
        let end = steps.last().unwrap().1;
        Ok(Some((start, end)))
    }
}


/// Node identifier.
type NID = usize;

/// Edge identifier.
type NIDPair = (usize, usize);
type EID = usize;

/// Alternative, simplified datastructure for Free-Space Diagram.
/// 
/// This one is easier to extend in comparison to ArrayView which is not intended to append rows/columns.
type FSD2 = Vec<Vec<Vec<OptLineBoundary>>>; // (x, y, horizontal|vertical)

/// A sequence of nodes to walk along the graph.
type Path = Vec<NID>;

/// Graph structure which stores relevant information a simple undirected graph.
#[pyclass]
pub struct Graph {
    /// Nodes with (x,y) coordinate.
    // nodes: Map<NID, Vector>,
    /// Adjacent nodes.
    adj: Map<NID, Set<NID>> ,

    /// List NIDs in graph.
    /// 
    /// Note: Since fixed position in vector this functions as a index as well.
    nid_list: Vec<NID>,
    /// Back-link for NIDs.
    nid_map: Map<NID, usize>,
    /// Node vectors.
    vec_list: Vec<Vector>,

    /// Back-link for EIDs.
    eid_map: Map<NIDPair, EID>,
    /// Map edge to unique identifier.
    /// 
    /// Note:  Since fixed position in vector this functions as a index as well.
    eid_list: Vec<NIDPair>,
}

impl Graph {

    /// Store a graph struct, re-used as we apply to different curves.
    pub fn new(vertices: Vec<(NID, Vector)>, edges: Vec<(NID, NID)>) -> Self {

        // Construct node information.
        let mut nid_list = Vec::new();
        let mut vec_list = Vec::new();
        let mut nid_map = Map::new();
        let mut i = 0;
        for (j, v) in &vertices {
            nid_list.push(j.clone());
            vec_list.push(v.clone());
            nid_map.insert(j.clone(), i);
            i += 1;
        }
        
        // Sanity check on NID backlink.
        assert_eq!(nid_list.len(), nid_map.len());
        for i in 0..nid_list.len() { 
            let nid = nid_list[i];
            assert_eq!(nid, *nid_map.get(&nid).unwrap());
        }

        // Sanity check on NID vectors.
        assert_eq!(nid_list.len(), vec_list.len());
        for (nid, v) in &vertices {
            let i = *nid_map.get(nid).unwrap();  
            assert_eq!(*v, vec_list[i]);
        }

        // Construct adjacency map.
        let mut adj: Map<NID, Set<NID>> = Map::new();
        for (u, v) in edges {

            if !adj.contains_key(&u) { adj.insert(u, Set::new()); }
            if !adj.contains_key(&v) { adj.insert(v, Set::new()); }

            adj.get_mut(&u).unwrap().insert(v);
            adj.get_mut(&v).unwrap().insert(u);

        }

        // Construct edge information.
        let mut eid_list = Vec::new();
        let mut eid_map = Map::new();
        let mut i = 0;
        for u in adj.keys() {
            for v in adj.get(u).unwrap() {
                eid_list.push((*u, *v));
                eid_map.insert((*u, *v), i);
                i += 1;
            }
        }

        // Sanity check edge information.
        for i in 0..eid_list.len() {
            let eid = eid_list[i];
            assert_eq!(i, *eid_map.get(&eid).unwrap());
        }

        Graph { nid_list, nid_map, vec_list, adj, eid_list, eid_map }
    }


}


fn fsd_height(fsd: &FSD2) -> usize {
    fsd.len()
}


fn fsd_width(fsd: &FSD2) -> usize {
    fsd[0].len()
}


/// Propagate bottom cell boundary reachability to the right while continuous.
pub fn propagate_bottom_row(mut fsd: FSD2) -> FSD2{
    let y = 0;
    let n = fsd_width(&fsd);

    // We propagate right while we have full boundaries.
    let mut i = 0;
    for x in 0..n { // n-1 latest width but n is set to None anyways.

        let horizontal = fsd[y][x][0];
        if horizontal.is_none() {
            break;
        }

        let LineBoundary { a, b } = horizontal.unwrap();
        if a > 0. {
            break;
        } 

        i += 1;
        if b < 1. {
            break;
        }
    }

    // Set unreachable cell boundaries to none.
    for x in i..n {
        fsd[y][x][0] = None;
    }

    fsd
}


/// Compute reachability in right and top cell boundary.
/// 
/// Left and bottom boundary are from RSD.
/// Right and top boundary are from FSD.
pub fn propagate_cell_reachability(bottom: OptLineBoundary, left: OptLineBoundary, right: OptLineBoundary, top: OptLineBoundary) -> (OptLineBoundary, OptLineBoundary) {

    // Decide reachability of right cell boundary based on bottom and left cell boundary.
    let goal_right = if bottom.is_some() {
        right
    } else if right.is_none() || left.is_none() {
        None
    } else {
        let LineBoundary { a, b } = left.unwrap();
        let LineBoundary { a: c, b: d } = right.unwrap();
        LineBoundary::new(a.max(c), d)
    };

    // Decide reachability of top cell boundary based on bottom and left cell boundary.
    let goal_top = if left.is_some() {
        top
    } else if top.is_none() || bottom.is_none() {
        None
    } else {
        let LineBoundary { a, b } = bottom.unwrap();
        let LineBoundary { a: c, b: d } = top.unwrap();
        LineBoundary::new(a.max(c), d)
    };
    
    
    (goal_right, goal_top)
}


/// Convert an FSD into an RSD.
pub fn fsd_to_rsd(mut fsd: FSD2, propagate: bool) -> FSD2 {
    
    let m = fsd_height(&fsd);
    let n = fsd_width(&fsd);

    // Initiate bottom row. (Leave left boundary untouched, because we do PCM.)
    if propagate {
        fsd = propagate_bottom_row(fsd);

        if DEBUG {
            println!("fsd with bottom row propagated:");
            print_fsd(&fsd);
            println!("");
        }
    }

    // Walk cells from left to right, bottom to top, and propagate reachability within cell boundaries.
    for y in 0..m-1 {
        for x in 0..n-1 {
            let bottom = fsd[y][x][0];
            let left   = fsd[y][x][1];
            let top    = fsd[y+1][x][0];
            let right  = fsd[y][x+1][1];

            let (right, top) = propagate_cell_reachability(bottom, left, right, top);
            fsd[y+1][x][0] = top;
            fsd[y][x+1][1] = right;
        }
    }

    fsd
}


/// Check whether the second [`OptLineBoundary`] is a subset of the first [`OptLineBoundary`].
pub fn lb_is_subset(lb1: OptLineBoundary, lb2: OptLineBoundary) -> bool {
    if lb2.is_none() { // If lb2 is none, lb1 is guaranteed to be larger.
        return true;
    } else if lb1.is_none() { // If lb2 is not none but lb1 is none, then lb2 must be larger.
        return false;
    } else { // If both do have some interval, simply check the subset of the interval.
        let lb1 = lb1.unwrap();
        let lb2 = lb2.unwrap();
        return lb1.has_subset(lb2);
    } 
}


/// Extract top boundary.
pub fn extract_top(fsd: &FSD2) -> Vec<OptLineBoundary> {
    let mut top = vec![];
    let n = fsd_width(fsd);
    let m = fsd_height(fsd);
    for x in 0..n {
        top.push(fsd[m-1][x][0]);
    }
    return top;
}


/// Returns true if the top is empty.
pub fn is_top_empty(fsd: &FSD2) -> bool {
    !extract_top(fsd).into_iter().any(|lb| lb.is_some())
}

#[pyfunction]
pub fn make_graph(vertices: Vec<(NID, Vector)>, edges: Vec<(NID, NID)>) -> Graph {
    Graph::new(vertices, edges)
}

/// Match curve against a graph with epsilon threshold.
/// 
/// todo (optimization): Lazily evaluate FDij's, initiate by checking solely the left boundary of every FDij.
#[pyfunction]
pub fn partial_curve_graph(graph: &Graph, curve: Curve, eps: f64) -> Result<Option<Vec<NID>>, PyErr> {
    let n = curve.len();
    // Map eid to vector index (such as used for FDu, for which having a set is problematic (a technically)).

    // Free-Space Lines. (Per node do we have a one-dimensional line.)
    let mut FDus: Vec<Vec<OptLineBoundary>> = vec![];
    for u in 0..graph.nid_list.len() {
        // Construct FDu.
        let mut FDu = vec![];
        for i in 0..n-1 {
            // Construct.
            let p = graph.vec_list[u];
            let q0 = curve[i];
            let q1 = curve[i+1];
            let lb = LineBoundary::compute(p, q0, q1, eps);
            FDu.push(lb);
        }
        FDus.push(FDu);
    }
    // todo: Sanity check FDu's.
    
    // Shortcut Pointers. (Per node (per FDu) we have backward and forward shortcut pointers.)
    let mut SPus: Vec<Vec<(Option<usize>, Option<usize>)>> = vec![];
    for u in 0..graph.nid_list.len() {

        // Construct SPu.
        let mut SPu = vec![];
        let FDu = &FDus[u];
        let mut i = 0;
        let mut k = 0;
        for j in 0..n {

            // Walk k to next non-empty.
            if j == k { 
                while k < n && FDu[k].is_none() {
                    k += 1;
                }
            }

            // Set index.
            let mut opt_left = None;
            let mut opt_right = None;
            if j > 0 && FDu[i].is_some() { // Use is_some since index 0 may be empty.
                opt_left = Some(i);
            }
            if k < n {
                opt_right = Some(k);
            }
            
            // Walk i to current non-empty.
            if FDu[j].is_some() { 
                i = j;
            }

            SPu.push((opt_left, opt_right));
        }
        SPus.push(SPu);
    }
    // todo: Sanity check SPu's.
    if sanity_check {

    }

    // Reachability Pointers. 
    let mut RPuvs = vec![];
    // The problem with our reachability pointers is the assumption we start at the bottom of the cell.
    // If we start on the left boundary, we may not walk that far to the right.
    let mut RPuv0s = vec![];
    for (u, v) in &graph.eid_list {

        let u = *graph.nid_map.get(u).unwrap();
        let v = *graph.nid_map.get(v).unwrap();
        
        // First construct spikes.
        let mut ais = vec![1.]; // Lower spike. Initiate first to non-existent.
        let mut bis = vec![0.]; // Upper spike.
        for i in 1..n { // Do it up to n (relevant for partial curve matching).
            let p = curve[i];
            let q0 = graph.vec_list[u];
            let q1 = graph.vec_list[v];
            if let Some(LineBoundary { a, b }) = LineBoundary::compute(p, q0, q1, eps) {
                ais.push(a);
                bis.push(b);
            } else {
                ais.push(1.);
                bis.push(0.);
            }
        }

        // Construct RCuv. (Reachable Cell index in v starting at bottom of u.)
        let mut RCuv = vec![];
        let mut RPuv0 = None;
        let mut S: VecDeque<(usize, f64)> = VecDeque::new(); // Partial maximum stack. Initiate with top at the left boundary.
        for i in 1..n { // Checking first index is useless.
            let m = S.len();
            let ai = ais[i];
            let bi = bis[i];
            
            if ais[0] < 1. && RPuv0.is_none() && bi < ais[0] {
                RPuv0 = Some((0, i - 1));
            }

            // Check for a cutoff and directly clean up stack.
            let mut _s = None;
            while bi < S.back().unwrap_or(&(0, 0.)).1 {
                _s = S.pop_back();
            }

            // Push the reachability index till the cutoff point.
            if let Some((l, al)) = _s {
                while RCuv.len() < l {
                    RCuv.push(i - 1);
                }
            }

            // Update local maximum pointer.
            while ai >= S.front().unwrap_or(&(0, 1.)).1 {
                S.pop_front();
            }
            S.push_front((i, ai));
        }
        // Push remaining values.
        while RCuv.len() < n {
            RCuv.push(n);
        }
        RPuv0s.push(RPuv0);

        // Sanity check RCuv.
        assert_eq!(RCuv.len(), n);

        // Construct Jls.
        let mut Jls = vec![];
        for i in 0..n {
            if FDus[u][i].is_some() {
                Jls.push(Some(i));
            } else if let Some(l) = SPus[v][i].1 && l <= RCuv[i] {
                Jls.push(Some(l));
            } else {
                Jls.push(None);
            }
        }

        // Construct Jrs.
        let mut Jrs = vec![];
        for i in 0..n {
            let l = RCuv[i];
            if FDus[v][l].is_some() {
                Jrs.push(Some(l));
            } else if let Some(k) = SPus[v][i].0 && k >= i && k <= l {
                Jrs.push(Some(k));
            } else {
                Jrs.push(None);
            }
        }

        let mut RPuv = vec![];
        for (Jl, Jr) in zip(Jls, Jrs) {
            // Sanity check that either both let and right exist or neither do.
            assert_eq!(Jl.is_some(), Jr.is_some());
            RPuv.push(if Jl.is_none() {
                None
            } else {
                Some((Jl.unwrap(), Jr.unwrap()))
            });
        }
        RPuvs.push(RPuv);
    }

    // Sweep line.
    // Initialize event buckets.
    let mut event_buckets = vec![];
    for i in 0..n {
        // Note how we use a reversed since we want to pop lowest a's first (we sweep from left to right).
        let bucket: BinaryHeap<Reverse<PathPointer>> = BinaryHeap::new();
        event_buckets.push(bucket);
    }

    // Initiate with non-empty left boundaries.
    for (eid, (u, v)) in graph.eid_list.iter().enumerate() {
        let u = *graph.nid_map.get(u).unwrap();
        let v = *graph.nid_map.get(v).unwrap();
        let _eid = *graph.eid_map.get(&(u, v)).unwrap();
        assert_eq!(eid, _eid);
        let p = curve[0];
        let q0 = graph.vec_list[u];
        let q1 = graph.vec_list[v];
        if let Some((_, k)) = RPuv0s[eid] {
            if k == n {
                let mut path = vec![u, v];
                return Ok(Some(path));
            }

            for i in 0..=k {
                if let Some(LineBoundary { a, b }) = FDus[v][i] {
                    // Push entire interval.
                    let mut bucket = event_buckets.get_mut(i).unwrap();
                    bucket.push(Reverse(PathPointer { a, from: vec![u], curr: v }));
                }
            }
        }
    }

    // Start sweeping!
    let mut run = true;
    let mut i = 0;
    let mut x = 0.; // for sanity checking
    let mut evicted = Set::new(); // Tracking evicted NIDs to prevent pushing back nodes onto the current event bucket which have already been processed for the current interval.
    while i < n { // Loop is called after every FDuv interval processed or an empty event buffer occurring.

        // Obtain next event for our sweep line.
        let opt_pathpointer = {
            let current_bucket = event_buckets.get_mut(i).unwrap();
            if current_bucket.is_empty() { // We have to check next bucket.
                i += 1; // Continue to next curve interval.
                evicted = Set::new(); // Thrash evicted nodes for new event bucket.
                None
            } else {
                Some(current_bucket.pop().unwrap().0)
            }
        };

        if opt_pathpointer.is_none() {
            continue;
        }

        // Take out a lowest value.
        let PathPointer { a, from, curr } = opt_pathpointer.unwrap();

        // Sanity check we are sweeping upwards.
        assert!(x <= i as f64 + a);
        x = i as f64 + a;

        // Try to walk to adjacents
        //                                             left reachable                        right reachable                 
        //                                             index top                             index top                         
        //                                                   j                                     k                               
        //    .------------------.- - - - - - - - - .------------------.- - - - - - - - - .------------------.
        //    |                  |                  |                  |                  |                  |
        //    |                  |                  |                  |                  |                  |
        //    |                  |                  |                  |                  |                  |
        //    .------------------.- - - - - - - - - .------------------.- - - - - - - - - .------------------.
        //              i
        //      curr index bottom
        // 
        let u = curr;
        evicted.insert(u);
        for v in graph.adj.get(&u).unwrap().clone() {
            let eid = graph.eid_map.get(&(u, v)).unwrap().clone();
            let RPuv = RPuvs[eid][i];
            if RPuv.is_none() {
                continue;
            }

            let (j, k) = RPuv.unwrap();

            // todo improve logic flow here below.

            // Obtain top left directly alongside offset (which can be shifted to the right due to starting interval at bottom).
            let top_left_with_offset = if j == i { 
                let top = FDus[v][i].unwrap();
                if top.b >= a {
                    Some((i, a.max(top.a)))
                } else if let Some(j) = SPus[v][i].1 && j <= k { // We must traverse to the right.
                    let top = FDus[v][j].unwrap();
                    Some((j, top.a))
                } else { 
                    None
                }
            } else { // We walk at least one cell to the right so we can just pick the start of this index.
                let top = FDus[v][j].unwrap();
                Some((j, top.a))
            };

            let top_right = if top_left_with_offset.is_none() {
                // Sanity check.
                assert!(SPus[v][k].0.is_none() || SPus[v][k].0.unwrap() < i || (SPus[v][k].0.unwrap() == i && FDus[v][i].unwrap().b < a));
                None
            } else {
                // Start at final index and walk left till we find something.
                if FDus[v][k].is_some() {
                    Some(k)
                } else {
                    let k = SPus[v][k].0.unwrap();
                    assert!(k >= i);
                    Some(k)
                }
            };

            if let Some(k) = top_right && k == n {
                let mut path = from.clone();
                path.push(v);
                return Ok(Some(path));
            }

            // Create new sweep line events.
            if top_left_with_offset.is_some() {
                let (j, a) = top_left_with_offset.unwrap();
                let k = top_right.unwrap();

                // Push top left.
                if j == i && evicted.contains(&v) { // Only add if we haven't evicted already in current event bucket interval.

                } else {
                    let bucket = event_buckets.get_mut(j).unwrap();
                    let extracted = extract_bucket_nid(bucket, v);
                    if let Some(path_pointer) = extracted {
                        if path_pointer.a <= a { // Ignore our just created event (reinsert old one).
                            bucket.push(Reverse(path_pointer));
                        } else { // Add our event (overwrite previously stored), because a is lower.
                            let mut _from = from.clone();
                            _from.push(u);
                            bucket.push(Reverse(PathPointer { a, from: _from, curr: v }));
                        }
                    }
                }

                // Iterate onwards.
                for l in j+1..=k  {
                    let a = FDus[v][l].unwrap().a;
                    let bucket = event_buckets.get_mut(j).unwrap();
                    let extracted = extract_bucket_nid(bucket, v);
                    if let Some(path_pointer) = extracted {
                        assert_eq!(path_pointer.a, a);
                        bucket.push(Reverse(path_pointer)); // Ignore our just created event (reinsert old one).
                    } else {
                        let mut _from = from.clone();
                        _from.push(u);
                        event_buckets.get_mut(l).unwrap().push(Reverse(PathPointer { a, from: _from, curr: v }));
                    }
                }
            }
        }
    }

    Ok(None)
}


/// Print OptLineBoundary (for debugging purposes).
fn print_lb(opt_lb : OptLineBoundary) {
    if opt_lb.is_none() {
        print!("(     -     )");
    } else {
        let LineBoundary { a, b } = opt_lb.unwrap();
        print!("({a:.2} - {b:.2})");
    }
}


/// Print FSD2 (for debugging purposes).
fn print_fsd(fsd: &FSD2) {
    let n = fsd_width(fsd);
    let m = fsd_height(fsd);
    let offset = 11;
    // (0..10).rev()
    for y in (0..m).rev() {
        for a in [1, 0] {
            if a == 0 {
                print!("             "); // 13
            }
            for x in 0..n {
                // let mut space = String::from("");
                // for i in 0..10*x {
                //     space.push_str(" ");
                // }
                // if a == 1 {
                //     for i in 0..5 {
                //         space.push_str(" ");
                //     }
                // }
                // print!("{space}.");
                print_lb(fsd[y][x][a]);
                print!("             "); // 13
            }
            println!();
        }
    }
}


/// Convert a Free-Space Diagram boundary into a Shortcut Pointer list.
fn free_space_line_to_shortcut_pointers_next(FDi: Vec<OptLineBoundary>) -> Vec<(Option<NID>, Option<NID>)> {
    // todo
    vec![]
}


fn free_space_row_into_reachable_interval(FDij: FSD2) -> Vec<(Option<NID>, Option<NID>)> {
    // todo
    vec![]
}


const sanity_check: bool = true;

// Construct FDi.
// Construct SPi.
// Construct RPij.
// Construct RIij.
// Initiate Ci's and sweep line, maintain Ci's while sweeping to the right.


/// Ordered sweep line events to process while traversing the reachable points in the Free-Space Surface.
type EventBucket = BinaryHeap<Reverse<PathPointer>>;

fn does_bucket_contain_nid(bucket: BinaryHeap<Reverse<PathPointer>>, nid: usize) -> bool {
    bucket.into_vec().iter().any(|Reverse(PathPointer { a, from, curr })| *curr == nid)
}

/// Extract PathPointer from the EventBucket if such exists with specified nid.
/// 
/// Note: Quite inefficient because we fully reconstruct the bucket (which concerns ordering when pushing).
fn extract_bucket_nid(bucket: &mut EventBucket, nid: usize) -> Option<PathPointer> {
    let mut reinsert = vec![];
    let mut extracted = None;
    while let Some(Reverse(pathpointer)) = bucket.pop() {
        if pathpointer.curr != nid {
            reinsert.push(Reverse(pathpointer));
        } else {
            extracted = Some(pathpointer);
        }
    }
    for element in reinsert {
        bucket.push(element);
    }
    extracted
}


/// Pointer living at FDu within the event bucket i.
/// When sweeping we send 
#[derive(Clone)]
struct PathPointer {
    /// This is the offset which could be endorsed by the bottom interval if it propagates to the same cell.
    a: f64,
    /// The walk started from some vertex and some curve interval (we don't care from what curve interval, we only care for reconstructing path).
    from: Vec<NID>,
    /// The current index we walk to.
    curr: NID,
}

impl PartialEq for PathPointer {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a
    }
}

impl Eq for PathPointer {

}

impl PartialOrd for PathPointer {
    fn ge(&self, other: &Self) -> bool {
        self.a >= other.a
    }
    fn gt(&self, other: &Self) -> bool {
        self.a > other.a
    }
    fn le(&self, other: &Self) -> bool {
        self.a <= other.a
    }
    fn lt(&self, other: &Self) -> bool {
        self.a < other.a
    }
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.a < other.a {
            Some(Ordering::Less)
        } else if self.a > other.a {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl Ord for PathPointer {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.a < other.a {
            Ordering::Less
        } else if self.a > other.a {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}


pub mod prelude {   
    pub use crate::{OptLineBoundary, LineBoundary, Vector, FSD, partial_curve, EPS, Curve, Graph, partial_curve_graph, make_graph};
}



#[test]
fn test_taking_path_works() {
    let p1 = Vector::new(0., 0.); 
    let p2 = Vector::new(0.5, 0.5);
    let p3 = Vector::new(1., 1.); 
    let ps = vec![p1, p3];
    let vertices = vec![(1, p1), (2, p2), (3, p3)];
    let edges = vec![(1,2), (2,3)];
    let graph = Graph::new(vertices, edges);
    let result = partial_curve_graph(&graph, ps, 0.01).unwrap();
    assert!(result.is_some());
    let result = result.unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}
