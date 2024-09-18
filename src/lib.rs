#![feature(let_chains)]
use std::ops::{Add, Div, Mul, Sub};
use ndarray::{prelude::*, OwnedRepr};
use pyo3::{exceptions::PyTypeError, prelude::*};
use serde_derive::{Deserialize, Serialize};
use std::collections::BTreeMap as Map;
use std::collections::BTreeSet as Set;
pub const EPS: f64 = 0.00001;

// ==============
// === Vector ===
// ==============

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
type EID = (usize, usize);

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
    nodes: Map<NID, Vector>,
    /// Adjacent nodes.
    adj: Map<NID, Set<NID>> 
}

impl Graph {

    /// Store a graph struct, re-used as we apply to different curves.
    pub fn new(vertices: Vec<(NID, Vector)>, edges: Vec<(NID, NID)>) -> Self {
        let mut nodes = Map::new();
        for (i, v) in vertices {
            nodes.insert(i, v);
        }
        let mut adj: Map<NID, Set<NID>> = Map::new();
        for (u, v) in edges {

            if !adj.contains_key(&u) { adj.insert(u, Set::new()); }
            if !adj.contains_key(&v) { adj.insert(v, Set::new()); }

            adj.get_mut(&u).unwrap().insert(v);
            adj.get_mut(&v).unwrap().insert(u);

        }
        Graph { nodes, adj }
    }

    pub fn curvature(&self, uv: EID) -> Curve {
        let (u, v) = uv;
        vec![self.nodes[&u], self.nodes[&v]]
    }

}

/// Compute FSD out of two curves (first curve on horizontal axis, second curve on vertical axis).
pub fn compute_fsd(ps: Curve, qs: Curve, eps: f64) -> FSD2 {

    let n = ps.len();
    let m = qs.len();
    let mut fsd = vec![];
    for y in 0..m {
        let mut row = vec![];
        for x in 0..n {
            let horizontal = if x == n - 1 { 
                None
            } else {
                LineBoundary::compute(qs[y], ps[x], ps[x+1], eps)
            };
            let vertical = if y == m - 1 {
                None
            } else {
                LineBoundary::compute(ps[x], qs[y], qs[y+1], eps)
            };
            row.push(vec![horizontal, vertical]);
        }
        fsd.push(row);
    }

    fsd
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
pub fn fsd_to_rsd(mut fsd: FSD2) -> FSD2 {
    
    let m = fsd_height(&fsd);
    let n = fsd_width(&fsd);

    // Initiate bottom row. (Leave left boundary untouched, because we do PCM.)
    fsd = propagate_bottom_row(fsd);

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
// pub fn compute(graph: Graph, ps: Curve, eps: f64) -> Result<Option<Vec<NID>>, String> {
pub fn partial_curve_graph(graph: &Graph, ps: Curve, eps: f64) -> Result<Option<Vec<NID>>, PyErr> {

    // Initiate FDij's and initial reachable paths.
    let mut FDijs : Map<EID, FSD2> = Map::new();
    let mut queue : Vec<(Path, Vec<OptLineBoundary>)> = vec![];
    let n = ps.len();
    for (i, js) in graph.adj.clone() { // Iterate all edges.
        for j in js { // These are all the possible starting points we could have.
            let eid: EID = (i, j);
            let fsd = compute_fsd(ps.clone(), graph.curvature(eid), eps);
            FDijs.insert(eid, fsd.clone());
            let rsd = fsd_to_rsd(fsd);
            if rsd[0][n-1][1].is_some() { // Non-empty right boundary found.
                return Ok(Some(vec![i, j]));
            }
            if !is_top_empty(&rsd) { // If we can reach top, we can increment and check next element.
                // Push horizontal top line segment.
                // We are able to start at this edge for pcm curve matching. Initiate queue for evaluation.
                queue.push((vec![i, j], extract_top(&rsd)));
            }
        }
    }

    // Per starting point, walk till we find a feasible path.
    let mut totalcount = 0;
    while queue.len() > 0 {
        let (path, top) = queue.pop().unwrap();
        // println!("Checking path: {path:?}.");
        let i = path.last().unwrap().clone();

        // Seek adjacent vertices which have not been visited yet.
        let adjacents = graph.adj.get(&i).unwrap().clone();
        let adjacents: Vec<NID> = adjacents.into_iter().filter(|&v| !path.contains(&v)).collect();

        // If we can reach next element, make walk (add new vertex to path).
        for j in adjacents.clone() {
            totalcount += 1;

            let eid: (usize, usize) = (i, j);
            let mut fsd = FDijs.get(&eid).unwrap().clone();

            // Sanity: Check top of RSD is a subset of bottom of FSD.
            for x in 0..fsd_width(&fsd) {
                let fsd_bot = fsd[0][x][0];
                let rsd_top = top[x];
                assert!(lb_is_subset(fsd_bot, rsd_top));
            }

            // Update bottom of fsd to top of rsd.
            for x in 0..n { 
                fsd[0][x][0] = top[x];
            }           

            // Walk from left to right and update rsd.
            let rsd = fsd_to_rsd(fsd);
            let mut path = path.clone();
            path.push(j); 

            if rsd[0][n-1][1].is_some() { // Non-empty right boundary found.
                return Ok(Some(path));
            }

            if !is_top_empty(&rsd) {
                // If not empty we push it to the stack.
                queue.push((path, extract_top(&rsd)));
            }
        }
    }
    println!("Total paths checkted: {totalcount}");

    Ok(None)
}


pub mod prelude {   
    pub use crate::{OptLineBoundary, LineBoundary, Vector, FSD, partial_curve, EPS, Curve, Graph, partial_curve_graph, make_graph};
}
}