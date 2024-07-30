#![feature(let_chains)]
/// Author: jvtubergen
/// * This code has copied pyfrechet for computing the FSD.
/// The choice to switch to Rust is an arbitrary personal choice:
/// * Building rust code is fun.
/// * Practice with linking Rust code to Python (thus building python packages using rust code).
/// * Potentially upgrading to parallel implementation.
/// 
/// Use of code repo:
/// This code solves the PCMP (Partial Curve Matching Problem).
/// PCMP: Given a curve P, a curve Q and a distance threshold epsilon, is there some subcurve Q' of Q in such that the Fréchet distance between P and Q' is below epsilon? 
/// We limit ourselves to curves P and Q that are polygonal chains, and use the continuous strong Fréchet distance as a (sub)curve similarity/distance measure.
/// 
/// 
/// 
/// Design choices:
/// We assume the partial curve to be P, and the curve to find some subcurve to be in Q.
/// In the FSD (Free-Space Diagram) we place P on the horizontal axis and Q on the vertical axis.
/// Thus, since any subcurve of Q suffices, we may start at any vertical position on the left side of the FSD and end at any vertical position on the right side of the FSD.
/// The path still has to be monotonic though (since its the strong Fréchet distance)
/// 
/// Assumptions:
/// Assert all edges are non-zero length (thus no subsequent duplicated vertices).
/// 
use std::{fs, io::Read, iter::zip, ops::{Add, Mul, Sub}, path::Path};
extern crate rand;
use full_palette::{GREEN_400, RED_300};
use rand::{rngs::mock::StepRng, Rng};
use ndarray::{prelude::*, OwnedRepr};


use serde_derive::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use bincode;

use plotters::prelude::*;

const EPS: f64 = 0.00001;


// ==============
// === Vector ===
// ==============

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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


    
// =============
// === Curve ===
// =============

type Curve = Vec<Vector>;
/// Construct curve with n number of random points in f64 domain.
fn random_curve(n: usize, fieldsize: f64) -> Curve {
    let mut rng = rand::thread_rng();
    let c: Curve = (0..n).into_iter().map(|_| Vector {x: rng.gen_range(0.0..fieldsize), y: rng.gen_range(0.0..fieldsize)}).collect();
    // Chance of generating curve with zero-length edges is too small to actually counter..
    for i in 0..c.len()-1 {
        assert!((c[i+1] - c[i]).dot(c[i+1] - c[i]) > EPS * EPS);
    }
    c
}

/// Translate all points of a curve c1 by a vector q.
fn translate_curve(c1: Curve, q: Vector) -> Curve {
    c1.into_iter().map(|p| p + q).collect()
}

/// Add some random noise to curve points.
fn perturb_curve(c: Curve, deviation: f64) -> Curve {
    let mut rng = rand::thread_rng();
    let d = (1./2.0_f64.sqrt()) * deviation * deviation;
    c.into_iter().map(|p| p + d * rng.gen::<f64>() * Vector {x: 1., y: 1.} ).collect()
}

/// Compute curve length.
fn curve_length(c: &Curve) -> f64 {
    let mut length = 0.;
    for (p1, p2) in zip(c, &c[1..]) {
        length += p1.distance(*p2);
    }
    length
}


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

fn has_startpoint(opt_boundary: &OptLineBoundary) -> bool {
    if let Some(LineBoundary { a, b }) = opt_boundary {
        *a == 1.
    } else {
        false
    }
}
fn has_endpoint(opt_boundary: &OptLineBoundary) -> bool {
    if let Some(LineBoundary { a, b }) = opt_boundary {
        *b == 1.
    } else {
        false
    }
}


/// Intersecting two line boundaries. 
fn intersect(opt_b1: OptLineBoundary, opt_b2: OptLineBoundary) -> OptLineBoundary {
    if opt_b1.is_none() || opt_b2.is_none() {
        None
    } else {
        let b1 = opt_b1.unwrap();
        let b2 = opt_b2.unwrap();

        let a = b1.a.max(b2.a); 
        let b = b1.b.min(b2.b);

        if a < b { Some(LineBoundary { a, b }) }
        else { None }
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

        if a < b { Some(LineBoundary { a, b }) }
        else { None }
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
        let segs = Array3::from_shape_simple_fn((2,n,m), || None);
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
        let curves = [(&ps, &qs), (&qs, &ps)];

        // Constructing cell boundaries.
        for axis in 0..2 {
            let dims = fsd.dims[axis];
            let (c1, c2) = curves[axis];
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

    // /// Check boundary segments match with corner point existence.
    // fn is_corner_valid(&self, i: usize, j: usize) -> Result<(), String> {
    //     if i > 0 { // Check left.
    //         if has_endpoint(&self.horizontals[(j,i-1)]) != self.corners[(i,j)] {
    //             return Err(format!("Left of ({i},{j}) mismatches with corner."));
    //         }
    //     }
    //     Ok(())
    // }

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
                    rsd.segs[curr] = fsd.segs[curr];
                } else if let Some(para) = opt_para {
                    // Custom intersect.
                    if let Some(LineBoundary { a: a_, b: b_ }) = fsd.segs[para] {
                        if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                            rsd.segs[curr] = LineBoundary::new(a.max(a_), b);
                        }
                    }
                } else if let Some(prev) = opt_prev { 
                    if let Some(LineBoundary { a: a_, b: b_ }) = fsd.segs[prev] {
                        if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                            if b_ == 1. && a == 0. {
                                rsd.segs[curr] = LineBoundary::new(a_, b);
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
/// TODO: Provide subcurve of c2.
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
        let a = if j_off == 0. { 0 } else { 1 };

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
                }
            }
        }

        if next.is_none() && let Some(orth) = opt_orth {
            if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[orth] {
                if (y_off == 0. && b_ == 1.) || y_off > 0. {
                    next = Some(orth);
                    x_off = a_;
                    y_off = 0.;
                } 
            }
        }

        if next.is_none() && let Some(para) = opt_para {
            if let Some(LineBoundary { a: a_, b: b_ }) = rsd.segs[para] {
                if y_off >= a_ {
                    next = Some(para);
                    x_off = 0.;
                    y_off = a_;
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


// ===========================
// === Visualization logic ===
// ===========================

/// Drawing Free-Space Diagram as an image to disk.
fn draw_fsd(fsd: &FSD, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let margin = 20; // 20 pixels margin
    let width = fsd.n * 20 + 2 * margin;
    let height = fsd.m * 20 + 2 * margin;
    let filename = format!("{}.png", filename);
    let drawing_area = BitMapBackend::new(&filename, (width as u32, height as u32)).into_drawing_area();
    drawing_area.fill(&WHITE)?;

    let drawing_area = drawing_area.margin(20, 20, 20, 20);

    let n = fsd.n;
    let m = fsd.m;

    let unreachable = ShapeStyle {
        color: RED_300.mix(0.6),
        filled: true,
        stroke_width: 1,
    };
    let reachable = ShapeStyle {
        color: GREEN_400.mix(0.6),
        filled: true,
        stroke_width: 1,
    };

    let mut reachable_coords = vec![];
    let mut unreachable_coords = vec![];

    // Reachable corner points.
    for i in 0..n {
        for j in 0..m-2 {
            let coord = (20*i as i32, 20*j as i32);
            if let Some(LineBoundary { a, b }) = fsd.segs[(0,i,j)] && a == 0. {
                reachable_coords.push(coord);
            } else {
                unreachable_coords.push(coord);
            }
        }
        let j = m - 2;
        let coord = (20*i as i32, 20*j as i32);
        if let Some(LineBoundary { a, b }) = fsd.segs[(0,i,j)] && a == 0. {
            reachable_coords.push(coord);
        } else {
            unreachable_coords.push(coord);
        }
        let coord = (20*i as i32, 20*(j+1) as i32);
        if let Some(LineBoundary { a, b }) = fsd.segs[(0,i,j)] && b == 1. {
            reachable_coords.push(coord);
        } else {
            unreachable_coords.push(coord);
        }
    }


    let mut reachable_segments = vec![];
    let mut unreachable_segments = vec![];
    
    // Find reachable and unreachable segments.
    for j in 0..m {
        for i in 0..n {
            for axis in 0..2 {
                let (w,h) = fsd.dims[axis];
                let (x,y) = [(i,j), (j,i)][axis];
                if y < h {
                    let curr = (axis, x, y);
                    if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                        if a > 0. { unreachable_segments.push(vec![ (axis, x as f64, y as f64    ), (axis, x as f64, y as f64 + a) ]); }
                                      reachable_segments.push(vec![ (axis, x as f64, y as f64 + a), (axis, x as f64, y as f64 + b) ]);
                        if b < 1. { unreachable_segments.push(vec![ (axis, x as f64, y as f64 + b), (axis, x as f64, y as f64 + 1.) ]); }
                    } else {
                        unreachable_segments.push(vec![ (axis, x as f64, y as f64), (axis, x as f64, y as f64 + 1.) ])
                    }
                }
            }
        }
    }

    // Draw reachable and unreachable line segments.
    // println!("reachable:");
    for seg in reachable_segments {
        // println!("{seg:?}");
        let seg: Vec<(i32, i32)> = seg.into_iter().map(|(axis, x, y)| {
            // ((20.*x) as i32, (20.*y) as i32)
            if axis == 0 { ((20.*x) as i32, (20.*y) as i32) }
            else         { ((20.*y) as i32, (20.*x) as i32) }
        }).collect();
        drawing_area.draw(&Polygon::new(seg, reachable))?;
    }

    // println!("unreachable:");
    for seg in unreachable_segments {
        // println!("{seg:?}");
        let seg: Vec<(i32, i32)> = seg.into_iter().map(|(axis, x, y)| {
            // ((20.*x) as i32, (20.*y) as i32)
            if axis == 0 { ((20.*x) as i32, (20.*y) as i32) }
            else         { ((20.*y) as i32, (20.*x) as i32) }
        }).collect();
        drawing_area.draw(&Polygon::new(seg, unreachable))?;
    }

    // Draw reachable and unreachable cornerpoints.
    for coord in reachable_coords {
        drawing_area.draw(&Circle::new(coord, 1, reachable))?;
    }
    for coord in unreachable_coords {
        drawing_area.draw(&Circle::new(coord, 1, unreachable))?;
    }

    Ok(())
}




// =====================
// === Testing logic ===
// =====================

/// Check the presence/absence of fully covered line segments with free/unfree line segments.
fn check_corner_consistency(fsd: &FSD) -> Result<(), String> {
    let (n,m) = (fsd.n, fsd.m);
    for j in 0..m {
        for i in 0..n {
            let has_corner = fsd.corners[(i,j)];
            for axis in 0..2 {
                let (w,h) = fsd.dims[axis];
                let (x,y) = [(i,j), (j,i)][axis];
                let opt_curr = if y < h { Some((axis, x, y)) } else { None };
                let opt_prev = if y > 0 { Some((axis, x, y-1)) } else { None };

                if !has_corner {
                    if let Some(curr) = opt_curr {
                        if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                            if a == 0.0 {
                                return Err(format!("Start of boundary exists at {curr:?} while no corner at (({i},{j})) "));
                            }
                        }
                    }
                    if let Some(prev) = opt_prev {
                        if let Some(LineBoundary { a, b }) = fsd.segs[prev] {
                            if b == 1.0 {
                                return Err(format!("End of boundary exists at {prev:?} while no corner at (({i},{j})) "));
                            }
                        }
                    }
                } else { // if has_corner {
                    if let Some(curr) = opt_curr {
                        if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                            if a >= EPS {
                                return Err(format!("Start of boundary does not exist at {curr:?} while corner at (({i},{j})) "));
                            }
                        } else {
                            return Err(format!("Boundary does not exist at {curr:?} while corner at (({i},{j})) "));
                        }
                    }
                    if let Some(prev) = opt_prev {
                        if let Some(LineBoundary { a, b }) = fsd.segs[prev] {
                            if b < 1.0 - EPS {
                                return Err(format!("End of boundary does not exist at {prev:?} while corner at (({i},{j})) "));
                            }
                        } else {
                            return Err(format!("Boundary does not exist at {prev:?} while corner at (({i},{j})) "));
                        }
                    }
                } 
            }
        }
    }
    
    Ok(())
}

/// Check steps result is within distance.
fn check_steps(c1: Curve, c2: Curve, steps: Vec<(f64, f64)>, eps: f64) -> Result<(), String> {
    // Check distance while walking is within threshold.
    for (_i, _j) in steps {
        let i = _i.floor();
        let i_off = _i - i;
        let j = _j.floor();
        let j_off = _j - j;
        
        let p = if i_off == 0. { // Interpolate.
            c1[i as usize] 
        } else { (1. - i_off) * c1[i as usize] + i_off * c1[i as usize + 1] };

        let q = if j_off == 0. { // Interpolate.
            c2[j as usize] 
        } else { (1. - j_off) * c2[j as usize] + j_off * c2[j as usize + 1] };

        if !(p.distance(q) < eps + EPS) {
             return Err(format!("Distance at step ({i}+{i_off}, {j}+{j_off}) should be below threshold."));
        }
    }
    Ok(())
}

/// Test validity of running a state.
// fn run_test(state: State) -> Result<(), Box<dyn std::error::Error>> {
fn run_test(state: State) -> Result<(), String> {
    let State { ps, qs, eps } = state.clone();

    let fsd = FSD::new(ps.clone(), qs.clone(), eps);
    check_corner_consistency(&fsd)?;
    draw_fsd(&fsd, "fsd");

    let rsd = compute_rsd(fsd);
    draw_fsd(&rsd, "rsd");
    let partial = rsd_pcm(&rsd);
    println!("Is there a partial curve match?: {partial:?}.");

    let opt_steps = rsd_pcm_steps(&rsd)?;
    if partial && opt_steps.is_none() {
        return Err(format!("Should find steps if partial curve match is true."));
    }
    check_steps(ps, qs, opt_steps.unwrap(), eps)?;
    
    Ok(())
}


// ========================
// === IO functionality ===
// ========================

/// Testing state for storage/retrieval.
#[derive(Serialize, Deserialize, Clone)]
struct State {
    ps: Curve,
    qs: Curve,
    eps: f64
}

/// Listing files in subfolder.
fn list_files_in_subfolder<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(path_str) = path.to_str() {
                files.push(path_str.to_string());
            }
        }
    }

    Ok(files)
}

/// Write state to testdata folder as a new testcase to debug.
fn write_new_testcase(state: State) -> Result<(), Box<dyn std::error::Error>> {
    let bin = bincode::serialize(&state)?;
    fs::create_dir("testdata"); // Folder probably already exists, then will throw error.
    let files = list_files_in_subfolder("testdata")?;
    let n = files.len();
    let file_path = Path::new("testdata").join(format!("case_{n}.bin"));
    let mut file = File::create(file_path)?;
    file.write_all(&bin)?;
    Ok(())
}

/// Read states from disk, should represent testcases previously crashed (thus to debug).
fn read_cases() -> Result<Vec<State>, Box<dyn std::error::Error>> {
    let files = list_files_in_subfolder("testdata")?;
    let mut result = vec![];
    for file in files {
        let mut file = File::open(file)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer);
        let state = bincode::deserialize(&buffer)?;
        result.push(state);
    }

    Ok(result)
}


// ==================
// === Executable ===
// ==================

const DISCOVER: bool = false;
const RUN_COUNT: usize = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let cases = 
    if DISCOVER {
        (0..RUN_COUNT).map(|_| {
            let ps = random_curve(20, 2.);
            // let c2 = translate_curve(ps, Vector{ x: 3. , y: 1. });
            let qs = perturb_curve(ps.clone(), 1.);
            State { ps, qs, eps: 1. }
        }).collect()
    } else {
        let mut r = read_cases()?;
        r.truncate(RUN_COUNT);
        r
    };

    for (i, case) in cases.into_iter().enumerate() {
        let res_test = run_test(case.clone());
        if res_test.is_err() {
            // Print we got an error.
            println!("Test case {} failed. Error message:", i);
            println!("{:?}", res_test.unwrap_err());
            // Only write new tast case in disovery mode, 
            //   otherwise we are duplicating testcases 
            //   (writing new case we just read).
            if DISCOVER { 
                write_new_testcase(case);
            }
        }
    }

    Ok(())
}
