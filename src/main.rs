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
use std::{fs, io::Read, iter::zip, ops::{Add, Mul, Sub}, path::Path, vec};
extern crate rand;
use rand::Rng;

use serde_derive::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use bincode;


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

const EPS: f64 = 0.00001;


/// Free-Space Diagram.
#[derive(Debug)]
struct FSD {
    n: usize,
    m: usize,
    /// Vertical cell boundaries: free space of points of P with edges of Q. 
    verticals  : Vec<Vec<OptLineBoundary>>,
    /// Horizontal cell boundaries: free space of edges of P with points of Q.
    horizontals: Vec<Vec<OptLineBoundary>>
}
impl FSD {
    fn new(n: usize, m: usize) -> FSD {

        // Initiate empty verticals and horizontals.
        let mut verticals = vec![]; // Contains n rows, with m-1 intervals.
        verticals.resize_with(n, || {
            let mut col: Vec<Option<LineBoundary>> = vec![];
            col.resize_with(m-1, || { None });
            col
        });

        let mut horizontals = vec![]; // Contains m rows, with n-1 intervals.
        horizontals.resize_with(m, || {
            let mut row = vec![];
            row.resize_with(n-1, || { None });
            row
        });

        FSD { n, m, verticals, horizontals }
    }
}

    
/// Compute unit-distance free space between line segment p1p2 and point q.
fn compute_cell_boundary(p1: Vector, p2: Vector, q: Vector, eps: f64 ) -> OptLineBoundary {
    let dp = p2 - p1;
    // let dq = q2 - q1;
    let divisor = dp.dot(dp);

    let b = dp.dot(q - p1);
    let c = divisor * (p1.dot(p1) + q.dot(q) - 2. * p1.dot(q) - eps * eps);
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

type Curve = Vec<Vector>;


/// Compute the free-space diagram between curve P (points ps) and curve Q (points qs).
/// Placing P point indices on the horizontal axis and Q on the vertical axis.
fn compute_fsd(ps: Curve, qs: Curve, eps: f64) -> FSD {

    let n = ps.len();
    let m = qs.len();

    let mut horizontals= vec![]; // Contains m rows, with n-1 intervals.
    let mut verticals = vec![]; // Contains n rows, with m-1 intervals.

    horizontals.reserve_exact(m); // 
    verticals.reserve_exact(n); //

    // Construcing verticals (Intersection of point in P with line segments of Q).
    for i in 0..n {
        let mut col = vec![];
        col.reserve_exact(m-1);
        let p = ps[i];
        for (q1, q2) in zip(&qs, &qs[1..]) {
            let q1 = q1.to_owned();
            let q2 = q2.to_owned();
            let p = p.to_owned();
            let line_boundary = compute_cell_boundary(q1, q2, p, eps);
            col.push(line_boundary);
        }
        verticals.push(col);
    }

    // Construcing horizontals (Intersection of point in Q with line segments of P).
    for j in 0..m {
        let mut row = vec![];
        row.reserve_exact(n-1);
        let q = qs[j];
        for (p1, p2) in zip(&ps, &ps[1..]) {
            let p1 = p1.to_owned();
            let p2 = p2.to_owned();
            let q = q.to_owned();
            let line_boundary = compute_cell_boundary(p1, p2, q, eps);
            row.push(line_boundary);
        }
        horizontals.push(row);
    }

    FSD {
        n, m, 
        horizontals,
        verticals
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

        if a < b - EPS { Some(LineBoundary { a, b }) }
        else { None }
    }
}

/// Compute reachable space diagram out of a free space diagram.
fn compute_rsd(fsd: FSD) -> FSD {
    let mut rsd = FSD::new(fsd.n, fsd.m);
    
    // Walk horizontally and vertically till a partially empty boundary occurs.
    let mut i = 0;
    while i <= fsd.n - 1 {
        let element = &fsd.horizontals[0][i];
        if element.is_none() {
            break;
        }
        let LineBoundary { a, b } = element.unwrap();
        if a == 0. {
            rsd.horizontals[0][i] = Some(LineBoundary { a, b });
        } 
        if b < 1. {
            break;
        }
        i += 1;
    }
    
    let mut j = 0;
    while j <= fsd.m - 1 {
        let element = &fsd.verticals[0][j];
        if element.is_none() {
            break;
        }
        let LineBoundary { a, b } = element.unwrap();
        if a == 0. {
            rsd.verticals[0][j] = Some(LineBoundary { a, b });
        } 
        if b < 1. {
            break;
        }
        j += 1;
    }

    for i in 0..rsd.n-1 {
        for j in 0..rsd.m-1 {

            // Constructing upper horizontal boundary.
            if rsd.verticals[i][j].is_some() {
                // If the left boundary of the RSD is non-empty, all of the top boundary of the FSD can be reached due to convex cell.
                rsd.horizontals[j+1][i] = fsd.horizontals[j+1][i];
            } else {
                // Otherwise, cascade bottom horizontal boundary of the RSD alongside the horizontal boundary of the FSD.
                rsd.horizontals[j+1][i] = intersect(rsd.horizontals[j][i], fsd.horizontals[j+1][i]);
            }

            // Constructing vertical boundary to the right.
            if rsd.horizontals[j][i].is_some() {
                // If the lower boundary of the RSD is non-empty, all of the right boundary of the FSD can be reached due to convex cell.
                rsd.verticals[i+1][j] = fsd.verticals[i+1][j];
            } else {
                // Otherwise, cascade forward the boundary on the left of the RSD alongside the vertical boundary of the FSD.
                rsd.verticals[i+1][j] = intersect(rsd.verticals[i][j], fsd.verticals[i+1][j]);
            }
        }
    }

    rsd
}

/// Check whether c1 is a partial curve of c2
/// TODO: Provide subcurve of c2.
/// TODO: Provide step sequence.
fn partial_curve_matching(c1: Curve, c2: Curve, eps: f64) -> bool {
    let fsd = compute_fsd(c1, c2, eps);
    let rsd = compute_rsd(fsd);
    // Check for non-zero boundary on the right side of the reachability space diagram.
    rsd.verticals[rsd.n-1].iter().any(|b| b.is_some())
}

/// Given a reachability space diagram, find steps to walk along P and Q.
fn partial_curve_matching_path(rsd: FSD) -> Result<Option<Vec<(f64, f64)>>, &'static str> {
    let mut steps = vec![];
    let mut i = rsd.n - 1;
    let mut i_off = 0.;
    let mut j = 0;
    let mut j_off = 0.;
    let mut found = false;

    // Seek lowest non-empty boundary on right side of the RSD.
    for (_j, lb) in rsd.verticals[rsd.n-1].iter().enumerate() {
        if let Some(LineBoundary { a, b }) = lb {
            j = _j;
            j_off = *a;
            found = true;
            break;
        }
    }
    if !found { return Ok(None) }

    steps.push((i as f64 + i_off, j as f64 + j_off));

    // Walk backwards.
    while !(i == 0 && i_off == 0.) {
        let _v = (i as f64 + i_off, j as f64 + j_off);
        let _w = (i_off == 0., j_off == 0.);
        // println!("walk: {i}, {i_off}, {j}, {j_off}");
        // println!("walk: {_v:?}, {_w:?}");

        // Vertical/Horizontal/Corner.
        if !(i_off == 0. || j_off == 0.) { return Err("Not at any RSD boundary.") }

        if j == 0 && j_off == 0. { // At bottom of RSD, walk left.

            // Expect to have walked to start of node.
            if !(i_off == 0.) { return Err("At bottom of RSD but not at a cornerpoint.") }
            if !(i > 0) { return Err("Negative i.") }
            i -= 1;
        }
        else {

            if i_off > 0. { // At horizontal boundary line.
                if !(j > 0) { return Err("Negative j.") }
                j -= 1;
            } 
            else if j_off > 0. { // At vertical boundary line.
                if !(i > 0) { return Err("Negative i.") }
                i -= 1;
            }
            else { // At corner point. 
                if !(i > 0) { return Err("Negative i.") }
                if !(j > 0) { return Err("Negative j.") }
                i -= 1;
                j -= 1;
            }

            let left = rsd.verticals[i][j];
            let below = rsd.horizontals[j][i];

            let (first, second) = 
                if j_off > 0. || j_off == i_off { (below, left) } else { (left, below) };

            if let Some(LineBoundary { a, b }) = first {  
                if j_off > 0. || j_off == i_off { // Check horizontal first
                    j_off = 0.;
                    i_off = a;
                } else { // Check vertical first
                    j_off = a;
                    i_off = 0.;
                }
            } else if let Some(LineBoundary { a, b }) = second {  
                if j_off > 0. || j_off == i_off { // Check vertical second
                    i_off = 0.;
                    j_off = a;
                } else { // Check horizontal second
                    i_off = a;
                    j_off = 0.;
                }
            }
            else {
                return Err("Expect to walk either left or down.")
            }
        }
        
        steps.push((i as f64 + i_off, j as f64 + j_off));
    }

    steps.reverse();
    Ok(Some(steps))
}

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


#[derive(Serialize, Deserialize, Clone)]
struct State {
    c1: Curve,
    c2: Curve,
    eps: f64
}

const DISCOVER: bool = false;
const RUN_COUNT: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let cases = 
    if DISCOVER {
        (0..RUN_COUNT).map(|_| {
            let c1 = random_curve(20, 2.);
            // let c2 = translate_curve(c1, Vector{ x: 3. , y: 1. });
            let c2 = perturb_curve(c1.clone(), 1.);
            State { c1, c2, eps: 1. }
        }).collect()
    } 
    else {
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

/// Check steps result is within distance.
fn check_steps(c1: Curve, c2: Curve, steps: Vec<(f64, f64)>, eps: f64) -> bool {
    // Check distance while walking is within threshold.
    for (_i, _j) in steps {
        let i = _i.floor();
        let i_off = _i - i;
        let j = _j.floor();
        let j_off = _j - j;
        
        let p = if i_off == 0. { 
            c1[i as usize] 
        } else { (1. - i_off) * c1[i as usize] + i_off * c1[i as usize + 1] };

        let q = if j_off == 0. { 
            c2[j as usize] 
        } else { (1. - j_off) * c2[j as usize] + j_off * c2[j as usize + 1] };

        if !(p.distance(q) < eps + EPS) {
             return false;
        }
    }
    true
}

/// Test validity of running a state.
fn run_test(state: State) -> Result<(), Box<dyn std::error::Error>> {
    let State { c1, c2, eps } = state.clone();

    let partial = partial_curve_matching(c1.clone(), c2.clone(), eps);
    println!("{partial:?}");

    if partial { // Found some partial curve, compute steps to attain this.
        let fsd = compute_fsd(c1.clone(), c2.clone(), 1.);
        let rsd = compute_rsd(fsd);

        let res_opt_steps = partial_curve_matching_path(rsd);
        if res_opt_steps.is_err() {
            return Err("We expect no error, and we do expect to find a solution.".into());
        } 
        let opt_steps = res_opt_steps.unwrap();
        if opt_steps.is_none() {
            return Err("We expect no error, and we do expect to find a solution.".into());
        }
        let steps = opt_steps.unwrap();
        if !check_steps(c1, c2, steps, eps) {
            return Err("When reconstructing the path steps, the threshold is exceeded.".into());
        }
    }

    Ok(())
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