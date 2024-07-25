use std::{iter::zip, ops::{Add, Mul, Sub}, vec};
extern crate rand;
use rand::Rng;


#[derive(Debug, Clone, Copy)]
struct Vector {
    x: f64,
    y: f64
}
impl Vector {
    fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y
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
    
    // Copy over the left and lower boundary of the FSD to the RSD.
    rsd.verticals[0] = fsd.verticals[0].clone();
    rsd.horizontals[0] = fsd.horizontals[0].clone();

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

/// Construct curve with n number of random points in f64 domain.
fn random_curve(n: usize) -> Curve {
    let mut rng = rand::thread_rng();
    let c: Curve = (0..n).into_iter().map(|_| Vector {x: rng.gen_range(0.0..1000.), y: rng.gen_range(0.0..1000.)}).collect();
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
    let d = 1./2.0_f64.sqrt() * deviation * deviation;
    c.into_iter().map(|p| p + d * rng.gen::<f64>() * Vector {x: 1., y: 1.} ).collect()
}


fn main() {

    let c1 = random_curve(20);
    // let c2 = translate_curve(c1, Vector{ x: 3. , y: 1. });
    let c2 = perturb_curve(c1.clone(), 1.);

    let partial = partial_curve_matching(c1, c2, 1.);
    println!("{partial:?}");
}
