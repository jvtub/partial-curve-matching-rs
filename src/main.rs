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

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64
}
impl Sub for Point {
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
#[derive(Debug)]
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
    /// Vertical cell boundaries: free space of points of P with edges of Q. 
    verticals  : Vec<Vec<OptLineBoundary>>,
    /// Horizontal cell boundaries: free space of edges of P with points of Q.
    horizontals: Vec<Vec<OptLineBoundary>>
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

    verticals.reserve_exact(n); //
    horizontals.reserve_exact(m); // 

    // Construcing verticals (Columns of vertical cell boundaries).
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
        horizontals,
        verticals
    }
}

/// Assert all edges are non-zero length (thus no subsequent duplicated vertices).


/// Construct curve with n number of random points in f64 domain.
fn random_curve(n: u64) -> Curve {
    let mut rng = rand::thread_rng();
    (0..n).into_iter().map(|_| Vector {x: rng.gen_range(0.0..1000.), y: rng.gen_range(0.0..1000.)}).collect()
}


/// Translate all points of a curve c1 by a vector q.
fn translate_curve(c1: Curve, q: Vector) -> Curve {
    c1.into_iter().map(|p| p + q).collect()
}

/// Add some random noise to curve points.
fn perturb_curve(c: Curve, deviation: f64) -> Curve {
    let mut rng = rand::thread_rng();
    c.into_iter().map(|p| p + deviation * rng.gen::<f64>() * Vector {x: 1., y: 1.} ).collect()
}


fn main() {

    let c1 = random_curve(20);
    // let c2 = translate_curve(c1, Vector{ x: 3. , y: 1. });
    let c2 = perturb_curve(c1.clone(), 1.);

    println!("c1: {c1:?}");
    println!("-----------");
    println!("c2: {c2:?}");
    println!("-----------");

    let fsd = compute_fsd(c1, c2, 2.);


    println!("{fsd:?}");
}
