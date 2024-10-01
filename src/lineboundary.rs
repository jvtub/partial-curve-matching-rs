use crate::vector::Vector;

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
        // let p0 = q0 + t0 * v; // First point of intersection.
        // let p1 = q0 + t1 * v; // Second point of intersection.

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

/// LineBoundaries are ponentially empty which we denote with the None type.
pub type OptLineBoundary = Option<LineBoundary>;

/// Print OptLineBoundary (for debugging purposes).
pub fn print_lb(opt_lb : OptLineBoundary) {
    if opt_lb.is_none() {
        print!("(     -     )");
    } else {
        let LineBoundary { a, b } = opt_lb.unwrap();
        print!("({a:.2} - {b:.2})");
    }
}


