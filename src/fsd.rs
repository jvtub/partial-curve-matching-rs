use ndarray::{s, Array2, Array3, ArrayBase, Dim, OwnedRepr};

use crate::{curve::Curve, lineboundary::OptLineBoundary, LineBoundary};


/// Position on the FSD considering axis.
type FSDPosition = (usize, usize, usize, f64);

/// Convert a FSD position into curve positions.
fn position_to_ij((axis, x, y, off): FSDPosition) -> (f64, f64) {
    [(x as f64, y as f64 + off), (y as f64 + off, x as f64)][axis]
}

/// Convert position into a FSD horizontal/vertical cell.
fn position_to_seg((axis, x, y, _): FSDPosition) -> (usize, usize, usize) {
    (axis, x, y)
}

/// Check whether position is on the left boundary of the FSD.
fn position_on_left_boundary((axis, x, y, _): (usize, usize, usize, f64)) -> bool {
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
                    //     println!("a {:?}", a); 
                    //     println!("b {:?}", b); 
                    //     println!("x {:?}", x); 
                    //     println!("y {:?}", y); 
                    //     println!("c1[x] {:?}", c1[x]); 
                    //     println!("c2[y] {:?}", c2[y]); 
                    //     println!("c2[y+1] {:?}", c2[y+1]); 
                    //     println!("eps {:?}", eps); 
                    //     println!("c1[x].distance(c2[y]) {:?}", c1[x].distance(c2[y])); 
                    //     println!("c1[x].distance(c2[y+1]) {:?}", c1[x].distance(c2[y+1])); 
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
                    //     println!("c1[x].distance(c2[y]) {:?}", c1[x].distance(c2[y])); 
                        assert!(c1[x].distance(c2[y])   >= eps - 5.*0.0001);
                    //     println!("c1[x].distance(c2[y+1]) {:?}", c1[x].distance(c2[y+1])); 
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
                        if let Some(LineBoundary { a: a_, b: _b }) = rsd.segs[para] {
                            if let Some(LineBoundary { a, b }) = fsd.segs[curr] {
                                rsd.segs[curr] = LineBoundary::union(rsd.segs[curr], LineBoundary::new(a.max(a_), b));
                            }
                        }
                    } 
                    if let Some(prev) = opt_prev { 
                        if let Some(LineBoundary { a: _a, b: b_ }) = rsd.segs[prev] {
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

    pub fn is_mostright_boundary_nonempty(&self) -> bool {
        self.segs.slice(s![0, self.n-1, ..]).iter().any(|b| b.is_some())
    }

    pub fn is_mostleft_boundary_nonempty(&self) -> bool {
        self.segs.slice(s![0, 0, ..]).iter().any(|b| b.is_some())
    }

    pub fn is_mosttop_boundary_nonempty(&self) -> bool {
        self.segs.slice(s![1, self.m-1, ..]).iter().any(|b| b.is_some())
    }

    pub fn is_mostbottom_boundary_nonempty(&self) -> bool {
        self.segs.slice(s![1, 0, ..]).iter().any(|b| b.is_some())
    }

    pub fn bottom(&self) -> ArrayBase<ndarray::ViewRepr<&Option<LineBoundary>>, Dim<[usize; 1]>> {
        self.segs.slice(s![1, 0, ..])
    }

    /// Compute steps to walk along curves for partial matching solution.
    /// 
    /// Note: Should be appied to a reachability-space diagram.
    pub fn pcm_steps(&self) -> Option<Vec<(f64,f64)>> {

        let rsd = if self.is_rsd { self } else { &self.to_rsd() };
        let n = rsd.n;
        // let m = rsd.m;

        let mut steps = vec![];
        let mut curr = (0, 0, 0, 0.);

        // Seek lowest non-empty boundary on right side of the RSD.
        // (Basically performs PCM existence check as well.)
        let mut found = false;
        for (_j, lb) in rsd.segs.slice(s![0,n-1,..]).iter().enumerate() {
            if let Some(LineBoundary { a, b: _ }) = lb {
                let x = n - 1;
                let y = _j;
                let off = *a;
                curr = (0, x, y, off);
                found = true;
                break;
            }
        }
        if !found { return None }

        // Final step is found.
        steps.push(position_to_ij(curr));

        // Walk backwards. (Walk greedily, it should not matter).
        while !(position_on_left_boundary(curr)) { // Walk while we're not at the start position of P.

            // println!("curr: {curr:?}");
            if rsd.segs[position_to_seg(curr)].is_none() {
                // Sanity check:
                panic!("Current segment while walking ({curr:?}) should not be empty.");
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
                        if let Some(LineBoundary { a: a_, b: _ }) = rsd.segs[para] {
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
                // Sanity check:
                panic!("Should find next step in backwards walk at {curr:?}.\n{rsd:?}");
            }
            // println!("next: {next:?}");
            curr = next.unwrap();
            steps.push(position_to_ij(curr));

        }

        steps.reverse();
        Some(steps)
    }

}

