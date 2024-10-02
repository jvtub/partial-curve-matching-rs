use pyo3::pyclass;
use pyo3::pyfunction;
use std::collections::BTreeMap as Map;
use std::collections::BTreeSet as Set;
use std::iter::zip;

use crate::Curve;
use crate::LineBoundary;
use crate::NIDPair;
use crate::Path;
use crate::NID;
use crate::{OptLineBoundary, Vector};

const DEBUG: bool = false;

/// Alternative, simplified datastructure for Free-Space Diagram.
/// 
/// This one is easier to extend in comparison to ArrayView which is not intended to append rows/columns.
type FSD2 = Vec<Vec<Vec<OptLineBoundary>>>; // (x, y, horizontal|vertical)

/// Graph structure which stores relevant information a simple undirected graph.
#[pyclass]
pub struct ExponentialGraph {
    /// Nodes with (x,y) coordinate.
    nodes: Map<NID, Vector>,
    /// Adjacent nodes.
    adj: Map<NID, Set<NID>> 
}
impl ExponentialGraph {

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
        ExponentialGraph { nodes, adj }
    }


    /// Extract the curvature of an edge (which is just a single line segment for this vectorized graph).
    pub fn curvature(&self, uv: NIDPair) -> Curve {
        let (u, v) = uv;
        vec![self.nodes[&u], self.nodes[&v]]
    }

}

/// Compute FSD out of two curves (first curve on horizontal axis, second curve on vertical axis).
fn compute_fsd(ps: Curve, qs: Curve, eps: f64) -> FSD2 {

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
fn propagate_bottom_row(mut fsd: FSD2) -> FSD2{
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
fn propagate_cell_reachability(bottom: OptLineBoundary, left: OptLineBoundary, right: OptLineBoundary, top: OptLineBoundary) -> (OptLineBoundary, OptLineBoundary) {

    // Decide reachability of right cell boundary based on bottom and left cell boundary.
    let goal_right = if bottom.is_some() {
        right
    } else if right.is_none() || left.is_none() {
        None
    } else {
        let LineBoundary { a, b: _ } = left.unwrap();
        let LineBoundary { a: c, b: d } = right.unwrap();
        LineBoundary::new(a.max(c), d)
    };

    // Decide reachability of top cell boundary based on bottom and left cell boundary.
    let goal_top = if left.is_some() {
        top
    } else if top.is_none() || bottom.is_none() {
        None
    } else {
        let LineBoundary { a, b: _ } = bottom.unwrap();
        let LineBoundary { a: c, b: d } = top.unwrap();
        LineBoundary::new(a.max(c), d)
    };
    
    
    (goal_right, goal_top)
}


/// Convert an FSD into an RSD.
fn fsd_to_rsd(mut fsd: FSD2, propagate: bool) -> FSD2 {
    
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
fn lb_is_subset(lb1: OptLineBoundary, lb2: OptLineBoundary) -> bool {
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
fn extract_top(fsd: &FSD2) -> Vec<OptLineBoundary> {
    let mut top = vec![];
    let n = fsd_width(fsd);
    let m = fsd_height(fsd);
    for x in 0..n {
        top.push(fsd[m-1][x][0]);
    }
    return top;
}


/// Returns true if the top is empty.
fn is_top_empty(fsd: &FSD2) -> bool {
    !extract_top(fsd).into_iter().any(|lb| lb.is_some())
}

#[pyfunction]
pub fn make_exponential_graph(vertices: Vec<(NID, Vector)>, edges: Vec<(NID, NID)>) -> ExponentialGraph {
    ExponentialGraph::new(vertices, edges)
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
    let _offset = 11;
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


/// Match curve against a graph with epsilon threshold.
/// 
/// todo (optimization): Lazily evaluate FDij's, initiate by checking solely the left boundary of every FDij.
#[allow(non_snake_case)]
#[pyfunction]
pub fn partial_curve_graph_exponential(graph: &ExponentialGraph, ps: Curve, eps: f64) -> Option<Vec<NID>> {

    // Initiate FDij's and initial reachable paths.
    let mut FDijs : Map<NIDPair, FSD2> = Map::new();
    let mut queue : Vec<(Path, Vec<OptLineBoundary>)> = vec![];
    let n = ps.len();
    for (i, js) in graph.adj.clone() { // Iterate all edges.
        for j in js { // These are all the possible starting points we could have.
            let eid = (i, j);
            let fsd = compute_fsd(ps.clone(), graph.curvature(eid), eps);
            FDijs.insert(eid, fsd.clone());
            let rsd = fsd_to_rsd(fsd, true);
            if rsd[0][n-1][1].is_some() { // Non-empty right boundary found.
                return Some(vec![i, j]);
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
        let i = path.last().unwrap().clone();

        // Seek adjacent vertices which have not been visited yet.
        let adjacents = graph.adj.get(&i).unwrap().clone();
        let adjacents: Vec<NID> = adjacents.into_iter().filter(|&v| !path.contains(&v)).collect();

        // If we can reach next element, make walk (add new vertex to path).
        for j in adjacents.clone() {
            totalcount += 1;
            let mut path = path.clone();
            path.push(j); 
            let eid: (usize, usize) = (i, j);

            if DEBUG {
                println!("Checking path: {path:?} at edge {eid:?}.");
                // Print relevant edges.
                for (u, v) in zip(&path, &path[1..]) {
                    let _eid = (*u, *v);
                    let _fsd = FDijs.get(&_eid).unwrap();
                    println!("edge {_eid:?}:");
                    print_fsd(_fsd);
                    println!("");
                }
            }

            let mut fsd = FDijs.get(&eid).unwrap().clone();

            if DEBUG {
                println!("fsd initially:");
                print_fsd(&fsd);
                println!("");
            }

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

            if DEBUG {
                println!("fsd with bottom of rsd:");
                print_fsd(&fsd);
                println!("");
            }

            // Walk from left to right and update rsd.
            let rsd = fsd_to_rsd(fsd, false);

            if DEBUG {
                println!("rsd:");
                print_fsd(&rsd);
                println!("");
            }

            if rsd[0][n-1][1].is_some() { // Non-empty right boundary found.
                return Some(path);
            }

            if !is_top_empty(&rsd) {
                // If not empty we push it to the stack.
                queue.push((path, extract_top(&rsd)));
            }
        }
    }
    println!("Total paths checkted: {totalcount}");

    None
}


