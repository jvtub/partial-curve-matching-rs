use pyo3::pyclass;
use pyo3::pyfunction;

use crate::curve::Curve;
use crate::lineboundary::print_lb;
use crate::sanity_check;
use crate::{prelude::LineBoundary, vector::Vector};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BTreeMap as Map;
use std::collections::BTreeSet as Set;
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::iter::zip;


type Opt<T> = Option<T>;

/// Node identifier.
/// 
type NID = usize;

/// Edge identifier.
type EID = usize;

/// A pair of nodes, thus an edge.
type NIDPair = (NID, NID);

/// A sequence of nodes to walk along the graph.
type Path = Vec<NID>;

/// Curve on horizontal axis, vertical axis is a single point namely a graph node.
type FreeSpaceLine = Vec<Opt<LineBoundary>>;

/// Free-Space Diagram row.
/// Linear approach works on rows of Free-space diagrams, curve on horizontal axis, edge on vertical axis.
type FSDrow = Vec<[[std::option::Option<LineBoundary>; 2]; 2]>; // (x, y, [horizontal, vertical])

/// Print FSDrow (for debugging purposes).
pub fn print_fsdrow(fsd: &FSDrow) {
    let n = fsd.len();
    let m = fsd[0].len();
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
                print_lb(fsd[x][y][a]);
                print!("             "); // 13
            }
            println!();
        }
    }
}

/// Specific for linear implementation.
type ShortcutPointer = Vec<[Opt<usize>; 2]>;

/// Specific for linear implementation.
type ReachabilityPointer = Vec<usize>;

/// Ordered sweep line events to process while traversing the reachable points in the Free-Space Surface.
type EventBucket = BinaryHeap<Reverse<PathPointer>>;

/// Pointer living at FDu within the event bucket i.
/// When sweeping we send 
#[derive(Debug, Clone)]
struct PathPointer {
    /// This is the offset which could be endorsed by the bottom interval if it propagates to the same cell.
    c: f64,
    /// The walk started from some vertex and some curve interval (we don't care from what curve interval, we only care for reconstructing path).
    from: Vec<NID>,
    /// The current index we walk to.
    curr: NID,
}
impl PartialEq  for PathPointer {
    fn eq(&self, other: &Self) -> bool {
        self.c == other.c
    }
}
impl Eq         for PathPointer {

}
impl PartialOrd for PathPointer {
    fn ge(&self, other: &Self) -> bool {
        self.c >= other.c
    }
    fn gt(&self, other: &Self) -> bool {
        self.c > other.c
    }
    fn le(&self, other: &Self) -> bool {
        self.c <= other.c
    }
    fn lt(&self, other: &Self) -> bool {
        self.c < other.c
    }
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.c < other.c {
            Some(Ordering::Less)
        } else if self.c > other.c {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}
impl Ord        for PathPointer {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.c < other.c {
            Ordering::Less
        } else if self.c > other.c {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}


/// LinearGraph structure which stores relevant information a simple undirected graph.
/// Essentially a graph consists of a collection of nodes and edges.
/// For simplicity we assume the graph to be vectorized, which means that edges have no curvature (beyond a straight line segment between the vertices the edge connects).
#[pyclass]
pub struct LinearGraph {
    /// Adjacent nodes.
    adj: Map<NID, Set<NID>> ,

    /// List NIDs in graph.
    /// 
    /// Note: Since fixed position in vector this functions as a index as well.
    nid_list: Vec<NID>,
    /// Node vectors.
    vec_list: Vec<Vector>,
    nid_map: Map<NID, NID>,

    /// Map edge to unique identifier.
    /// 
    /// Note: Since fixed position in vector this functions as a index as well.
    eid_list: Vec<NIDPair>,
    /// Back-link for EIDs.
    eid_map: Map<NIDPair, EID>,
}

impl LinearGraph {

    /// Map transformed node indices back to their original index.
    pub fn map_path(&self, path: Path) -> Path {
        path.iter().map(|&u| self.nid_list[u]).collect()
    }


    /// Store a graph struct, re-used as we apply to different curves.
    pub fn new(vertices: Vec<(NID, Vector)>, edges: Vec<(NID, NID)>) -> Self {

        // Construct node information.
        let mut nid_list = Vec::new();
        let mut vec_list = Vec::new();
        let mut nid_map = Map::new();
        for (i, (j, v)) in vertices.iter().enumerate() {
            nid_list.push(j.clone());
            vec_list.push(v.clone());
            nid_map.insert(j.clone(), i);
        }
        
        // Sanity check on NID backlink.
        assert_eq!(nid_list.len(), nid_map.len());
        for i in 0..nid_list.len() { 
            let nid = nid_list[i];
            assert_eq!(i, *nid_map.get(&nid).unwrap());
        }
        for nid in &nid_list { 
            assert_eq!(*nid, nid_list[*nid_map.get(nid).unwrap()]);
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

            // Act on mapped node ids.
            let u = *nid_map.get(&u).unwrap();
            let v = *nid_map.get(&v).unwrap();

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
        for eid in &eid_list {
            assert_eq!(*eid, eid_list[*eid_map.get(eid).unwrap()]);
        }

        LinearGraph { nid_list, nid_map, vec_list, adj, eid_list, eid_map }
    }
}


/// Make a graph out of a list of vertices and a list of edges.
#[pyfunction]
pub fn make_linear_graph(vertices: Vec<(NID, Vector)>, edges: Vec<(NID, NID)>) -> LinearGraph {
    LinearGraph::new(vertices, edges)
}


/// Convert a Free-Space Diagram boundary line into a Shortcut Pointer list.
fn construct_shortcut_pointers(FDu: &FreeSpaceLine) -> ShortcutPointer {
    let mut i = 0;
    let mut k = 0;
    let mut SPu = vec![];
    let n = FDu.len();
    for j in 0..n {
        // Walk k to next non-empty.
        if j == k { 
            k += 1; // Step to the right.
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
        SPu.push([opt_left, opt_right]);
    }
    SPu
}

/// Computes RPuv0, RPuvk.
fn construct_reachability_pointers(FDuv: &FSDrow) -> (usize, ReachabilityPointer) {
    // FSDrow = (x, y, [horizontal, vertical])
    let mut RPuv  = vec![];
    let mut RPuv0 = None;
    let n = FDuv.len();
    assert_eq!(FDuv[0].len(), 2);
    let mut S: VecDeque<(usize, f64)> = VecDeque::new();
    // Initiate with the left row boundary on the stack. Note how the stack has the potential to be empty.
    if let Some(LineBoundary { a: a_0, b: b_0 }) = FDuv[0][0][1] {
        S.push_front((0, a_0));
    } else {
        RPuv0 = Some(0);
    }

    // Perform iteration.
    for j in 1..n {
        let i = j - 1;
        let k = RPuv.len();

        // Some sanity checks to perform on each iteration.
        S.make_contiguous();
        if S.len() > 0 {  
            // Note how we only have to check invariants if the stack is non-empty.
            // Invariants: (given x < y)
            // * jx < jy
            // * a_jx > a_jy
            let v = S.as_slices();
            let v1 = v.0;
            let v2 = &v.0[1..];
            for ((jx, a_jx), (jy, a_jy)) in zip(v1, v2) {
                assert!(jx < jy);
                assert!(a_jx > a_jy);
            }
        } else {
            // Expect the reachable pointer count (thus k) is at i.
            assert_eq!(i, k);
        }

        // Check cutoff occurs, obtain related index, add new intervals reachabilities, pop back processed stack.
        let mut cutoff = None;
        let mut exceed = None;
        let opt_lb = FDuv[j][0][1];
        if opt_lb.is_none() { // We always cut off and exceed.
            cutoff = Some(j);
            exceed = Some(j);
        } else {
            let LineBoundary { a: a_j, b: b_j } = &opt_lb.unwrap();
            // Try to cut off: Seek highest jx for which a_jx > b_j.
            for (jx, a_jx) in &S {
                if a_jx > b_j {
                    cutoff = Some(*jx);
                }
            }
            // Try to exceed: Seek lowest jx for which a_jx <= a_j.
            for (jx, a_jx) in &S {
                if a_jx <= a_j {
                    exceed = Some(*jx);
                }
            }
        }

        if let Some(jx) = cutoff {
            // Check for left row boundary.
            if RPuv0.is_none() {
                RPuv0 = Some(i);
            }
            // Drop back of stack including jx.
            while S.len() > 0 && S.back().unwrap().0 <= jx {
                S.pop_back();
            }
            // Set intervals k up to j_x to i.
            while RPuv.len() < jx {
                RPuv.push(i);
            }
        }
        if let Some(jx) = exceed {
            // Drop front of stack including jx.
            while S.len() > 0 && S.front().unwrap().0 >= jx {
                S.pop_front();
            }
        }
        if let Some(LineBoundary { a: a_j, b: b_j }) = opt_lb {
            S.push_front((j, a_j));
        }
    }

    // Most-right cell boundary is reachable by current interval k.
    let k = RPuv.len();

    // Push remaining intervals to reach final.
    while RPuv.len() < n - 1 {
        RPuv.push(n - 1); // Indicates that we reach reachable space on right row boundary (thus that from that bottom interval we will have a partial match).
    }

    (RPuv0.unwrap_or(n), RPuv)
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

/// Extract unique elements out of a vector and store as a Set.
fn unique<T>(elements: Vec<T>) -> Set<T> where T: Ord {
    let mut set = Set::new();
    for element in elements {
        set.insert(element);
    }
    set
}

/// Linear implementation of partial Curve Matching of a curve against a graph. Partial match a curve against a graph with epsilon threshold.
/// 
/// (todo optimization: Lazily evaluate FDuv's, initiate by checking solely the left boundary of every FDuv.)
/// 
/// Steps consist of:
/// * Constructing Free-Space lines.
/// * Constructing Free-Space rows.
/// * Constructing Shortcut Pointers.
/// * Constructing Reachability Pointers.
/// * Creating initial events.
/// * Sweeping.
#[pyfunction]
pub fn partial_curve_graph_linear(graph: &LinearGraph, curve: Curve, eps: f64) -> Option<Vec<NID>> {

    let n = curve.len() - 1; // Number of intervals.

    // Free-Space Lines. (Per node do we have a one-dimensional line.)
    // println!("Computing Free-Space Lines.");
    let mut FDus: Vec<FreeSpaceLine> = vec![];
    for u in 0..graph.nid_list.len() {
        let mut FDu = vec![];
        for i in 0..n {
            let p = graph.vec_list[u];
            let q0 = curve[i];
            let q1 = curve[i+1];
            let lb = LineBoundary::compute(p, q0, q1, eps);
            FDu.push(lb);
        }
        FDus.push(FDu);
    }
    if sanity_check {
        for FDu in &FDus {
            // Expect length to match row length.
            assert_eq!(FDu.len(), n);
            // If we have at index i a value of b == 1., expect to have a == 0. at index i + 1;
            for i in 0..n {
                if let Some(LineBoundary { a, b }) = FDu[i] && b == 1. && i < n - 1{
                    assert_eq!(FDu[i + 1].unwrap().a, 0.);
                }
            }
            // Similarly for the reverse.
            for i in (0..n).rev() {
                if let Some(LineBoundary { a, b }) = FDu[i] && b == 0. && i > 0 {
                assert_eq!(FDu[i - 1].unwrap().b, 1.);
                }
            }
        }
    }

    // Free-Space Rows. (Per node pair we have a row of free-space.)
    // println!("Computing Free-Space Rows");
    let mut FDuvs: Vec<FSDrow> = vec![];
    for &(u, v) in &graph.eid_list {
        let q0 = graph.vec_list[u];
        let q1 = graph.vec_list[v];
        let FDu = &FDus[u];
        let FDv = &FDus[v];
        let mut FDuv: FSDrow = vec![];
        let n = curve.len();
        for x in 0..n {
            let mut col = vec![];
            for y in 0..2 {
                let horizontal = if x < n - 1 { [FDu, FDv][y][x] } else { None };
                let vertical = if y == 0 { LineBoundary::compute(curve[x], q0, q1, eps) } else { None };
                col.push([horizontal, vertical]);
            }
            FDuv.push([col[0], col[1]]);
        }
        FDuvs.push(FDuv);
    }
    if sanity_check {
        for FDuv in &FDuvs {
            let n = FDuv.len();
            assert!(FDuv[n-1][0][0].is_none());
            assert!(FDuv[n-1][1][0].is_none());
            for i in 0..n {
                assert!(FDuv[i][1][1].is_none());
            }
        }
    }

    // Shortcut Pointers. (Per node (per FDu) we have backward and forward shortcut pointers.)
    // println!("Computing Shortcut Pointers.");
    let mut SPus: Vec<ShortcutPointer> = vec![];
    for u in 0..graph.nid_list.len() {
        let FDu = &FDus[u];
        let SPu = construct_shortcut_pointers(FDu);
        SPus.push(SPu);
    }
    if sanity_check {
        for u in 0..graph.nid_list.len() {
            let SPu = &SPus[u];
            let FDu = &FDus[u];
            // * Expect each reference to point to actual existing point of FDu.
            for [opt_left, opt_right] in SPu {
                if let Some(i) = opt_left {
                    assert!(FDu[*i].is_some());
                }
                if let Some(l) = opt_right {
                    assert!(FDu[*l].is_some());
                }
            }
            // * Expect not to point at itself.
            for j in 0..SPu.len() {
                if let Some(i) = SPu[j][0] {
                    assert!(i < j);
                }
                if let Some(k) = SPu[j][1] {
                    assert!(j < k);
                }
            }
            // * Expect all non-empty intervals on FDu be referenced at least once (unless n == 1)
            let mut non_empty: Vec<usize> = FDu.iter().enumerate().filter(|(_, opt_lb)| opt_lb.is_some()).map(|(i, _)| i).collect();
            let mut non_empty = unique(non_empty);
            if n > 1 {
                for [opt_left, opt_right] in SPu {
                    if let Some(i) = opt_left {
                        non_empty.remove(i);
                    }
                    if let Some(l) = opt_right {
                        non_empty.remove(l);
                    }
                }
                assert_eq!(non_empty.len(), 0);
            }
        }
    }

    // Reachability Pointers. 
    // 
    // Note the existence of RPuv0s:
    //   The problem with our reachability pointers is the assumption we start at the bottom of the cell.
    //   If we start on the left boundary, we may not walk that far to the right.
    //
    // RPuv0: Most-right reachable row segment starting at a non-empty left row boundary.
    // println!("Computing Reachability Pointers.");
    let mut RPuvs = vec![];
    let mut RPuv0s = vec![]; // What interval can we reach when starting at the (free space on the) left cell boundary.
    for eid in 0..graph.eid_list.len() {
        let FDuv = &FDuvs[eid];
        let (RPuv0, RPuv) = construct_reachability_pointers(FDuv);
        RPuv0s.push(RPuv0);
        RPuvs.push(RPuv);
    }
    if sanity_check {

        // RPuv0s: Walk from 0 to k and expect within boundary.
        for eid in 0..graph.eid_list.len() {
            let RPuv0 = RPuv0s[eid];
            let FDuv = &FDuvs[eid];
            let k = RPuv0; // Interval reachable on FDv.
            let n = FDuv.len();
            assert!(k <= n);
            if FDuv[0][0][1].is_none() {
                assert_eq!(RPuv0, 0);
            } else {
                let a_0 = FDuv[0][0][1].unwrap().a; // Get boundary.
                // print_fsdrow(&FDuv);
                // println!("Reachable interval from left row boundary is {k:?}.");
                for j in 1..k { // Walk all the way.
                    let b_j = FDuv[j][0][1].unwrap().b;
                    assert!(a_0 <= b_j);
                }
            }
        }

        // RPuvs: 
        for eid in 0..graph.eid_list.len() {
            let RPuv = &RPuvs[eid];
            let FDuv = &FDuvs[eid];
            let n = FDuv.len();
            assert_eq!(RPuv.len(), n - 1);
            // print_fsdrow(&FDuv);
            // println!("{RPuv:?}");
            for i in 0..n - 1 {
                let k = RPuv[i];
                if i == k { // Expect empty boundary on the right side.
                    assert!(FDuv[i+1][0][1].is_none());
                } else { // Expect boundaries larger than 
                    assert!(k > i);
                    let a = FDuv[i+1][0][1].unwrap().a; // This boundary must be non-zero (otherwise we cannot move to cell on the right).
                    for l in i+1..k {
                        let LineBoundary { a: a_l, b: b_l } = FDuv[l][0][1].unwrap(); // This boundary must be non-zero (otherwise we cannot move to cell on the right).
                        assert!(b_l >= a); // Expect not to be cut off as we walk to the right.
                    }
                }
            }
        }
    }
    
    // Sweep line.
    // Initialize event buckets.
    // println!("Initializing event buckets.");
    let mut event_buckets = vec![];
    for i in 0..n {
        // Note how we use a reversed since we want to pop lowest a's first (we sweep from left to right).
        let bucket: BinaryHeap<Reverse<PathPointer>> = BinaryHeap::new();
        event_buckets.push(bucket);
    }
    // Initiate with non-empty left boundaries.
    let tmp = graph.eid_list.len();
    // println!("Number of edges: {tmp}");
    for eid in 0..graph.eid_list.len() {
        let uv = &graph.eid_list[eid];
        let _eid = *graph.eid_map.get(uv).unwrap();
        assert_eq!(eid, _eid);
        let &(u, v) = uv;
        if FDuvs[eid][0][0][1].is_some() { // If left row boundary exists.
            let k = RPuv0s[eid];
            if k >= n {
                let mut path = vec![u, v];
                // Map path back to original vectors.
                // println!("Found direct path.");
                return Some(graph.map_path(path));
            }
            let mut l = 0;
            while l <= k {
                // println!("l: {l:?}, k: {k:?}, n: {n:?}");
                let FDv = &FDus[v];
                if let Some(LineBoundary { a: c, b: d }) = FDv[l] {
                    // Push entire interval.
                    let mut bucket = event_buckets.get_mut(l).unwrap();
                    bucket.push(Reverse(PathPointer { c, from: vec![u], curr: v }));
                }
                l = SPus[v][l][1].unwrap_or(n);
            }
        }
    }

    // Start sweeping!
    // println!("Sweeping.");
    let mut i = 0;
    let mut x = 0.; // for sanity checking
    let mut evicted = Set::new(); // Tracking evicted NIDs to prevent pushing back nodes onto the current event bucket which have already been processed for the current interval.
    while i < n { // Loop is called after every FDuv interval processed or an empty event buffer occurring.
        // println!("Currently at event bucket {i:?}."); // Current column we are at (we sweep to the right).

        // Obtain next event for our sweep line.
        let opt_pathpointer = {
            let current_bucket = event_buckets.get_mut(i).unwrap();
            // println!("Current bucket: {current_bucket:?}.");
            if current_bucket.is_empty() { // We have to check next bucket.
                i += 1; // Continue to next curve interval.
                evicted = Set::new(); // Thrash evicted nodes for new event bucket.
                None
            } else {
                Some(current_bucket.pop().unwrap().0)
            }
        };

        if opt_pathpointer.is_none() {
            continue; // Restart iteration at next event bucket.
        }

        // Take out a lowest value.
        let pathpointer = opt_pathpointer.unwrap();
        // println!("Current pathpointer: {pathpointer:?}.");
        let PathPointer { c, from, curr } = pathpointer;

        // Sanity check we are sweeping to the right.
        assert!(x <= i as f64 + c);
        x = i as f64 + c;

        // Try to walk to adjacents:                   left reachable    pointer to iter     right reachable                 
        //                                               index top      from left to right     index top                         
        //                                                   j                 l                   k                               
        //    .------------------.- - - - - - - - - .------------------.- - - - - - - - - .------------------.
        //    |                  |                  |                  |                  |                  |
        //    |                  |                  |                  |                  |                  |
        //    |                  |                  |                  |                  |                  |
        //    .------------------.- - - - - - - - - .------------------.- - - - - - - - - .------------------.
        //              i
        //      curr index bottom
        // 
        let u = curr;
        if evicted.contains(&u) { // We already processed u, skip this iteration.
            // println!("Skipping evicted node {u}.");
            continue;
        };
        evicted.insert(u); // Don't check this node again for this event bucket.
        for v in graph.adj.get(&u).unwrap().clone() {
            // println!("Processing edge ({u},{v}).");
            let eid = graph.eid_map.get(&(u, v)).unwrap().clone();
            let k = RPuvs[eid][i];
            if k == n { // See whether we can reach the end.
                let mut path = from;
                path.push(u);
                path.push(v);
                // println!("Found reachable to end {path:?}.");
                return Some(graph.map_path(path)); // Map path back to original vectors.
            }
            // Iterate from left to right on reachable elements and push to respective event bucket.
            let mut l = i;
            while l <= k { // Note that k < n.
                let tmp = FDus[v][l];
                // println!("Checking FDuv[{v}][{l}]: {tmp:?}");
                if let Some(LineBoundary { a: c_n, b }) = FDus[v][l] { // Check necessary for first element.
                    // Add to event bucket.
                    let bucket = event_buckets.get_mut(l).unwrap();
                    let extracted = extract_bucket_nid(bucket, v);
                    if let Some(path_pointer) = extracted && path_pointer.c <= c_n {
                         // Ignore our just created event (reinsert old one).
                        // println!("Event already present {path_pointer:?}.");
                        bucket.push(Reverse(path_pointer));
                    } else { // Add our event (overwrite previously stored), because a is lower.
                        let mut path = from.clone();
                        path.push(u);
                        let pathpointer = PathPointer { c: c_n.max(c), from: path, curr: v };
                        // println!("Add new event {pathpointer:?}.");
                        bucket.push(Reverse(pathpointer));
                    }
                }
                l = SPus[v][l][1].unwrap_or(n);
            }
            // println!("End of this iteration.")
        }
    }

    None
}

