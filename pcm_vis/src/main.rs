/// Author: jvtubergen
/// * This code has copied pyfrechet for computing the FSD.
// The choice to switch to Rust is an arbitrary personal choice:
/// * Building rust code is fun.
/// * Practice with linking Rust code to Python (thus building python packages using rust code).
/// * Potentially upgrading to parallel implementation.
/// 
/// Use of code repo:
/// This code solves the PCMP (Partial Curve Matching Problem).
/// PCMP: Given a curve P, a curve Q and a distance threshold epsilon, is there some subcurve Q' of Q in such that the Fréchet distance between P and Q' is below epsilon? 
/// We limit ourselves to curves P and Q that are polygonal chains, and use the continuous strong Fréchet distance as a (sub)curve similarity/distance measure.
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
use std::{fs, io::Read, iter::zip, path::Path};
extern crate rand;
use pcm::prelude::*;
use rand::Rng;

use serde_derive::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use bincode;

use plotters::prelude::*;
use full_palette::{GREEN_400, RED_300};


// =============
// === Curve ===
// =============

/// Construct curve with n number of random points in f64 domain.
/// Chance of generating points which break general position is sufficiently small to ignore testing.
fn random_curve(n: usize, fieldsize: f64) -> Curve {
    let mut rng = rand::thread_rng();
    let c: Curve = (0..n).into_iter().map(|_| Vector {x: rng.gen_range(0.0..fieldsize), y: rng.gen_range(0.0..fieldsize)}).collect();
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
    let height = 20*m as i32;
    // println!("reachable:");
    for seg in reachable_segments {
        // println!("{seg:?}");
        let seg: Vec<(i32, i32)> = seg.into_iter().map(|(axis, x, y)| {
            // ((20.*x) as i32, (20.*y) as i32)
            if axis == 0 { ((20.*x) as i32, height - (20.*y) as i32) }
            else         { ((20.*y) as i32, height - (20.*x) as i32) }
        }).collect();
        drawing_area.draw(&Polygon::new(seg, reachable))?;
    }

    // println!("unreachable:");
    for seg in unreachable_segments {
        // println!("{seg:?}");
        let seg: Vec<(i32, i32)> = seg.into_iter().map(|(axis, x, y)| {
            // ((20.*x) as i32, (20.*y) as i32)
            if axis == 0 { ((20.*x) as i32, height - (20.*y) as i32) }
            else         { ((20.*y) as i32, height - (20.*x) as i32) }
        }).collect();
        drawing_area.draw(&Polygon::new(seg, unreachable))?;
    }


    // let mut reachable_corners = vec![];
    // let mut unreachable_corners = vec![];

    // Reachable corner points.
    // for i in 0..n {
    //     for j in 0..m {
    //         let coord = (20*i as i32, 20*j as i32);
    //         if fsd.corners[(i,j)] {
    //             reachable_corners.push(coord);
    //         } else {
    //             unreachable_corners.push(coord);
    //         }
    //     }
    // }
    // Draw reachable and unreachable cornerpoints.
    // for coord in reachable_corners {
    //     drawing_area.draw(&Circle::new(coord, 1, reachable))?;
    // }
    // for coord in unreachable_corners {
    //     drawing_area.draw(&Circle::new(coord, 1, unreachable))?;
    // }

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
            for axis in 0..2 {
                let (w,h) = fsd.dims[axis];
                let (x,y) = [(i,j), (j,i)][axis];
                let opt_curr = if y < h { Some((axis, x, y)) } else { None };
                let opt_prev = if y > 0 { Some((axis, x, y-1)) } else { None };
                let has_corner = fsd.corners[(i, j)];

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

        let d = p.distance(q);
        if !(d < eps + EPS) {
             return Err(format!("Distance {d} at step ({i}+{i_off}, {j}+{j_off}) should be below threshold {eps}+{EPS}."));
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

    let rsd = fsd.to_rsd();
    draw_fsd(&rsd, "rsd");
    let partial = rsd.check_pcm();
    println!("Is there a partial curve match?: {partial:?}.");

    let opt_steps = rsd.pcm_steps()?;
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
const RUN_COUNT: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let cases = 
    if DISCOVER {
        (0..RUN_COUNT).map(|_| {
            let ps = random_curve(5, 2.);
            // let c2 = translate_curve(ps, Vector{ x: 3. , y: 1. });
            // let qs = perturb_curve(ps.clone(), 1.);
            // let qs = random_curve(3, 2.);
            let qs = ps.clone();
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