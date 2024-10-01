use pcm::prelude::*;

fn main() {
    println!("Running test.");

    // test 1 (basic)
    let p1 = Vector::new(0., 0.); 
    let p2 = Vector::new(0.5, 0.5);
    let p3 = Vector::new(1., 1.); 
    let ps = vec![p1, p3];
    let vertices = vec![(0, p1), (1, p2), (2, p3)];
    let edges = vec![(0,1), (1,2)];
    println!("Constructing graph.");
    let graph = LinearGraph::new(vertices, edges);
    println!("Computing partial curve on graph.");
    let result = partial_curve_graph_linear(&graph, ps, 1.01).unwrap();
    assert_eq!(result, vec![0, 1, 2]);


    // test 2 (node id transformation)
    let p1 = Vector::new(0., 0.); 
    let p2 = Vector::new(0.5, 0.5);
    let p3 = Vector::new(1., 1.); 
    let ps = vec![p1, p3];
    let vertices = vec![(4, p1), (9, p2), (2, p3)];
    let edges = vec![(4,9), (9,2)];
    println!("Constructing graph.");
    let graph = LinearGraph::new(vertices, edges);
    println!("Computing partial curve on graph.");
    let result = partial_curve_graph_linear(&graph, ps, 1.01).unwrap();
    assert_eq!(result, vec![4, 9, 2]);

    // test 3 (more nodes)
    println!("test 3:");
    let vertices: Vec<(usize, Vector)> = (0..10).map(|i| (i, Vector::new(i as f64 * 0.1, (i+1) as f64 * 0.1))).collect();
    let ps: Vec<Vector> = vec![vertices.first().unwrap().1, vertices.last().unwrap().1];
    let edges = (0..9).map(|i| (i, i+1)).collect();
    let graph = LinearGraph::new(vertices, edges);
    let result = partial_curve_graph_linear(&graph, ps, 1.01).unwrap();
    assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}