use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use petal_neighbors::{distance::Euclidean, BallTree};

pub struct DBSCAN {
    internal: BallTree<'static, f64, Euclidean>,
    epsilon: f64,
    min_points: usize,
    core_points: Array2<f64>,
}

impl DBSCAN {}
