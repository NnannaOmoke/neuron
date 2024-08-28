use super::*;
use crate::*;
use kdtree::KdTree;
use ndarray::{array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use num_traits::{Float, Zero};
use petal_neighbors::{
    distance::{Euclidean, Metric},
    BallTree,
};

use std::sync::Arc;
use std::sync::RwLock;
mod knnclassifier;
mod knnregressor;

#[derive(Default, Debug, Clone)]
pub enum VotingChoice {
    #[default]
    Uniform,
    Distance,
    //the labels and the actual distances from the point
    Custom(fn(ArrayView1<f64>, ArrayView1<f64>) -> f64),
}
#[derive(Copy, Clone, Default, Debug)]
pub enum Distance {
    #[default]
    Euclidean,
    Manhattan,
}

#[derive(Clone)]
pub(crate) struct BallTreeKNN<'a, M: Metric<f64>> {
    tree: Arc<RwLock<BallTree<'a, f64, M>>>,
}

pub(crate) struct KdTreeKNN {}

impl<M: Metric<f64>> BallTreeKNN<'_, M> {
    fn new(points: ArrayView2<f64>, metric: M) -> Self
    where
        M: Metric<f64>,
    {
        let tree = BallTree::new(points.to_owned(), metric).unwrap();
        let tree = Arc::new(RwLock::new(tree));
        Self { tree }
    }

    fn query(&self, points: ArrayView2<f64>, n: usize) -> (Array2<usize>, Array2<f64>) {
        let mut closest = Array2::from_elem((points.nrows(), n), 0);
        let mut distances = Array2::from_elem((points.nrows(), n), 0f64);
        let tree = self.tree.read().unwrap();
        points
            .rows()
            .into_iter()
            .enumerate()
            .for_each(|(index, row)| {
                let results = tree.query(&row, n);
                let closest_indices = Array1::from_vec(results.0);
                let distance_results = Array1::from_vec(results.1);
                closest.row_mut(index).assign(&closest_indices);
                distances.row_mut(index).assign(&distance_results);
            });
        (closest, distances)
    }
}

impl<F: Float + AddAssign> Metric<F> for Distance {
    fn distance(&self, input_one: &ArrayView1<F>, input_two: &ArrayView1<F>) -> F {
        match self {
            Distance::Euclidean => Euclidean {}.distance(input_one, input_two),
            Distance::Manhattan => zip(input_one, input_two)
                .map(|(x, y)| (*x - *y).abs())
                .reduce(|x, y| x + y)
                .unwrap(),
        }
    }

    fn rdistance(&self, input_one: &ArrayView1<F>, input_two: &ArrayView1<F>) -> F {
        match self {
            Distance::Euclidean => Euclidean {}.rdistance(input_one, input_two),
            Distance::Manhattan => self.distance(input_one, input_two),
        }
    }

    fn rdistance_to_distance(&self, d: F) -> F {
        match self {
            Distance::Euclidean => Euclidean {}.rdistance_to_distance(d),
            Distance::Manhattan => d,
        }
    }

    fn distance_to_rdistance(&self, d: F) -> F {
        match self {
            Distance::Euclidean => Euclidean {}.distance_to_rdistance(d),
            Distance::Manhattan => d,
        }
    }
}
