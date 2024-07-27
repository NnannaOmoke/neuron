use crate::base_array::BaseDataset;
use crate::knn::Distance;
use crate::utils::math::argmax_1d;
use crate::utils::math::argmin_1d_f64;
use crate::utils::scaler::{Scaler, ScalerState};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use petal_neighbors::distance::Metric;
use rand::{thread_rng, Rng};
use std::iter::zip;

pub struct KMeans {
    k: usize,
    train: Array2<f64>,
    scaler: ScalerState,
    centroids: Array2<f64>,
    max_iters: usize,
}

impl KMeans {
    pub fn new(k: usize, max_iters: usize) -> Self {
        Self {
            k,
            train: Array2::default((0, 0)),
            scaler: ScalerState::default(),
            centroids: Array2::default((0, 0)),
            max_iters,
        }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn fit(&mut self, dataset: &BaseDataset) {
        self.train = dataset.into_f64_array();
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(self.train.view());
        scaler.transform(&mut self.train.view_mut());
        self.centroids = Self::lloyds_algorithm(self.train.view(), self.k, self.max_iters);
    }

    fn lloyds_algorithm(features: ArrayView2<f64>, k: usize, max_iters: usize) -> Array2<f64> {
        let init_centroids = Self::kmeans_plus_plus(features, k);
        let mut classes = Array1::from_elem(features.nrows(), 0);
        let mut centroids = Array2::from_elem((features.nrows(), k), 0f64);
        //initialize the centroids with data from k++
        zip(init_centroids.view(), centroids.rows_mut().into_iter()).for_each(
            |(&row_index, mut row_mut)| {
                row_mut.assign(&features.row(row_index));
            },
        );
        let mut distances = Array2::from_elem((k, features.nrows()), 0f64);
        for _ in 0..max_iters {
            //distance update
            distances
                .rows_mut()
                .into_iter()
                .enumerate()
                .for_each(|(index, mut row)| {
                    //for each empty row, we can calculate the distance from a centroid, assign it to a value
                    let distance = Array1::from_shape_fn(k, |kindex| {
                        Distance::Euclidean.distance(&features.row(index), &centroids.row(kindex))
                    });
                    row.assign(&distance);
                });
            //class update
            classes.iter_mut().enumerate().for_each(|(index, class)| {
                let index = argmin_1d_f64(distances.row(index));
                *class = index;
            });
            //centroid update
            centroids
                .rows_mut()
                .into_iter()
                .enumerate()
                .for_each(|(index, mut row)| {
                    let members = classes
                        .iter()
                        .filter(|&&x| x == index)
                        .map(|x| *x)
                        .collect::<Array1<usize>>();
                    let size = members.len();
                    let mut averages = Array1::from_elem(features.nrows(), 0f64);
                    members.iter().enumerate().for_each(|(index, &target)| {
                        averages[index] += features.row(target)[index];
                    });
                    averages /= size as f64;
                    row.assign(&averages);
                });
        }
        centroids
    }

    fn kmeans_plus_plus(features: ArrayView2<f64>, k: usize) -> Array1<usize> {
        let mut rng = thread_rng();
        let choice = rng.gen_range(0..features.nrows());
        let mut res = Array1::from_elem(k, 0);
        res[0] = choice;
        for centroid in 1..k {
            let mut distances = Vec::new();
            for point in 0..features.nrows() {
                let row_p = features.row(point);
                let mut distance = usize::MAX as f64;
                for cent in 0..centroid {
                    let curr_distance = Distance::Euclidean.distance(&row_p, &features.row(cent));
                    distance = f64::min(distance, curr_distance);
                }
                distances.push(distance);
            }
            let distances = Array1::from_vec(distances);
            let next = argmax_1d(distances.view());
            res[centroid] = next;
        }
        res
    }
}
