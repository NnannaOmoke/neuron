use core::num;

use ndarray::{s, Array1, Array2};
use num_traits::ToPrimitive;
use rand::{random, thread_rng, Rng};

use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
};

pub(crate) struct LinearRegressorBuilder {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    num_iters: usize,
}

impl LinearRegressorBuilder {
    fn new() -> Self {
        Self {
            weights: vec![],
            bias: 0f64,
            learning_rate: 0.00875f64,
            num_iters: 30000,
        }
    }

    fn all_colunms_dot(
        dataset: &BaseDataset,
        target_col: usize,
        dataset_index: usize,
        weights: &Vec<f64>,
    ) -> f64 {
        let num_cols = dataset.shape().1;
        let mut result = 0.0;
        let mut wt_index = 0usize;
        for i in 0..num_cols {
            if i == target_col {
                continue;
            }
            result += dataset.get(dataset_index, i).to_f64().unwrap() * weights[wt_index];
            wt_index += 1;
        }
        result
    }
    fn fit(&mut self, dataset: &BaseDataset, target_col: usize) {
        //if theres logic errors its probable in colunm_dot or because i mixed row and colunm somewhere
        let num_weights = dataset.shape().1 - 1;
        let n = dataset.shape().0;
        let n_inv = 1.0 / (n as f64);

        self.weights.resize(num_weights, 0.0);
        self.bias = 0.0;
        let mut dws: Vec<f64> = (0..num_weights).map(|_| random::<f64>()).collect();
        let mut db = 0.0;
        for _current_iter in 0..self.num_iters {
            for dataset_index in 0..n {
                //I couldn't figure out how to use ndarray's dot while skipping a colunm
                let error = self.bias
                    + Self::all_colunms_dot(dataset, target_col, dataset_index, &self.weights)
                    - dataset.get(dataset_index, target_col).to_f64().unwrap();
                let mut col_index = 0usize;
                for dw in &mut dws {
                    if col_index == target_col {
                        continue;
                    }
                    *dw += dataset.get(dataset_index, col_index).to_f64().unwrap() * error * n_inv;
                    col_index += 1;
                }
                db += error;
            }
            db *= n_inv;

            self.bias -= self.learning_rate * db;
            for i in 0..num_weights {
                self.weights[i] -= dws[i] * self.learning_rate;
            }
        }
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    fn normalize(dataset: &mut BaseDataset, standardizer: Standardizer, targetcol: usize) {
        match standardizer {
            Standardizer::MinMax => {
                let mins = dataset
                    .columns()
                    .iter()
                    .enumerate()
                    .filter(|x| x.0 != targetcol)
                    .map(|x| dataset.min(x.1))
                    .collect::<Vec<DType>>();
                let maxs = dataset
                    .columns()
                    .iter()
                    .enumerate()
                    .filter(|x| x.0 != targetcol)
                    .map(|x| dataset.max(x.1))
                    .collect::<Vec<DType>>();
                for (index, mut col) in dataset.cols_mut().into_iter().enumerate() {
                    if index == targetcol {
                        continue;
                    }
                    for elem in col.iter_mut() {
                        *elem = (&*elem - &mins[index]) / (&maxs[index] - &mins[index])
                    }
                }
            }
            Standardizer::ZScore => {
                unimplemented!()
            }
        }
    }

    fn nfit(&mut self, dataset: &BaseDataset, target_col: &str) {
        assert!(self.num_iters != 0);
        let col_index = dataset._get_string_index(target_col);
        let (nsamples, nfeatures) = dataset.shape();
        println!("{:?}", dataset.shape());
        let mut rng = thread_rng();
        //turn everything
        let x = dataset.into_f64_array(col_index);
        let target = dataset
            .get_col(col_index)
            .map(|x| x.to_f64().unwrap())
            .into_owned();
        let mut weights = Array1::from_vec(
            (1..nfeatures)
                .map(|_| rng.gen::<f64>())
                .collect::<Vec<f64>>(),
        );
        let mut preds = Array1::from_elem(nsamples, 0f64);
        let mut dw = Array1::from_elem(nsamples, 0f64);
        let mut db = 0f64;
        for _ in 0..self.num_iters {
            preds = x.dot(&weights) + self.bias;
            dw = (1f64 / nsamples as f64) * (x.t().dot(&(&preds - &target)));
            db = (1f64 / nsamples as f64) * (&preds - &target).sum();
            weights = weights - &(self.learning_rate * dw);
            self.bias = self.bias - self.learning_rate * db;
        }
        self.weights = weights.to_vec();
    }

    fn predict(&self, data: &Array2<f64>) -> Vec<f64> {
        let mut predictions = (0..data.shape()[1]).map(|_| 0f64).collect::<Vec<f64>>();

        predictions
    }
}

enum Standardizer {
    MinMax,
    ZScore,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    #[test]
    fn test_convergence() {
        let mut dataset = base_array::base_dataset::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        dataset.drop_na(None, true);
        let mut learner = LinearRegressorBuilder::new();
        LinearRegressorBuilder::normalize(&mut dataset, Standardizer::MinMax, 13);
        learner.nfit(&dataset, "MEDV");
        println!(
            "MAE: {}",
            8
        )
    }
}
