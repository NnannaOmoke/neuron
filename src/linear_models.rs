use core::num;

use num_traits::ToPrimitive;
use rand::random;

use crate::base_array::{base_dataset::BaseDataset, BaseMatrix};

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
            learning_rate: 0f64,
            num_iters: 0,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    #[test]
    fn test_convergence() {
        let dataset = base_array::base_dataset::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut learner = LinearRegressorBuilder::new();
        learner.fit(&dataset, 13);
        println!("{:?}", learner.weights())
    }
}
