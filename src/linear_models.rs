use core::num;

use float_derive::utils::eq;
use ndarray::{s, Array1, Array2, ArrayView1};
use num_traits::ToPrimitive;
use rand::{random, thread_rng, Rng};

use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
    utils::{linalg::{dot, solve_linear_systems}, scaler::Scaler, model_selection},
    *,
};

pub struct LinearRegressorBuilder{
    weights: Vec<f64>,
    bias: f64,
    scaler: Scaler,
    train_test_split: model_selection::TrainTestSplitStrategy,
    target_col: usize,
}

impl LinearRegressorBuilder {
    pub fn new() -> Self {
        Self {
            weights: vec![],
            bias: 0f64,
            scaler: Scaler::None,
            train_test_split: model_selection::TrainTestSplitStrategy::None,
            target_col: 0, 
        }
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn predict(&self, data: &Array2<f64>, target_col: usize) -> Vec<f64> {
        let mut predictions = Vec::new();
        for row in data.rows() {
            let mut current = 0f64;
            for (index, elem) in row.iter().enumerate() {
                if index != target_col {
                    current += elem * self.weights[index];
                }
            }
            predictions.push(current + self.bias);
        }
        predictions
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        let target_col_index = dataset._get_string_index(target);
        let target = dataset.get_col(target_col_index);
        let (nrows, ncols) = dataset.shape();
        let nweights = ncols - 1; //cause we're taking in the full dataset; makes sense
        let mut eqns = Array2::from_elem((0, ncols + 1), 0f64); //we don't have info about the shape of this array
        let mut first = Array1::from_elem(ncols + 1, 0f64);
        self.weights = Vec::from_iter((0..nweights).map(|_| 0f64));
        first[0] = nrows as f64;
        (0..ncols)
            .filter(|index| *index != target_col_index)
            .for_each(|index| {
                first[index + 1] = dataset.get_col(index).sum().to_f64().unwrap();
            });
        //pushes the target col to the last in eqns; nice
        first[ncols] = target.sum().to_f64().unwrap();
        eqns.push_row(first.view())
            .expect("First eqn couldn't fit in");
        let nsums = ((ncols - 1) * (ncols)) / 2;
        let mut sums: Vec<f64> = Vec::from_iter((0..nsums).map(|_| 0f64));
        for elem in 0..nsums {
            let mut first_col = 0;
            let mut group_ind: isize = elem as isize;
            loop {
                group_ind -= nweights as isize - first_col as isize;
                if group_ind < 0 {
                    break;
                }
                first_col += 1;
            }
            let mut second_col = (elem as isize
                - ((nweights as isize * 2 - first_col as isize + 1) * first_col as isize / 2))
                + first_col as isize;
            first_col = if first_col < target_col_index {
                first_col
            } else {
                first_col + 1
            };
            second_col = if second_col < target_col_index as isize {
                second_col
            } else {
                second_col + 1
            };
            sums[elem] = utils::linalg::dot(
                dataset.get_col(first_col),
                dataset.get_col(second_col as usize),
            );
        }
        for elem in 0..nweights {
            let mut current = Vec::new();
            //bias
            current.push(eqns[(0, elem + 1)]);
            for elem_two in 0..nweights {
                current.push(sums[Self::_sum_index(elem, elem_two, nweights)]);
            }
            let non_target_index = if elem < target_col_index {
                elem
            } else {
                elem + 1
            };
            let dot = utils::linalg::dot(
                dataset.get_col(non_target_index),
                dataset.get_col(target_col_index),
            );
            current.push(dot);
            eqns.push_row(Array1::from_vec(current).view())
                .expect("Shape error");
        }
        solve_linear_systems(&mut eqns.view_mut());
        self.bias = eqns[(0, nweights + 1)];
        self.weights.resize(nweights, 0f64);
        for elem in 1..=nweights {
            self.weights[elem - 1] = eqns[(elem, nweights + 1)];
        }
    }


    fn _sum_index(eqn: usize, param: usize, nweights: usize) -> usize {
        let first = usize::min(eqn, param);
        let second = usize::max(eqn, param);
        let mut s_index = 0;
        s_index = (nweights * 2 - first + 1) * first / 2;
        s_index += second - first;
        s_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        utils::metrics::{mean_abs_error, mean_squared_error, root_mean_square_error},
        *,
    };
    #[test]
    fn test_convergence() {
        let mut dataset = base_array::base_dataset::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        //dataset.drop_na(None, true);
        let mut learner = LinearRegressorBuilder::new();
        utils::scaler::Scaler::normalize(&mut utils::scaler::Scaler::MinMax, &mut dataset, 13);
        learner.fit(&dataset, "MEDV");
        let predicted = learner.predict(&dataset.into_f64_array(), 13);
        let error = root_mean_square_error(
            dataset
                .get_col(13)
                .map(|x| x.to_f64().unwrap())
                .as_slice()
                .unwrap(),
            predicted.as_slice(),
        );
        println!("MAE: {}", error)
    }
}
