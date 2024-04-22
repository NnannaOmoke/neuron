use core::num;
use std::{collections::HashSet, default};

use float_derive::utils::eq;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut2};
use num_traits::ToPrimitive;
use rand::{random, rngs, seq::SliceRandom, thread_rng, Rng};

use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
    utils::{
        linalg::{dot, solve_linear_systems},
        model_selection::{self, TrainTestSplitStrategy},
        scaler::{Scaler, ScalerState},
    },
    *,
};

pub struct LinearRegressorBuilder {
    weights: Vec<f64>,
    bias: f64,
    scaler: ScalerState,
    train_test_split: model_selection::TrainTestSplitStrategy,
    target_col: usize,
    train: Array2<f64>,
    test: Array2<f64>,
    eval: Array2<f64>,
}

impl LinearRegressorBuilder {
    pub fn new() -> Self {
        Self {
            weights: vec![],
            bias: 0f64,
            scaler: ScalerState::None,
            train_test_split: model_selection::TrainTestSplitStrategy::None,
            target_col: 0,
            train: Array2::default((0, 0)),
            test: Array2::default((0, 0)),
            eval: Array2::default((0, 0)),
        }
    }
    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn get_train_data(&self) -> ArrayView2<f64> {
        self.train.view()
    }

    pub fn get_train_data_mut(&mut self) -> ArrayViewMut2<f64> {
        self.train.view_mut()
    }

    pub fn get_test_data(&self) -> ArrayView2<f64> {
        self.test.view()
    }

    pub fn get_test_data_mut(&mut self) -> ArrayViewMut2<f64> {
        self.test.view_mut()
    }

    pub fn get_eval_data(&self) -> ArrayView2<f64> {
        self.eval.view()
    }

    pub fn get_eval_data_mut(&mut self) -> ArrayViewMut2<f64> {
        self.eval.view_mut()
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self {
            train_test_split: strategy,
            ..self
        }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }
    pub fn predict_external(&self, data: &Array2<f64>, target_col: usize) -> Vec<f64> {
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
    pub fn predict(&self) -> Vec<f64> {
        match self.train_test_split {
            TrainTestSplitStrategy::None => self.predict_external(&self.train, self.target_col),
            TrainTestSplitStrategy::TrainTest(_) => {
                self.predict_external(&self.test, self.target_col)
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                self.predict_external(&self.eval, self.target_col)
            }
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_col = dataset._get_string_index(target);
        let complete_array = dataset.into_f64_array();
        let mut indices = Vec::from_iter(0..dataset.len());
        let mut rngs = thread_rng();
        indices.shuffle(&mut rngs);
        //this should shuffle the indices, create an intermediate 2d array that we'll split based on the train-test-split strategy
        let mut train = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut test = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut eval = Array2::from_elem((0, dataset.shape().1), 0f64);
        match self.train_test_split {
            TrainTestSplitStrategy::None => {
                for elem in indices {
                    train
                        .push_row(complete_array.row(elem))
                        .expect("Shape error");
                }
            }
            TrainTestSplitStrategy::TrainTest(train_r) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train
                        .push_row(complete_array.row(*elem))
                        .expect("Shape error");
                }
                for elem in &indices[train_ratio..] {
                    test.push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
            }
            TrainTestSplitStrategy::TrainTestEval(train_r, test_r, _) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                let test_ratio = (test_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train
                        .push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
                for elem in &indices[train_ratio..test_ratio + train_ratio] {
                    test.push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
                for elem in &indices[train_ratio + test_ratio..] {
                    eval.push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
            }
        }
        drop(complete_array);
        self.train = train;
        self.test = test;
        self.eval = eval;
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(&self.train, self.target_col);
        scaler.transform(&mut self.train, self.target_col);
        scaler.transform(&mut self.test, self.target_col);
        scaler.transform(&mut self.eval, self.target_col);
        self.afit();
    }

    fn afit(&mut self) {
        let target_index = self.target_col;
        let target = self.train.column(target_index);
        let (nrows, ncols) = (self.train.shape()[0], self.train.shape()[1]);
        let nweights = ncols - 1; //cause we're taking in the full dataset; makes sense
        let mut eqns = Array2::from_elem((0, ncols + 1), 0f64); //we don't have info about the shape of this array
        let mut first = Array1::from_elem(ncols + 1, 0f64);
        self.weights = Vec::from_iter((0..nweights).map(|_| 0f64));
        first[0] = nrows as f64;
        (0..ncols)
            .filter(|index| *index != target_index)
            .for_each(|index| {
                first[index + 1] = self.train.column(index).sum().to_f64().unwrap();
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
            first_col = if first_col < target_index {
                first_col
            } else {
                first_col + 1
            };
            second_col = if second_col < target_index as isize {
                second_col
            } else {
                second_col + 1
            };
            sums[elem] = utils::linalg::dot(
                self.train.column(first_col),
                self.train.column(second_col as usize),
            );
        }
        for elem in 0..nweights {
            let mut current = Vec::new();
            //bias
            current.push(eqns[(0, elem + 1)]);
            for elem_two in 0..nweights {
                current.push(sums[Self::_sum_index(elem, elem_two, nweights)]);
            }
            let non_target_index = if elem < target_index { elem } else { elem + 1 };
            let dot = utils::linalg::dot(
                self.train.column(non_target_index),
                self.train.column(target_index),
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
        let dataset = base_array::base_dataset::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut learner = LinearRegressorBuilder::new()
            .train_test_split_strategy(TrainTestSplitStrategy::TrainTest(0.7))
            .scaler(ScalerState::MinMax);
        learner.fit(&dataset, "MEDV");
        let preds = learner.predict();
        let exact = learner.test.column(13).to_vec();
        let mae = mean_abs_error(&exact, &preds);
        println!("The MAE is: {}", mae);
    }
}
