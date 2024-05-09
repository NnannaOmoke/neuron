use core::num;
use ndarray::{linalg, s, Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2};
use ndarray_linalg::{solve::Inverse, Scalar};
use num_traits::ToPrimitive;
use rand::{random, rngs, seq::SliceRandom, thread_rng, Rng};
use std::{collections::HashSet, default, ops::Rem};

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
    regularizer: LinearRegularizer,
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
            regularizer: LinearRegularizer::None,
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

    pub fn regularizer(self, regularizer: LinearRegularizer) -> Self {
        Self {
            regularizer,
            ..self
        }
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
        self.tfit();
    }

    fn tfit(&mut self) {
        let mut features = if self.target_col != self.train.ncols() - 1 {
            let mut train_features = Array2::from_elem((self.train.nrows(), 0), 0f64);
            for (_, col) in self
                .train
                .columns()
                .into_iter()
                .enumerate()
                .filter(|(index, _)| *index != self.target_col)
            {
                train_features.push_column(col).unwrap();
            }
            train_features
        } else {
            self.train
                .slice(s![.., ..self.train.ncols() - 1])
                .to_owned()
        };
        let target = self.train.column(self.target_col);
        let ones = Array1::ones(features.nrows());
        features.push_column(ones.view()).expect("Shape error");
        let check = match self.regularizer {
            LinearRegularizer::None => Self::non_regularizing_fit(features.view(), target),
            LinearRegularizer::Ridge(var) => {
                Self::ridge_regularizing_fit(features.view(), target, var)
            }
            LinearRegularizer::Lasso(var, iters) => {
                Self::_cordinate_descent(features, target, var, None, iters)
            }
            LinearRegularizer::ElasticNet(l1, l2, iters) => {
                Self::_cordinate_descent(features, target, l1, Some(l2), iters)
            }
        };
        self.weights = check.to_vec()[..check.len() - 1].to_vec();
        self.bias = *check.last().unwrap();
    }

    fn non_regularizing_fit(features: ArrayView2<f64>, target: ArrayView1<f64>) -> Array1<f64> {
        let feature_t = features.t();
        let left = feature_t.dot(&features);
        let right = feature_t.dot(&target);
        let left = left.inv().expect("Inversion Failed");
        let check = left.dot(&right);
        check
    }

    fn ridge_regularizing_fit(
        features: ArrayView2<f64>,
        target: ArrayView1<f64>,
        regularizer: f64,
    ) -> Array1<f64> {
        let feature_t = features.t();
        let eye = Array2::eye(features.ncols());
        let l_left = feature_t.dot(&features);
        let l_right = regularizer * eye;
        let left = l_left + l_right;
        let left_inv = left.inv().expect("Inversion Failed");
        let right = feature_t.dot(&target);
        left_inv.dot(&right)
    }

    fn _cordinate_descent(
        features: Array2<f64>,
        target: ArrayView1<f64>,
        l1_regularizer: f64,
        l2_regularizer: Option<f64>,
        iters: usize,
    ) -> Array1<f64> {
        let ncols = features.shape()[1];
        let mut weights = Array1::ones(ncols);
        for _ in 0..iters {
            for (index, col) in features.columns().into_iter().enumerate() {
                let step =
                    Self::_compute_step_col(features.view(), target, weights.view(), index, col);
                let col_norm_factor = Self::_compute_norm_term(col);
                weights[index] = match l2_regularizer {
                    Some(var) => {
                        Self::elastic_net_soft_threshold(step, l1_regularizer, var, col_norm_factor)
                    }
                    None => Self::lasso_soft_threshold(step, l1_regularizer, col_norm_factor),
                }
            }
        }
        weights
    }

    fn lasso_soft_threshold(rho: f64, lambda: f64, col_norm_factor: f64) -> f64 {
        if rho < -lambda {
            (rho + lambda) / col_norm_factor
        } else if rho > lambda {
            (rho - lambda) / col_norm_factor
        } else {
            0f64
        }
    }

    fn elastic_net_soft_threshold(rho: f64, l1: f64, l2: f64, col_norm_factor: f64) -> f64 {
        let gamma = if rho > 0f64 && (l1 * l2) < rho.abs() {
            rho - (l1 * l2)
        } else if rho < 0f64 && (l1 * l2) < rho.abs() {
            rho + (l1 * l2)
        } else {
            0f64
        };
        let gamma_interlude = gamma / (1f64 + l1 * (1f64 - l2));
        gamma_interlude / col_norm_factor
    }

    fn _sign(var: f64) -> f64 {
        if var > 0f64 {
            1f64
        } else if var < 0f64 {
            -1f64
        } else {
            0f64
        }
    }
    fn _compute_step_col(
        features: ArrayView2<f64>,
        target: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        index: usize,
        col: ArrayView1<f64>,
    ) -> f64 {
        let mut feature_clone = features.to_owned();
        feature_clone.remove_index(Axis(1), index);
        let mut weight_clone = weights.to_owned();
        weight_clone.remove_index(Axis(0), index);
        let prediction = feature_clone.dot(&weight_clone.view());
        let res = target.to_owned() - prediction;
        let step_col = col.dot(&res);
        step_col
    }

    fn _compute_norm_term(col: ArrayView1<f64>) -> f64 {
        col.dot(&col)
    }
}

pub enum LinearRegularizer {
    None,
    Ridge(f64),
    Lasso(f64, usize),
    ElasticNet(f64, f64, usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        utils::{
            metrics::{mean_abs_error, mean_squared_error, root_mean_square_error},
            scaler,
        },
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
            .scaler(utils::scaler::ScalerState::ZScore)
            .train_test_split_strategy(utils::model_selection::TrainTestSplitStrategy::TrainTest(
                0.7,
            ))
            .regularizer(LinearRegularizer::ElasticNet(0.3, 0.7, 20));

        learner.fit(&dataset, "MEDV");
        let preds = learner.predict();
        let exact = learner.get_test_data().column(13).to_vec();
        let mae = utils::metrics::mean_abs_error(&exact, &preds);
        let rmse = utils::metrics::root_mean_square_error(&exact, &preds);
        let mse = utils::metrics::mean_squared_error(&exact, &preds);
        println!(
            "
            MAE: {mae}, 
            RMSE: {rmse}, 
            MSE: {mse}
        "
        );
    }
}
