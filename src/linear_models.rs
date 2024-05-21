use core::num;
use ndarray::{linalg, s, Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2};
use ndarray_linalg::{solve::Inverse, Scalar};
use num_traits::ToPrimitive;
use rand::{
    random,
    rngs::{self, ThreadRng},
    seq::SliceRandom,
    thread_rng, Rng,
};
use std::{cell::RefCell, collections::HashSet, default, ops::Rem, os::unix::thread};

use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
    utils::{
        linalg::{dot, solve_linear_systems},
        model_selection::{
            self, CTrainTestSplitStrategyData, RTrainTestSplitStrategyData, TrainTestSplitStrategy,
        },
        scaler::{Scaler, ScalerState},
    },
    *,
};

pub struct LinearRegressorBuilder {
    weights: Vec<f64>,
    bias: f64,
    scaler: ScalerState,
    strategy: TrainTestSplitStrategy,
    strategy_data: RTrainTestSplitStrategyData,
    target_col: usize,
    regularizer: LinearRegularizer,
}

impl LinearRegressorBuilder {
    pub fn new() -> Self {
        Self {
            weights: vec![],
            bias: 0f64,
            scaler: ScalerState::None,
            strategy: TrainTestSplitStrategy::None,
            strategy_data: RTrainTestSplitStrategyData::default(),
            target_col: 0,
            regularizer: LinearRegularizer::None,
        }
    }
    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
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
        match self.strategy {
            TrainTestSplitStrategy::None => {
                self.predict_external(&self.strategy_data.train, self.target_col)
            }
            TrainTestSplitStrategy::TrainTest(_) => {
                self.predict_external(&self.strategy_data.test, self.target_col)
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                self.predict_external(&self.strategy_data.eval, self.target_col)
            }
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_col = dataset._get_string_index(target);
        self.strategy_data = RTrainTestSplitStrategyData::new(self.strategy, dataset);
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(&self.strategy_data.train, self.target_col);
        scaler.transform(&mut self.strategy_data.train, self.target_col);
        scaler.transform(&mut self.strategy_data.test, self.target_col);
        scaler.transform(&mut self.strategy_data.eval, self.target_col);
        self.tfit();
    }

    fn tfit(&mut self) {
        let mut features = if self.target_col != self.strategy_data.train.ncols() - 1 {
            let mut train_features = Array2::from_elem((self.strategy_data.train.nrows(), 0), 0f64);
            for (_, col) in self
                .strategy_data
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
            self.strategy_data
                .train
                .slice(s![.., ..self.strategy_data.train.ncols() - 1])
                .to_owned()
        };
        let target = self.strategy_data.train.column(self.target_col);
        let ones = Array1::ones(features.nrows());
        features.push_column(ones.view()).expect("Shape error");
        let check = match self.regularizer {
            LinearRegularizer::None => Self::non_regularizing_fit(features.view(), target),
            LinearRegularizer::Ridge(var) => {
                Self::ridge_regularizing_fit(features.view(), target, var)
            }
            LinearRegularizer::Lasso(var, iters) => {
                Self::_coordinate_descent(features, target, var, None, iters)
            }
            LinearRegularizer::ElasticNet(l1, l2, iters) => {
                Self::_coordinate_descent(features, target, l1, Some(l2), iters)
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
        let weights = left.dot(&right);
        weights
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

    fn _coordinate_descent(
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

#[derive(Clone, Copy)]
pub enum LinearRegularizer {
    None,
    Ridge(f64),
    Lasso(f64, usize),
    ElasticNet(f64, f64, usize),
}

struct LogisticRegressorBuilder {
    weights: Vec<f64>,
    bias: f64,
    scaler: ScalerState,
    strategy: TrainTestSplitStrategy,
    data: CTrainTestSplitStrategyData,
    target_index: usize,
    decision_point: f64,
}

impl LogisticRegressorBuilder {
    pub fn new() -> Self {
        Self {
            weights: vec![],
            bias: 0.0,
            scaler: ScalerState::None,
            strategy: TrainTestSplitStrategy::None,
            data: CTrainTestSplitStrategyData::default(),
            target_index: 0,
            decision_point: 0.5,
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_index = dataset._get_string_index(target);
        //splits into tts
        self.data = CTrainTestSplitStrategyData::new(self.strategy, dataset, self.target_index);
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(&self.data.train_features, self.target_index);
        scaler.transform(&mut self.data.train_features, self.target_index);
        scaler.transform(&mut self.data.test_features, self.target_index);
        scaler.transform(&mut self.data.eval_features, self.target_index);
        self.tfit()
    }

    pub fn tfit(&mut self) {}

    pub fn binary_fit(features: ArrayView2<f64>, target: ArrayView1<u32>) -> Array1<f64> {
        let weights = Array1::ones(features.ncols());

        weights
    }
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
        let exact = learner.strategy_data.get_test().column(13).to_vec();
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
