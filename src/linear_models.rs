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
        model_selection::{self, TrainTestSplitStrategy, TrainTestSplitStrategyData},
        scaler::{Scaler, ScalerState},
    },
    *,
};

pub struct LinearRegressorBuilder {
    weights: Vec<f64>,
    bias: f64,
    scaler: ScalerState,
    strategy: TrainTestSplitStrategy,
    strategy_data: TrainTestSplitStrategyData<f64, f64>,
    target_col: usize,
    regularizer: LinearRegularizer,
    include_bias: bool,
}

impl LinearRegressorBuilder {
    pub fn new(include_bias: bool) -> Self {
        Self {
            weights: vec![],
            bias: 0f64,
            scaler: ScalerState::None,
            strategy: TrainTestSplitStrategy::None,
            strategy_data: TrainTestSplitStrategyData::default(),
            target_col: 0,
            regularizer: LinearRegularizer::None,
            include_bias,
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

    pub fn predict_external(&self, data: ArrayView2<f64>) -> Vec<f64> {
        let mut predictions = Vec::new();
        for row in data.rows() {
            let mut current = 0f64;
            for (index, elem) in row.iter().enumerate() {
                current += elem * self.weights[index];
            }
            predictions.push(current + self.bias);
        }
        predictions
    }

    pub fn predict(&self) -> Vec<f64> {
        match self.strategy {
            TrainTestSplitStrategy::None => self.predict_external(self.strategy_data.get_train().0),
            TrainTestSplitStrategy::TrainTest(_) => {
                self.predict_external(self.strategy_data.get_test().0)
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                self.predict_external(self.strategy_data.get_eval().0)
            }
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_col = dataset._get_string_index(target);
        self.strategy_data =
            TrainTestSplitStrategyData::<f64, f64>::new_r(dataset, self.target_col, self.strategy);
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(self.strategy_data.get_train().0);
        scaler.transform(&mut self.strategy_data.get_train_mut().0);
        match self.strategy {
            TrainTestSplitStrategy::TrainTest(_) => {
                scaler.transform(&mut self.strategy_data.get_test_mut().0)
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                scaler.transform(&mut self.strategy_data.get_test_mut().0);
                scaler.transform(&mut self.strategy_data.get_eval_mut().0);
            }
            _ => {}
        };
        self.tfit();
    }

    fn tfit(&mut self) {
        if self.include_bias {
            self.strategy_data
                .train_features
                .push_column(Array1::ones(self.strategy_data.train_features.nrows()).view())
                .unwrap();
        }
        let (features, target) = self.strategy_data.get_train();
        let weights = match self.regularizer {
            LinearRegularizer::None => Self::non_regularizing_fit(features.view(), target),
            LinearRegularizer::Ridge(var) => Self::ridge_regularizing_fit(features, target, var),
            LinearRegularizer::Lasso(var, iters) => {
                Self::_coordinate_descent(features, target, var, None, iters)
            }
            LinearRegularizer::ElasticNet(l1, l2, iters) => {
                Self::_coordinate_descent(features, target, l1, Some(l2), iters)
            }
        };
        if self.include_bias {
            self.weights = weights.to_vec()[..weights.len() - 1].to_vec();
            self.bias = *weights.last().unwrap();
        } else {
            self.weights = weights.to_vec();
        }
    }

    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
    //using a vec because user evaluation functions might return maybe one value or three
    //all the functions we plan to build in will only return one value, however
    where
        F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Vec<f64>,
    {
        let (features, ground_truth) = match self.strategy {
            TrainTestSplitStrategy::None => {
                //get train data
                self.strategy_data.get_train()
            }
            TrainTestSplitStrategy::TrainTest(_) => self.strategy_data.get_test(),
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => self.strategy_data.get_eval(),
        };
        let preds = self.predict_external(features);
        let preds = Array1::from_vec(preds);
        function(ground_truth, preds.view())
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
        features: ArrayView2<f64>,
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
    data: TrainTestSplitStrategyData<f64, u32>,
    regularizer: Option<f64>,
    target_index: usize,
    include_bias: bool,
}

impl LogisticRegressorBuilder {
    pub fn new(include_bias: bool) -> Self {
        Self {
            weights: vec![],
            bias: 0.0,
            scaler: ScalerState::None,
            strategy: TrainTestSplitStrategy::None,
            data: TrainTestSplitStrategyData::default(),
            regularizer: None,
            target_index: 0,
            include_bias,
        }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn regularizer(self, value: f64) -> Self{
        Self {regularizer: Some(value), ..self}
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_index = dataset._get_string_index(target);
        let nlabels = dataset.nunique(target);
        //splits into tts
        self.data = TrainTestSplitStrategyData::<f64, u32>::new_c(
            dataset,
            self.target_index,
            self.strategy,
        );
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(self.data.get_train().0);
        scaler.transform(&mut self.data.get_train_mut().0);
        match self.strategy {
            TrainTestSplitStrategy::TrainTest(_) => {
                scaler.transform(&mut self.data.get_test_mut().0)
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                scaler.transform(&mut self.data.get_test_mut().0);
                scaler.transform(&mut self.data.get_eval_mut().0);
            }
            _ => {}
        };
        self.tfit(nlabels)
    }

    pub fn tfit(&mut self, nclasses: usize) {
        if self.include_bias {
            self.data
                .train_features
                .push_column(Array1::ones(self.data.train_features.nrows()).view())
                .unwrap();
        }
        let (features, target) = self.data.get_train();
        let weights = if nclasses == 2 {
            Self::binary_fit(features, target, 50, self.regularizer)
        } else if nclasses > 2 {
            unimplemented!()
        } else {
            panic!("Only one target class!")
        };
        if self.include_bias {
            self.weights = weights.to_vec()[..weights.len() - 1].to_vec();
            self.bias = *weights.last().unwrap();
        } else {
            self.weights = weights.to_vec();
        }
    }
    //uses the SAG algorithm
    //another alternative for super-fast convergence is the irls
    pub fn binary_fit(
        features: ArrayView2<f64>,
        target: ArrayView1<u32>,
        epochs: usize,
        l1_regularization: Option<f64>,
    ) -> Array1<f64> {
        let (nrows, ncols) = (features.shape()[0], features.shape()[1]);
        let mut weights = Array1::ones(ncols);
        let mut gradients = Array1::zeros(nrows);
        let mut gradient_sum = Array1::zeros(ncols);
        let mut rand_gen = rand::thread_rng();
        let mut curr_rand_index = rand_gen.gen_range(0..target.len());
        let mut seen = 1;
        for _ in 0..epochs {
            let curr_x = features.row(curr_rand_index);
            let curr_y = target[curr_rand_index];
            let predictions = utils::linalg::sigmoid(curr_x.dot(&weights));
            let gradient = curr_y as f64 - predictions;
            gradient_sum =
                gradient_sum - (gradients[curr_rand_index] * curr_x.to_owned()) + gradient;
            gradients[curr_rand_index] = gradient;
            match l1_regularization{
                Some(lambda) => weights = ((1.0 - (0.01 * lambda)) * weights) - ((0.01 / seen as f64) * gradient_sum.to_owned()),
                None => weights = weights - ((0.01 / seen as f64) * gradient_sum.to_owned())
            };
            seen += 1;
            curr_rand_index = rand_gen.gen_range(0..target.len());
        }
        weights
    }

    fn predict_external(&self, data: &ArrayView2<f64>) -> Array1<u32> {
        let weights_array = Array1::from_vec(self.weights.clone());
        let result = data.dot(&weights_array);
        result.map(|x| {utils::linalg::sigmoid(*x).to_u32().unwrap()}).to_owned()
    }

    pub fn predict(&self) -> Array1<u32> {
        match self.strategy {
            TrainTestSplitStrategy::None => self.predict_external(&self.data.get_train().0),
            TrainTestSplitStrategy::TrainTest(_) => self.predict_external(&self.data.get_test().0),
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                self.predict_external(&self.data.get_eval().0)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        utils::{
            metrics::{accuracy, mean_abs_error, mean_squared_error, root_mean_square_error},
            scaler,
        },
        *,
    };
    #[test]
    fn test_convergence_regression() {
        let dataset = base_array::base_dataset::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut learner = LinearRegressorBuilder::new(false)
            .scaler(utils::scaler::ScalerState::MinMax)
            .train_test_split_strategy(utils::model_selection::TrainTestSplitStrategy::TrainTest(
                0.7,
            ))
            .regularizer(LinearRegularizer::ElasticNet(0.3, 0.7, 10));

        learner.fit(&dataset, "MEDV");
        let preds = learner.predict();
        let exact = learner.strategy_data.get_test().1.to_vec();
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

    #[test]
    fn test_convergence_classification() {
        let dataset = base_array::base_dataset::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/diabetes.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut classifier = LogisticRegressorBuilder::new(false)
            .scaler(ScalerState::MinMax)
            .train_test_split_strategy(TrainTestSplitStrategy::TrainTest(0.7))
            .regularizer(0.01);
        classifier.fit(&dataset, "Outcome");
        let predictions = classifier.predict();
        let ground_truth = classifier.data.get_test().1;
        dbg!(accuracy(&ground_truth.to_vec(), &predictions.to_vec()));
    }

    #[test]
    fn test_sigmoid(){
        assert_eq!(utils::linalg::sigmoid(2.1), 0.8909031788043871);
    }
}
