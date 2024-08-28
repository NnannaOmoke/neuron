use core::num;
use ndarray::{linalg, s, Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2};
use ndarray_linalg::{solve::Inverse, InnerProduct, Scalar};
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
    linear_models::{
        non_regularizing_fit, ridge_regularizing_fit, LinearRegularizer, _coordinate_descent,
    },
    utils::{
        math::{dot, one_hot_encode_1d, softmax_1d, solve_linear_systems},
        model_selection::{self, TrainTestSplitStrategy, TrainTestSplitStrategyData},
        scaler::{Scaler, ScalerState},
    },
    *,
};

use neuron_macros::CrossValidator;

#[derive(Clone)]
pub struct LinearRegressorBuilder {
    scaler: ScalerState,
    strategy: TrainTestSplitStrategy,
    strategy_data: TrainTestSplitStrategyData<f64, f64>,
    target_col: usize,
    internal: RawLinearRegressor,
}

#[derive(Default, Clone, Debug)]
pub struct RawLinearRegressor {
    weights: Array1<f64>,
    bias: f64,
    include_bias: bool,
    regularizer: LinearRegularizer,
}

impl RawLinearRegressor {
    fn fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<f64>) {
        let weights = match self.regularizer {
            LinearRegularizer::None => non_regularizing_fit(features, target),
            LinearRegularizer::Ridge(var) => ridge_regularizing_fit(features, target, var),
            LinearRegularizer::Lasso(var, iters) => {
                _coordinate_descent(features, target, var, None, iters)
            }
            LinearRegularizer::ElasticNet(l1, l2, iters) => {
                _coordinate_descent(features, target, l1, Some(l2), iters)
            }
        };
        if self.include_bias {
            self.bias = *weights.last().unwrap();
            self.weights = weights.slice(s![..-1]).to_owned();
        } else {
            self.weights = weights;
        }
    }
    fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let mut array = data.dot(&self.weights.view());
        array += self.bias;
        array
    }
}

impl LinearRegressorBuilder {
    pub fn new() -> Self {
        Self {
            scaler: ScalerState::None,
            strategy: TrainTestSplitStrategy::None,
            strategy_data: TrainTestSplitStrategyData::default(),
            target_col: 0,
            internal: RawLinearRegressor::default(),
        }
    }
    pub fn bias(&self) -> f64 {
        self.internal.bias
    }

    pub fn weights(&self) -> ArrayView1<f64> {
        self.internal.weights.view()
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn regularizer(self, regularizer: LinearRegularizer) -> Self {
        let prev = self.internal;
        Self {
            internal: RawLinearRegressor {
                regularizer,
                ..prev
            },
            ..self
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
        self.internal_fit();
    }

    fn internal_fit(&mut self) {
        let (features, target) = self.strategy_data.get_train();
        let f = if self.internal.include_bias {
            let mut features = features.to_owned();
            features
                .push_column(Array1::ones(features.nrows()).view())
                .unwrap();
            features
        } else {
            Array2::default((0, 0))
        };
        let features = if self.internal.include_bias {
            f.view()
        } else {
            features
        };
        self.internal.fit(features, target);
    }

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        self.internal.predict(data)
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
        let preds = self.predict(features);
        function(ground_truth, preds.view())
    }

    pub(crate) fn raw_mut(&mut self) -> &mut RawLinearRegressor {
        &mut self.internal
    }
}

impl Display for LinearRegressorBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}]", self.internal)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        utils::{
            metrics::{mean_abs_error, mean_squared_error, root_mean_square_error},
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
        let mut learner = LinearRegressorBuilder::new()
            .scaler(utils::scaler::ScalerState::MinMax)
            .train_test_split_strategy(utils::model_selection::TrainTestSplitStrategy::TrainTest(
                0.7,
            ))
            .regularizer(LinearRegularizer::ElasticNet(0.3, 0.7, 10));

        learner.fit(&dataset, "MEDV");
        let mae = learner.evaluate(mean_abs_error);
        let mse = learner.evaluate(mean_squared_error);
        let rmse = learner.evaluate(root_mean_square_error);
        println!("mae: {}\nmse: {}\nrmse: {}", mae[0], mse[0], rmse[0]);
    }
}
