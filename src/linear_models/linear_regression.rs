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
    utils::{
        linalg::{dot, one_hot_encode_1d, softmax_1d, solve_linear_systems},
        model_selection::{self, TrainTestSplitStrategy, TrainTestSplitStrategyData},
        scaler::{Scaler, ScalerState},
    },
    linear_models::{LinearRegularizer,non_regularizing_fit, ridge_regularizing_fit, _coordinate_descent},
    
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

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<f64>{
        let weights = Array1::from_vec(self.weights.clone());
        let mut array = data.dot(&weights.view());
        array += self.bias;
        array
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
        if self.include_bias {
            self.strategy_data
                .train_features
                .push_column(Array1::ones(self.strategy_data.train_features.nrows()).view())
                .unwrap();
        }
        let (features, target) = self.strategy_data.get_train();
        let weights = match self.regularizer {
            LinearRegularizer::None => non_regularizing_fit(features.view(), target),
            LinearRegularizer::Ridge(var) => ridge_regularizing_fit(features, target, var),
            LinearRegularizer::Lasso(var, iters) => {
                _coordinate_descent(features, target, var, None, iters)
            }
            LinearRegularizer::ElasticNet(l1, l2, iters) => {
                _coordinate_descent(features, target, l1, Some(l2), iters)
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
        let preds = self.predict(features);
        function(ground_truth, preds.view())
    }

}



#[cfg(test)]
mod tests{
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
        let mut learner = LinearRegressorBuilder::new(false)
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
