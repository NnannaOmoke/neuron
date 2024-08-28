use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
    utils::{
        math::{
            argmax_1d_f64, dot, nunique, one_hot_encode_1d, outer_product, softmax_1d,
            solve_linear_systems,
        },
        model_selection::{self, TrainTestSplitStrategy, TrainTestSplitStrategyData},
        scaler::{Scaler, ScalerState},
    },
    *,
};
use core::num;
use either::Either;
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

pub struct LogisticRegressorBuilder {
    scaler: ScalerState,
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, u32>,
    target_index: usize,
    internal: RawLogisticRegressor,
}

#[derive(Clone, Default, Debug)]
pub struct RawLogisticRegressor {
    weights: Array2<f64>,
    bias: f64,
    regularizer: Option<f64>,
    nclasses: usize,
    include_bias: bool,
    max_iters: usize,
}

impl RawLogisticRegressor {
    //uses the SAG algorithm
    //another alternative for super-fast convergence is the irls
    pub fn fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<u32>) {
        self.nclasses = nunique(target);
        let f = if self.include_bias {
            let mut features = features.to_owned();
            features
                .push_column(Array1::ones(features.nrows()).view())
                .unwrap();
            features
        } else {
            Array2::default((0, 0))
        };
        let features = if self.include_bias {
            f.view()
        } else {
            features.view()
        };
        match self.nclasses {
            1 => panic!("Not enough classes to perform a classification task"),
            2 => self.binary_fit(features, target),
            _ => self.multinomial_fit(features, target),
        }
    }

    pub fn binary_fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<u32>) {
        let epochs = self.max_iters;
        let l1_regularization = self.regularizer;
        let (nrows, ncols) = (features.shape()[0], features.shape()[1]);
        let mut weights = Array1::ones(ncols);
        let mut grads = Array1::zeros(nrows);
        let mut grad_sum = Array1::zeros(ncols);
        let mut rand_gen = rand::thread_rng();
        let mut choice = rand_gen.gen_range(0..target.len());
        let mut seen = 1;
        for _ in 0..epochs {
            let curr_x = features.row(choice);
            let curr_y = target[choice];
            let predictions = utils::math::sigmoid(curr_x.dot(&weights));
            let grad = curr_y as f64 - predictions;
            grad_sum = grad_sum - (grads[choice] * curr_x.to_owned()) + grad;
            grads[choice] = grad;
            match l1_regularization {
                Some(lambda) => {
                    weights = ((1.0 - (0.01 * lambda)) * weights)
                        - ((0.01 / seen as f64) * grad_sum.to_owned())
                }
                None => weights = weights - ((0.01 / seen as f64) * grad_sum.to_owned()),
            };
            seen += 1;
            choice = rand_gen.gen_range(0..nrows);
        }
        let weights = if self.include_bias {
            self.bias = *weights.last().unwrap();
            weights.slice(s![..-1]).to_owned()
        } else {
            weights
        };
        let array = Array2::from_shape_fn((1, weights.len()), |(_, y)| weights[y]);
        self.weights = array;
    }

    pub fn multinomial_fit(&mut self, features: ArrayView2<f64>, labels: ArrayView1<u32>) {
        let nclasses = self.nclasses;
        let epochs = self.max_iters;
        let l1_regularization = self.regularizer;
        let (nrows, ncols) = (features.shape()[0], features.shape()[1]);
        let mut weights = Array2::ones((ncols, nclasses));
        let mut grads = Array2::zeros((nrows, nclasses));
        let mut grad_sum = Array2::zeros((ncols, nclasses));
        let mut rand_gen = thread_rng();
        let mut choice = rand_gen.gen_range(0..nrows);
        let mut seen = 1;
        for _ in 0..epochs {
            let curr_x = features.row(choice);
            let curr_y = one_hot_encode_1d(labels[choice], nclasses);
            let preds = softmax_1d(curr_x.dot(&weights).view());
            let grad = curr_y.map(|x| x.to_f64().unwrap()) - preds;
            grad_sum = grad_sum - outer_product(curr_x, grads.row(choice)) + grad.view();
            grads.row_mut(choice).assign(&grad);
            match l1_regularization {
                Some(lambda) => {
                    weights = ((1.0 - (0.01 * lambda)) * weights)
                        - ((0.01 / seen as f64) * grad_sum.to_owned())
                }
                None => weights = weights - ((0.01 / seen as f64) * grad_sum.to_owned()),
            };
            seen += 1;
            choice = rand_gen.gen_range(0..nrows);
        }
        self.weights = weights;
    }

    fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        if self.nclasses == 2 {
            return self.predict_binary(data);
        }
        self.predict_multinomial(data)
    }

    fn predict_binary(&self, data: ArrayView2<f64>) -> Array1<u32> {
        let weights_array = self.weights.row(0);
        let result = data.dot(&weights_array);
        result
            .map(|x| utils::math::sigmoid(*x + self.bias).to_u32().unwrap())
            .to_owned()
    }

    fn predict_multinomial(&self, data: ArrayView2<f64>) -> Array1<u32> {
        let result = data.dot(&self.weights.view());
        result
            .rows()
            .into_iter()
            .map(|x| argmax_1d_f64(x).to_u32().unwrap())
            .collect::<Array1<u32>>()
    }
}

impl LogisticRegressorBuilder {
    pub fn new() -> Self {
        Self {
            scaler: ScalerState::None,
            strategy: TrainTestSplitStrategy::None,
            data: TrainTestSplitStrategyData::default(),
            target_index: 0,
            internal: RawLogisticRegressor::default(),
        }
    }

    pub fn bias(&self) -> f64 {
        self.internal.bias
    }

    pub fn weights(&self) -> ArrayView2<f64> {
        self.internal.weights.view()
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn max_iters(self, max_iters: usize) -> Self {
        let prev = self.internal;
        Self {
            internal: RawLogisticRegressor { max_iters, ..prev },
            ..self
        }
    }

    pub fn include_bias(self, include_bias: bool) -> Self {
        let prev = self.internal;
        Self {
            internal: RawLogisticRegressor {
                include_bias,
                ..prev
            },
            ..self
        }
    }

    pub fn regularizer(self, value: f64) -> Self {
        let prev = self.internal;
        Self {
            internal: RawLogisticRegressor {
                regularizer: Some(value),
                ..prev
            },
            ..self
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_index = dataset._get_string_index(target);
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
        self.internal_fit()
    }

    fn internal_fit(&mut self) {
        let (features, target) = self.data.get_train();

        self.internal.fit(features, target);
    }
    //uses the SAG algorithm
    //another alternative for super-fast convergence is the irls

    fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        self.internal.predict(data)
    }

    pub fn raw_mut(&mut self) -> &mut RawLogisticRegressor {
        &mut self.internal
    }

    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
    //using a vec because user evaluation functions might return maybe one value or three
    //all the functions we plan to build in will only return one value, however
    where
        F: Fn(ArrayView1<u32>, ArrayView1<u32>) -> Vec<f64>,
    {
        let (features, ground_truth) = match self.strategy {
            TrainTestSplitStrategy::None => {
                //get train data
                self.data.get_train()
            }
            TrainTestSplitStrategy::TrainTest(_) => self.data.get_test(),
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => self.data.get_eval(),
        };
        let preds = self.predict(features);
        function(ground_truth, preds.view())
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
    fn test_convergence_classification() {
        let dataset = base_array::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/diabetes.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut classifier = LogisticRegressorBuilder::new()
            .scaler(ScalerState::MinMax)
            .train_test_split_strategy(TrainTestSplitStrategy::TrainTest(0.7));
        classifier.fit(&dataset, "Outcome");
        let accuracy = classifier.evaluate(accuracy);
        println!("Accuracy: {}", accuracy[0])
    }
}
