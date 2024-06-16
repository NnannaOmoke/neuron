use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
    utils::{
        linalg::{argmax_1d_f64, dot, one_hot_encode_1d, softmax_1d, solve_linear_systems},
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
    weights: Array2<f64>,
    bias: f64,
    scaler: ScalerState,
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, u32>,
    regularizer: Option<f64>,
    nclasses: usize,
    target_index: usize,
    include_bias: bool,
}

impl LogisticRegressorBuilder {
    pub fn new(include_bias: bool) -> Self {
        Self {
            weights: Array2::default((0, 0)),
            bias: 0.0,
            scaler: ScalerState::None,
            strategy: TrainTestSplitStrategy::None,
            data: TrainTestSplitStrategyData::default(),
            regularizer: None,
            nclasses: 0,
            target_index: 0,
            include_bias,
        }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn weights(&self) -> ArrayView2<f64> {
        self.weights.view()
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn regularizer(self, value: f64) -> Self {
        Self {
            regularizer: Some(value),
            ..self
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_index = dataset._get_string_index(target);
        let nlabels = dataset.nunique(target);
        self.nclasses = nlabels;
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
        if self.include_bias {
            self.data
                .train_features
                .push_column(Array1::ones(self.data.train_features.nrows()).view())
                .unwrap();
        }
        let (features, target) = self.data.get_train();
        if self.nclasses == 1 {
            panic!("Not enough classes!");
        }
        if self.nclasses == 2 {
            let weights_1d = Self::binary_fit(features, target, 100, self.regularizer);
            if self.include_bias {
                let mut weights_2d = Array2::from_elem((1, weights_1d.len() - 1), 0f64);
                weights_2d.row_mut(0).assign(&weights_1d.slice(s![..-1]));
                self.weights = weights_2d;
                self.bias = *weights_1d.last().unwrap();
            } else {
                let mut weights_2d = Array2::from_elem((1, weights_1d.len()), 0f64);
                weights_2d.row_mut(0).assign(&weights_1d.view());
                self.weights = weights_2d;
            }
        } else if self.nclasses > 2 {
            let weights =
                Self::multinomial_fit(features, target, 100, self.regularizer, self.nclasses);
            self.weights = weights;
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
        let mut grads = Array1::zeros(nrows);
        let mut grad_sum = Array1::zeros(ncols);
        let mut rand_gen = rand::thread_rng();
        let mut choice = rand_gen.gen_range(0..target.len());
        let mut seen = 1;
        for _ in 0..epochs {
            let curr_x = features.row(choice);
            let curr_y = target[choice];
            let predictions = utils::linalg::sigmoid(curr_x.dot(&weights));
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
        weights
    }

    pub fn multinomial_fit(
        features: ArrayView2<f64>,
        labels: ArrayView1<u32>,
        epochs: usize,
        l1_regularization: Option<f64>,
        nclasses: usize,
    ) -> Array2<f64> {
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
            grad_sum = grad_sum - (curr_x.inner(&grads.row(choice))) + grad.view();
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
        weights
    }

    fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        let weights_array = self.weights.row(0);
        let result = data.dot(&weights_array);
        result
            .map(|x| utils::linalg::sigmoid(*x + self.bias).to_u32().unwrap())
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
        let mut classifier = LogisticRegressorBuilder::new(false)
            .scaler(ScalerState::MinMax)
            .train_test_split_strategy(TrainTestSplitStrategy::TrainTest(0.7));
        classifier.fit(&dataset, "Outcome");
        let accuracy = classifier.evaluate(accuracy);
        println!("Accuracy: {}", accuracy[0])
    }
}
