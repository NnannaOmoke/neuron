use crate::{
    base_array::BaseDataset,
    svm::{
        linear_kernel_1d, linear_kernel_2d, linear_kernel_mixed, polynomial_kernel_1d,
        polynomial_kernel_2d, polynomial_kernel_mixed, rbf_kernel_1d, rbf_kernel_2d,
        rbf_kernel_mixed, sigmoid_kernel_1d, sigmoid_kernel_2d, sigmoid_kernel_mixed, SVMKernel,
    },
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
    utils::{
        math::{argmax_1d_f64, argmin_1d_f64},
        scaler::{Scaler, ScalerState},
    },
    Array2, ArrayView2,
};

use ndarray::{array, Array1, ArrayView1};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::{Arc, RwLock};
pub struct SVCBuilder {
    C: f64,
    kkt_value: f64,
    kernel: SVMKernel,
    svms: Vec<RawSVC>,
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, u32>,
    target_index: usize,
    nclasses: usize,
    scaler: ScalerState,
}

#[derive(Default, Clone)]
struct RawSVC {
    C: f64,
    kkt_value: f64,
    support_vectors: Array2<f64>,
    support_labels: Array1<i32>,
    alphas: Array1<f64>,
    bias: f64,
    kernel: SVMKernel,
}

impl RawSVC {
    pub fn binomial_fit(
        &mut self,
        features: ArrayView2<f64>,
        target: ArrayView1<i32>,
        epochs: usize,
    ) {
        let mut bias = 0.0;
        let (nrows, _) = (features.nrows(), features.ncols());
        let target = target.map(|x| x.to_f64().unwrap()).to_owned();
        let mut alphas: Array1<f64> = Array1::zeros(nrows);
        let mut support_labels = target.to_owned();
        let mut support_vectors = features.to_owned();
        let mut non_kkt_array = Array1::from_shape_fn(nrows, |x| x);
        let mut error_cache = self.predict_raw(features).to_owned() - target;
        for _ in 0..epochs {
            let i2 = self.heuristic_2(&mut non_kkt_array);
            if i2 == -1 {
                break;
            }
            let i1 = self.heuristic_1(error_cache.view(), i2);
            if i1 == i2 as usize {
                continue;
            }
            //get samples, labels and alpha values
            let (feature_one, label_one, alpha_one) = (
                self.support_vectors.row(i1),
                self.support_labels[i1],
                self.alphas[i1],
            );
            let (feature_two, label_two, alpha_two) = (
                self.support_vectors.row(i2 as usize),
                self.support_labels[i2 as usize],
                self.alphas[i2 as usize],
            );
            let (lower, upper) =
                self.compute_boundaries(alpha_one, alpha_two, label_one, label_two);
            if lower == upper {
                continue;
            }
            let eta = self.compute_eta(feature_one, feature_two);
            if eta == 0f64 {
                continue;
            }
            let holder_one = Array2::from_shape_fn((1, nrows), |(_, y)| feature_one[y]);
            let holder_two = Array2::from_shape_fn((1, nrows), |(_, y)| feature_two[y]);
            let score_one = self.predict_raw(holder_one.view());
            let score_two = self.predict_raw(holder_two.view());
            //predict should not return u32s!
            let e1 = score_one[0] as f64 - label_one as f64;
            let e2 = score_two[0] as f64 - label_two as f64;
            let mut alpha_two_new = alpha_two + label_two as f64 * (e1 - e2) / eta;
            alpha_two_new = alpha_two_new.min(upper);
            alpha_two_new = alpha_two_new.max(lower);
            let alpha_one_new =
                alpha_one + label_one as f64 * label_two as f64 * (alpha_two - alpha_two_new);
            bias = self.compute_b(alpha_one_new, alpha_two_new, e1, e2, i1, i2 as usize);
            alphas[i1] = alpha_one_new;
            alphas[i2 as usize] = alpha_two_new;
            error_cache[i1] = self.predict_raw(holder_one.view())[0] - label_one as f64;
            error_cache[i2 as usize] = self.predict_raw(holder_two.view())[0] - label_two as f64;
        }
        let support_indices = alphas
            .iter()
            .enumerate()
            .filter(|(__, value)| **value != 0f64)
            .map(|(index, _)| index)
            .collect::<Array1<usize>>();
        support_labels = Array1::from_shape_fn(support_indices.len(), |x| {
            support_labels[support_indices[x]]
        });
        support_vectors = Array2::from_shape_fn(
            (support_indices.len(), support_vectors.ncols()),
            |(x, y)| support_vectors[(support_indices[x], y)],
        );
        alphas = Array1::from_shape_fn(support_indices.len(), |x| alphas[support_indices[x]]);
        self.support_vectors = support_vectors;
        self.bias = bias;
        self.support_labels = support_labels.map(|x| x.to_i32().unwrap());
        self.alphas = alphas;
    }
    pub fn predict_raw(&self, input: ArrayView2<f64>) -> Array1<f64> {
        let weights =
            self.support_labels.map(|x| x.to_f64().unwrap()).to_owned() * self.alphas.view();
        let distance = self.kernel_op_mixed(
            self.support_labels.map(|x| x.to_f64().unwrap()).view(),
            input,
        );
        weights * distance + self.bias
    }

    pub fn heuristic_1(&self, error_cache: ArrayView1<f64>, i2: isize) -> usize {
        let e2 = error_cache[i2 as usize];
        let non_bounded_indices = self
            .alphas
            .iter()
            .enumerate()
            .filter(|(_, value)| (0f64 < **value) && (**value < self.C))
            .map(|(index, _)| index)
            .collect::<Array1<usize>>();
        let i1: usize;
        if non_bounded_indices.len() > 0 {
            let current = non_bounded_indices
                .iter()
                .map(|x| error_cache[*x])
                .collect::<Array1<f64>>();
            if e2 >= 0f64 {
                i1 = argmax_1d_f64(current.view());
            } else {
                i1 = argmin_1d_f64(current.view());
            }
        } else {
            i1 = argmax_1d_f64((error_cache.to_owned() - e2).map(|x| x.abs()).view());
        }
        i1
    }

    pub fn heuristic_2(&self, non_kkt_array: &mut Array1<usize>) -> isize {
        let mut init_i2 = -1isize;
        for elem in non_kkt_array.clone() {
            non_kkt_array.remove_index(ndarray::Axis(0), elem);
            if !self.check_kkt(elem) {
                init_i2 = elem as isize;
                break;
            }
        }
        if init_i2 == -1 {
            //all samples satisfy KKT conditions, so we build new kKT array
            let indices = Array1::from_shape_fn(self.alphas.len(), |x| x);
            let kkt_indexes = self.check_kkt_multi(indices.view());
            let not_kkt_array = indices
                .iter()
                .filter(|x| !kkt_indexes[**x])
                .map(|x| *x)
                .collect::<Array1<usize>>();
            if non_kkt_array.len() > 0 {
                //TODO: shuffle
                //still stuff to do
                init_i2 = non_kkt_array[0] as isize;
                *non_kkt_array = not_kkt_array.slice(ndarray::s![1..-1]).to_owned();
            }
        }
        init_i2
    }

    pub fn check_kkt(&self, index: usize) -> bool {
        let alpha_index = self.alphas[index];
        let row = self.support_vectors.row(index);
        let mut row_2d = Array2::zeros((1, row.len()));
        row_2d.row_mut(1).assign(&row.view());
        let score = self.predict_raw(row_2d.view())[0] as f64;
        let label_index = self.support_labels[index];
        let residual = label_index as f64 * score - 1f64;
        let condition_one = (alpha_index < self.C) && (residual < -self.kkt_value);
        let condition_two = (alpha_index > 0f64) && (residual > self.kkt_value);
        !(condition_one | condition_two)
    }

    fn check_kkt_multi(&self, indices: ArrayView1<usize>) -> Array1<bool> {
        let alphas = Array1::from_shape_fn(indices.len(), |x| self.alphas[indices[x]]);
        let array =
            Array2::from_shape_fn((alphas.len(), self.support_vectors.ncols()), |(x, y)| {
                self.support_vectors[(indices[x], y)]
            });
        let scores = self.predict_raw(array.view()).map(|x| x.to_f64().unwrap());
        let label_indices = Array1::from_shape_fn(indices.len(), |x| {
            self.support_labels[indices[x]].to_f64().unwrap()
        });
        let residuals = label_indices * scores - 1f64;
        let condition_one = Array1::from_shape_fn(alphas.len(), |x| {
            (alphas[x] < self.C) && (residuals[x] < -self.kkt_value)
        });
        let condition_two = Array1::from_shape_fn(alphas.len(), |x| {
            (alphas[x] > 0f64) && (residuals[x] > self.kkt_value)
        });
        let res = Array1::from_shape_fn(alphas.len(), |x| !(condition_one[x] | condition_two[x]));
        res
    }

    pub fn compute_boundaries(
        &self,
        alpha_one: f64,
        alpha_two: f64,
        label_one: i32,
        label_two: i32,
    ) -> (f64, f64) {
        let lower_bound: f64;
        let upper_bound: f64;

        if label_one == label_two {
            lower_bound = 0f64.max(alpha_one + alpha_two - self.C);
            upper_bound = self.C.min(alpha_one + alpha_two);
        } else {
            lower_bound = 0f64.max(alpha_one - alpha_two);
            upper_bound = self.C.min(alpha_two - alpha_one);
        }
        (lower_bound, upper_bound)
    }

    pub fn compute_eta(&self, row_one: ArrayView1<f64>, row_two: ArrayView1<f64>) -> f64 {
        let one = self.kernel_op_1d(row_one, row_one);
        let two = self.kernel_op_1d(row_two, row_two);
        let three = self.kernel_op_1d(row_one, row_two);
        one + two - (three.powi(2))
    }

    pub fn compute_b(
        &self,
        new_alpha_one: f64,
        new_alpha_two: f64,
        error_one: f64,
        error_two: f64,
        i1: usize,
        i2: usize,
    ) -> f64 {
        let (feature_one, feature_two) =
            (self.support_vectors.row(i1), self.support_vectors.row(i2));
        let b1 = self.bias
            - error_one
            - self.support_labels[i1] as f64
                * (new_alpha_one - self.alphas[i1])
                * self.kernel_op_1d(feature_one, feature_one)
            - self.support_labels[i2] as f64 * (new_alpha_two - self.alphas[i2])
            - self.kernel_op_1d(feature_one, feature_one);
        let b2 = self.bias
            - error_two
            - self.support_labels[i1] as f64
                * (new_alpha_one - self.alphas[i1])
                * self.kernel_op_1d(feature_one, feature_two)
            - self.support_labels[i2] as f64
                * (new_alpha_two - self.alphas[i2])
                * self.kernel_op_1d(feature_two, feature_two);
        if (0f64 < new_alpha_one) && (new_alpha_one < self.C) {
            b1
        } else if (0f64 < new_alpha_two) && (new_alpha_two < self.C) {
            b2
        } else {
            (b1 + b2) / 2f64
        }
    }

    pub fn kernel_op_1d(&self, row_one: ArrayView1<f64>, row_two: ArrayView1<f64>) -> f64 {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_1d(row_one, row_two, gamma),
            SVMKernel::Linear => linear_kernel_1d(row_one, row_two),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_1d(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => sigmoid_kernel_1d(row_one, row_two, coef_, alpha),
        }
    }
    pub fn kernel_op_2d(&self, row_one: ArrayView2<f64>, row_two: ArrayView2<f64>) -> Array2<f64> {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_2d(row_one, row_two, gamma),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_2d(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => sigmoid_kernel_2d(row_one, row_two, coef_, alpha),
            SVMKernel::Linear => linear_kernel_2d(row_one, row_two),
        }
    }

    pub fn kernel_op_mixed(
        &self,
        row_one: ArrayView1<f64>,
        row_two: ArrayView2<f64>,
    ) -> Array1<f64> {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_mixed(row_one, row_two, gamma),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_mixed(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => {
                sigmoid_kernel_mixed(row_one, row_two, coef_, alpha)
            }
            SVMKernel::Linear => linear_kernel_mixed(row_one, row_two),
        }
    }
}

impl SVCBuilder {
    pub fn new() -> Self {
        Self {
            C: 0.0,
            kkt_value: 1e-3,
            kernel: SVMKernel::Linear,
            svms: vec![],
            strategy: TrainTestSplitStrategy::None,
            data: TrainTestSplitStrategyData::default(),
            target_index: 0,
            nclasses: 0,
            scaler: ScalerState::default(),
        }
    }

    pub fn bias(&self) -> Array1<f64> {
        self.svms.iter().map(|x| x.bias).collect()
    }

    pub fn support_vectors(&self) -> Array1<(ArrayView2<f64>, ArrayView1<i32>)> {
        self.svms
            .iter()
            .map(|x| (x.support_vectors.view(), x.support_labels.view()))
            .collect()
    }

    pub fn alphas(&self) -> Array1<ArrayView1<f64>> {
        self.svms
            .iter()
            .map(|x| x.alphas.view())
            .collect::<Array1<ArrayView1<f64>>>()
    }

    pub fn set_kkt(self, value: f64) -> Self {
        Self {
            kkt_value: value,
            ..self
        }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn set_c(self, c: f64) -> Self {
        Self { C: c, ..self }
    }

    pub fn kernel(self, kernel: SVMKernel) -> Self {
        Self { kernel, ..self }
    }

    //TODO: put this in a trait, because ATP it's boilerplate
    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_index = dataset._get_string_index(target);
        self.nclasses = dataset.nunique(target);
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
                scaler.transform(&mut self.data.get_test_mut().0);
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                scaler.transform(&mut self.data.get_test_mut().0);
                scaler.transform(&mut self.data.get_eval_mut().0);
            }
            TrainTestSplitStrategy::None => {}
        };
        self.internal_fit();
    }

    pub fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        if self.nclasses == 1 {
            panic!("Not enough classes");
        }
        if self.nclasses == 2 {
            //train 1 svm
            //initialize it with details needed
            let mut estimator = RawSVC::default();
            //add in the characteristics
            estimator.C = self.C;
            estimator.kernel = self.kernel.clone();
            estimator.kkt_value = self.kkt_value;
            //train the estimator
            let labels = labels.map(|x| if *x == 0 { 1i32 } else { -1i32 });
            estimator.binomial_fit(features, labels.view(), 20);
            self.svms.push(estimator);
        }
        if self.nclasses > 2 {
            //we need a multithreaded multinomial fit
            //ideally we can actually even use a parallel iterator but it might get too messy
            (0..self.nclasses).for_each(|_| {
                let mut estimator = RawSVC::default();
                estimator.C = self.C;
                estimator.kkt_value = self.kkt_value;
                estimator.kernel = self.kernel.clone();
                self.svms.push(estimator);
            });
            let vector = Vec::from_iter((0..self.nclasses).map(|_| RawSVC::default()));
            let estimators = Arc::new(RwLock::new(vector));
            rayon::scope_fifo(|s| {
                for class in 0..self.nclasses {
                    let mut current_estimator = self.svms[class].clone();
                    let curr_ref = estimators.clone();
                    s.spawn_fifo(move |_| {
                        let labels =
                            labels.map(|x| if *x as usize == class { 1i32 } else { -1i32 });
                        current_estimator.binomial_fit(features, labels.view(), 20);
                        let mut lock = curr_ref.write().unwrap();
                        lock[class] = current_estimator;
                        drop(lock);
                    });
                }
            });
            //estimators should hopefully contain the raw svcs, and since execution of their fits should be done...
            //we can just do this:
            self.svms = Arc::into_inner(estimators).unwrap().into_inner().unwrap();
        }
    }

    fn predict_scores_one_class(&self, data: ArrayView2<f64>) -> Array1<f64> {
        self.svms[0].predict_raw(data)
    }

    fn predict_scores_multi_class(&self, data: ArrayView2<f64>) -> Array2<f64> {
        let mut allocated = Array2::from_elem((data.nrows(), self.nclasses), 0f64);
        self.svms.iter().enumerate().for_each(|(index, svm)| {
            allocated
                .column_mut(index)
                .assign(&svm.predict_raw(data).view())
        });
        allocated
    }

    pub fn predict_scores(&self, data: ArrayView2<f64>) -> Array2<f64> {
        self.predict_scores_multi_class(data)
    }

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        if self.nclasses == 2 {
            let scores = self.predict_scores_one_class(data);
            scores.map(|x| if *x >= 0f64 { 1 } else { 0 })
        } else {
            let scores = self.predict_scores_multi_class(data);
            Array1::from_shape_fn(data.nrows(), |x| argmax_1d_f64(scores.row(x)) as u32)
        }
    }

    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
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

    #[test]
    fn test_something() {}
}
