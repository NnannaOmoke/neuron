use crate::{
    base_array::BaseDataset,
    svm::{
        kernel_cache::IndexPair, kernel_cache::KernelCache, linear_kernel_1d, linear_kernel_2d,
        linear_kernel_mixed, polynomial_kernel_1d, polynomial_kernel_2d, polynomial_kernel_mixed,
        rbf_kernel_1d, rbf_kernel_2d, rbf_kernel_mixed, sigmoid_kernel_1d, sigmoid_kernel_2d,
        sigmoid_kernel_mixed, SVMKernel,
    },
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
    utils::{
        math::{argmax_1d_f64, argmin_1d_f64, into_column_matrix, outer_product, shuffle_1d},
        scaler::{Scaler, ScalerState},
    },
    Array2, ArrayView2,
};

use ndarray::{array, Array1, ArrayView1, ArrayViewMut1};
use num_traits::ToPrimitive;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::{cell::RefCell, collections::HashSet};

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

#[derive(Default, Clone, Debug)]
pub(crate) struct RawSVC {
    C: f64,
    kkt_value: f64,
    support_vectors: Array2<f64>,
    support_labels: Array1<i32>,
    alphas: Array1<f64>,
    bias: f64,
    kernel: SVMKernel,
}

impl RawSVC {
    fn take_step(
        &mut self,
        i1: usize,
        i2: usize,
        error_cache: &mut ArrayViewMut1<f64>,
        cache: &mut KernelCache,
    ) -> bool {
        if i1 == i2 {
            return false;
        }
        let alpha_one = self.alphas[i1];
        let alpha_two = self.alphas[i2];
        let label1 = self.support_labels[i1];
        let label2 = self.support_labels[i2];
        let feature_one = self.support_vectors.row(i1);
        let feature_two = self.support_vectors.row(i2);
        let error1 = error_cache[i1];
        let error2 = error_cache[i2];
        let s = label1 * label2;
        let (lower, upper) = self.compute_boundaries(alpha_one, alpha_two, label1, label2);
        if upper == lower {
            return false;
        }
        // let k11 = self.kernel_op_1d(feature_one, feature_one);
        let k11 = cache.get([i1, i1].into(), self);
        // let k22 = self.kernel_op_1d(feature_two, feature_two);
        let k22 = cache.get([i2, i2].into(), self);
        // let k12 = self.kernel_op_1d(feature_one, feature_two);
        let k12 = cache.get([i1, i2].into(), self);
        let eta = 2f64 * k12 - k11 - k22;
        let mut alpha_two_new;
        if eta < 0f64 {
            alpha_two_new = alpha_two - ((label2 as f64 * (error1 - error2)) / eta);
            alpha_two_new = if alpha_two_new >= upper {
                upper
            } else if (lower < alpha_two_new) && (alpha_two_new < upper) {
                alpha_two_new
            } else {
                lower
            };
        } else {
            let mut copy = self.alphas.clone();
            copy[i2] = lower;
            let lobj = self.objective_function(
                copy.view(),
                self.support_labels.map(|x| *x as f64).view(),
                self.support_vectors.view(),
            );
            copy[i2] = upper;
            let hobj = self.objective_function(
                copy.view(),
                self.support_labels.map(|x| *x as f64).view(),
                self.support_vectors.view(),
            );
            if lobj > (hobj + self.kkt_value) {
                alpha_two_new = lower;
            } else if lobj < (hobj + self.kkt_value) {
                alpha_two_new = hobj;
            } else {
                alpha_two_new = alpha_two;
            }
        }
        if alpha_two_new < 1e-9 {
            alpha_two_new = 0f64
        } else if alpha_two_new > (self.C - 1e-9) {
            alpha_two_new = self.C
        }

        if (alpha_two_new - alpha_two).abs()
            < self.kkt_value * (alpha_two + alpha_two_new + self.kkt_value)
        {
            return false;
        }
        let alpha_one_new = alpha_one + s as f64 * (alpha_two - alpha_two_new);

        let bias = self.compute_b(
            alpha_one_new,
            alpha_two_new,
            error1,
            error2,
            i1,
            i2,
            k11,
            k22,
            k12,
        );
        self.alphas[i1] = alpha_one_new;
        self.alphas[i2] = alpha_two_new;

        if (0f64 < alpha_two_new) && (alpha_two_new < self.C) {
            error_cache[i2] = 0.0;
        }
        if (0f64 < alpha_one_new) && (alpha_one_new < self.C) {
            error_cache[i1] = 0.0;
        }
        let non_optimized = (0..self.support_vectors.nrows())
            .filter(|&x| x != i1 && x != i2)
            .collect::<Vec<usize>>();
        non_optimized.iter().for_each(|&x| {
            let val = label1 as f64
                * (alpha_one_new - alpha_one)
                * cache.get([i1, x].into(), self)
                // * self.kernel_op_1d(self.support_vectors.row(i1), self.support_vectors.row(x))
                + label2 as f64
                    * (alpha_two_new - alpha_two)
                    * cache.get([i2, x].into(), self)
                    // * self.kernel_op_1d(self.support_vectors.row(i2), self.support_vectors.row(x))
                + (self.bias - bias);

            error_cache[x] += val;
        });
        self.bias = bias;
        true
    }

    pub fn _hueristic_2(
        &mut self,
        i2: usize,
        error_cache: &mut ArrayViewMut1<f64>,
        kernel_cache: &mut KernelCache,
    ) -> bool {
        let label2 = self.support_labels[i2];
        let alpha2 = self.alphas[i2];
        let error2 = error_cache[i2];
        let residual = error2 * label2 as f64;
        let i1;
        if ((residual < -self.kkt_value) && (alpha2 < self.C))
            || ((residual > self.kkt_value) && (alpha2 > 0.0))
        {
            if self
                .alphas
                .iter()
                .filter(|alpha| (**alpha != 0f64) && (**alpha != self.C))
                .count()
                > 1
            {
                if error2 > 0f64 {
                    i1 = argmin_1d_f64(error_cache.view());
                } else {
                    i1 = argmax_1d_f64(error_cache.view())
                }
                let flag = self.take_step(i1, i2, error_cache, kernel_cache);
                if flag {
                    return flag;
                }
            }
            let mut rng = rand::thread_rng();
            let mut candidates = self
                .alphas
                .iter()
                .enumerate()
                .filter(|(_, &alphas)| (0f64 < alphas) && (alphas < self.C))
                .map(|(index, _)| index)
                .collect::<Vec<usize>>();
            candidates.shuffle(&mut rng);
            for i1 in candidates {
                let flag = self.take_step(i1, i2, error_cache, kernel_cache);
                if flag {
                    return flag;
                }
            }
            //if nothing works, restart
            let random = rng.gen::<usize>();
            return (random..self.alphas.len() + random)
                .any(|x| self.take_step(x % random, i2, error_cache, kernel_cache));
        }
        false
    }

    pub fn _binary_fit(&mut self, features: ArrayView2<f64>, labels: ArrayView1<i32>) {
        let (nrows, ncols) = (features.nrows(), features.ncols());
        self.support_vectors = features.to_owned();
        self.support_labels = labels.to_owned();
        self.alphas = Array1::from_elem(nrows, 0f64);
        let mut error_cache =
            self.predict_raw(self.support_vectors.view()) - self.support_labels.map(|x| *x as f64);
        let mut kernel_cache = KernelCache::new_from_feature_size(features);
        let mut count = 0;
        let mut examined = true;
        while (count > 0) || examined {
            count = 0;
            if examined {
                for i2 in 0..nrows {
                    let flag =
                        self._hueristic_2(i2, &mut error_cache.view_mut(), &mut kernel_cache);
                    count += flag as i32;
                }
            } else {
                let candidates = self
                    .alphas
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| (x < 0f64) && (x < self.C))
                    .map(|(index, _)| index)
                    .collect::<Vec<usize>>();
                for index in candidates {
                    let flag =
                        self._hueristic_2(index, &mut error_cache.view_mut(), &mut kernel_cache);
                    count += flag as i32;
                }
            }
            if examined {
                examined = false;
            } else if count == 0 {
                examined = true;
            }
        }
        //shorten the support vector and support indices vectors by removing  those with 0 alpha values
        let support_indices = self
            .alphas
            .iter()
            .enumerate()
            .filter(|(_, alpha)| **alpha != 0f64)
            .map(|(index, _)| index)
            .collect::<Vec<usize>>();
        self.support_labels = Array1::from_shape_fn(support_indices.len(), |x| {
            self.support_labels[support_indices[x]]
        });
        self.support_vectors = Array2::from_shape_fn((support_indices.len(), ncols), |(x, y)| {
            self.support_vectors[(support_indices[x], y)]
        });
        self.alphas =
            Array1::from_shape_fn(support_indices.len(), |x| self.alphas[support_indices[x]]);
    }

    pub fn predict_raw(&self, input: ArrayView2<f64>) -> Array1<f64> {
        let weights = self.alphas.view().to_owned() * self.support_labels.map(|x| *x as f64);
        let distance = self.kernel_op_2d(self.support_vectors.view(), input);
        weights.dot(&distance.view()) - self.bias
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
            lower_bound = 0f64.max((alpha_one + alpha_two) - self.C);
            upper_bound = self.C.min(alpha_one + alpha_two);
        } else {
            lower_bound = 0f64.max(alpha_two - alpha_one);
            upper_bound = self.C.min(self.C + alpha_two - alpha_one);
        }
        (lower_bound, upper_bound)
    }

    pub fn compute_b(
        &self,
        new_alpha_one: f64,
        new_alpha_two: f64,
        error_one: f64,
        error_two: f64,
        i1: usize,
        i2: usize,
        k11: f64,
        k22: f64,
        k12: f64,
    ) -> f64 {
        let b1 = self.bias
            + error_one
            + self.support_labels[i1] as f64 * (new_alpha_one - self.alphas[i1]) * k11
            + self.support_labels[i2] as f64 * (new_alpha_two - self.alphas[i2]) * k12;
        let b2 = self.bias
            + error_two
            + self.support_labels[i1] as f64 * (new_alpha_one - self.alphas[i1]) * k12
            + self.support_labels[i2] as f64 * (new_alpha_two - self.alphas[i2]) * k22;
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

    pub(crate) fn kernel_op_helper(&self, i1: usize, i2: usize) -> f64 {
        self.kernel_op_1d(self.support_vectors.row(i1), self.support_vectors.row(i2))
    }

    pub fn objective_function(
        &self,
        alphas: ArrayView1<f64>,
        target: ArrayView1<f64>,
        input: ArrayView2<f64>,
    ) -> f64 {
        alphas.sum()
            - (0.5
                * outer_product(target, target)
                * self.kernel_op_2d(input, input)
                * outer_product(alphas, alphas))
            .sum()
    }
}

impl SVCBuilder {
    pub fn new() -> Self {
        Self {
            C: 0.0,
            kkt_value: 1e-2,
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
            let labels = labels.map(|x| if *x == 1 { 1i32 } else { -1i32 });
            estimator._binary_fit(features, labels.view());
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
                        current_estimator._binary_fit(features, labels.view());
                        dbg!("This done, {} class", class);
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
            dbg!(scores.view());
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
    use crate::utils::metrics::accuracy;
    use std::path::Path;

    #[test]
    fn test_convergence_svc_bi() {
        let dataset = BaseDataset::from_csv(
            Path::new("src/base_array/test_data/diabetes.csv"),
            true,
            true,
            b',',
        );
        let dataset = dataset.unwrap();
        let mut svc = SVCBuilder::new()
            .scaler(ScalerState::ZScore)
            .train_test_split_strategy(TrainTestSplitStrategy::TrainTest(0.7))
            .kernel(SVMKernel::Linear)
            .set_c(1f64);
        svc.fit(&dataset, "Outcome");
        let value = svc.evaluate(accuracy)[0];
    }

    #[test]
    fn test_convergence_svc_multi() {
        let dataset = BaseDataset::from_csv(
            Path::new("src/base_array/test_data/IRIS.csv"),
            true,
            true,
            b',',
        );
        let dataset = dataset.unwrap();
        let mut svc = SVCBuilder::new()
            .scaler(ScalerState::ZScore)
            .train_test_split_strategy(TrainTestSplitStrategy::TrainTest(0.9))
            .set_c(1f64)
            .kernel(SVMKernel::Linear);
        svc.fit(&dataset, "species");
        let value = svc.evaluate(accuracy)[0];
        dbg!(value);
    }
}
