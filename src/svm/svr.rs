use crate::base_array::BaseDataset;
use crate::svm::kernel_cache::CacheableSVM;
use crate::svm::kernel_cache::KernelCache;
use crate::svm::SVMKernel;
use crate::svm::{
    linear_kernel_1d, linear_kernel_2d, linear_kernel_mixed, polynomial_kernel_1d,
    polynomial_kernel_2d, polynomial_kernel_mixed, rbf_kernel_1d, rbf_kernel_2d, rbf_kernel_mixed,
    sigmoid_kernel_1d, sigmoid_kernel_2d, sigmoid_kernel_mixed,
};
use crate::utils::math::argmax_1d_f64;
use crate::utils::math::argmin_1d_f64;
use crate::utils::model_selection::TrainTestSplitStrategy;
use crate::utils::model_selection::TrainTestSplitStrategyData;
use crate::utils::scaler::Scaler;
use crate::utils::scaler::ScalerState;
use ndarray::prelude::*;
use rand::prelude::*;
#[derive(Clone, Default, Debug)]
pub struct RawSVR {
    C: f64,
    eps: f64,
    support_vectors: Array2<f64>,
    support_labels: Array1<f64>,
    alphas: Array1<f64>,
    alpha_stars: Array1<f64>,
    bias: f64,
    kernel: SVMKernel,
}

impl CacheableSVM for RawSVR {
    fn kernel_op_1d(&self, row_one: ArrayView1<f64>, row_two: ArrayView1<f64>) -> f64 {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_1d(row_one, row_two, gamma),
            SVMKernel::Linear => linear_kernel_1d(row_one, row_two),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_1d(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => sigmoid_kernel_1d(row_one, row_two, coef_, alpha),
        }
    }

    fn kernel_op_2d(&self, row_one: ArrayView2<f64>, row_two: ArrayView2<f64>) -> Array2<f64> {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_2d(row_one, row_two, gamma),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_2d(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => sigmoid_kernel_2d(row_one, row_two, coef_, alpha),
            SVMKernel::Linear => linear_kernel_2d(row_one, row_two),
        }
    }

    fn kernel_op_mixed(&self, row_one: ArrayView1<f64>, row_two: ArrayView2<f64>) -> Array1<f64> {
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

    fn kernel_op_helper(&self, i1: usize, i2: usize) -> f64 {
        self.kernel_op_1d(self.support_vectors.row(i1), self.support_vectors.row(i2))
    }
}

impl RawSVR {
    fn take_step(
        &mut self,
        i1: usize,
        i2: usize,
        error: &mut ArrayViewMut1<f64>,
        cache: &mut KernelCache,
    ) -> bool {
        if i1 == i2 {
            return false;
        }
        let alpha_one = self.alphas[i1];
        let alpha_one_star = self.alpha_stars[i2];
        let alpha_two = self.alphas[i1];
        let alpha_two_star = self.alpha_stars[i2];

        let error_one = error[i1];
        let error_two = error[i2];

        let k11 = cache.get([i1, i1].into(), self);
        let k12 = cache.get([i1, i2].into(), self);
        let k22 = cache.get([i2, i2].into(), self);
        let eta = 2f64 * k12 - k11 - k22;
        let gamma = (alpha_one - alpha_one_star) + (alpha_two - alpha_two_star);
        let [mut case1, mut case2, mut case3, mut case4, mut finished] = [0; 5];
        //cold storage for the current, but changeable alpha values
        let alpha_one_old = alpha_one;
        let alpha_two_old = alpha_two;
        let alpha_one_star_old = alpha_one_star;
        let alpha_two_star_old = alpha_two_star;

        let mut delta_phi = error_one - error_two;

        fn change(eps: f64, former: f64, new: f64) -> bool {
            if (new - former).abs() > eps {
                true
            } else {
                false
            }
        }

        while finished == 0 {
            if case1 == 0
                && (alpha_one > 0f64 || (alpha_one_star == 0f64 && delta_phi > 0f64))
                && (alpha_two > 0f64 || (alpha_two_star == 0f64 && delta_phi < 0f64))
            {
                let lower = 0f64.max(gamma - self.C);
                let upper = self.C.min(gamma);

                if lower < upper {
                    let mut alpha_two_new = alpha_two - delta_phi / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one - (alpha_two_new - alpha_two);
                    if change(self.eps, alpha_one, alpha_one_new)
                        && change(self.eps, alpha_two, alpha_two_new)
                    {
                        self.alphas[i1] = alpha_one_new;
                        self.alphas[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
                case1 = 1;
            } else if case2 == 0
                && (alpha_one > 0f64 || (alpha_one_star == 0f64 && delta_phi > 2f64 * self.eps))
                && (alpha_two_star > 0f64 || (alpha_two == 0f64 && delta_phi < 2f64 * self.eps))
            {
                let lower = 0f64.max(gamma);
                let upper = self.C.min(self.C + gamma);

                if lower < upper {
                    let mut alpha_two_new = alpha_two_star + (delta_phi - 2f64 * self.eps) / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one + (alpha_two_new - alpha_two_star);
                    if change(self.eps, alpha_one, alpha_one_new)
                        && change(self.eps, alpha_two_star, alpha_two_new)
                    {
                        self.alphas[i1] = alpha_one_new;
                        self.alpha_stars[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
                case2 = 1;
            } else if case3 == 0
                && (alpha_one_star > 0f64 || (alpha_one == 0f64 && delta_phi < 2f64 * self.eps))
                && (alpha_two > 0f64 || (alpha_two_star == 0f64 && delta_phi < 2f64 * self.eps))
            {
                let lower = 0f64.max(-gamma);
                let upper = self.C.min(-gamma + self.C);

                if lower < upper {
                    let mut alpha_two_new = alpha_two - (delta_phi - 2f64 * self.eps) / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one_star + (alpha_two_new - alpha_two);
                    if change(self.eps, alpha_one_star, alpha_one_new)
                        && change(self.eps, alpha_two, alpha_two_new)
                    {
                        self.alpha_stars[i1] = alpha_one_new;
                        self.alphas[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
                case3 = 1;
            } else if case4 == 0
                && (alpha_one_star > 0f64 || (alpha_one == 0f64 && delta_phi < 0f64))
                && (alpha_two_star > 0f64 || (alpha_two == 0f64 && delta_phi > 0f64))
            {
                let lower = 0f64.max(-gamma - self.C);
                let upper = self.C.min(-gamma);

                if lower < upper {
                    let mut alpha_two_new = alpha_two_star + delta_phi / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one_star + (alpha_two_new - alpha_two_star);
                    if change(self.eps, alpha_one_star, alpha_one_new)
                        && change(self.eps, alpha_two_star, alpha_two_new)
                    {
                        self.alpha_stars[i1] = alpha_one_new;
                        self.alphas[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
                case4 = 1;
            }

            delta_phi = error_one
                - error_two
                - eta
                    * ((self.alphas[i1] - self.alpha_stars[i1])
                        - (alpha_one_old - alpha_one_star_old));
        }
        //updating the bias term
        let b1 = self.bias
            + error_one
            + self.support_labels[i1]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k11
            + self.support_labels[i2]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            - self.eps;
        let b1_star = self.bias
            + error_one
            + self.support_labels[i1]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k11
            + self.support_labels[i2]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            + self.eps;
        let b2 = self.bias
            + error_two
            + self.support_labels[i2]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k22
            + self.support_labels[i1]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            - self.eps;
        let b2_star = self.bias
            + error_two
            + self.support_labels[i2]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k22
            + self.support_labels[i1]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            + self.eps;
        let bias = if (0f64 < self.alphas[i1]) && (self.alphas[i1] < self.C) {
            b1
        } else if (0f64 < self.alpha_stars[i1]) && (self.alpha_stars[i1] < self.C) {
            b1_star
        } else if (0f64 < self.alphas[i2]) && (self.alpha_stars[i2] < self.C) {
            b2
        } else if (0f64 < self.alpha_stars[i2]) && (self.alpha_stars[i2] < self.C) {
            b2_star
        } else {
            //just because, honestly
            let arr = [b1, b2, b1_star, b2_star];
            let min = arr.iter().fold(0f64, |left, &right| f64::min(left, right));
            let max = arr.iter().fold(0f64, |left, &right| f64::max(left, right));
            (min + max) * 0.5
        };

        //no we've update bias, next step is to update the error cache
        let non_optimized = (0..self.support_vectors.nrows())
            .filter(|&index| index != i2 && index != i1)
            .collect::<Vec<usize>>();
        non_optimized.iter().for_each(|&opt| {
            let val = self.support_labels[i1]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * cache.get([i1, opt].into(), self)
                + self.support_labels[i2]
                    * ((self.alphas[i2] - alpha_two_old)
                        - (self.alpha_stars[i2] - alpha_two_star_old))
                    * cache.get([i2, opt].into(), self)
                + (self.bias - bias);
            error[opt] += val;
        });
        true
    }

    fn heuristic_2(
        &mut self,
        i2: usize,
        error_cache: &mut ArrayViewMut1<f64>,
        cache: &mut KernelCache,
    ) -> bool {
        // let label2 = self.support_labels[i2];
        let alpha2 = self.alphas[i2];
        let alpha2_star = self.alpha_stars[i2];
        let error2 = error_cache[i2];
        let i1;
        if (error2 > self.eps && alpha2_star < self.C)
            || (error2 < self.eps && alpha2_star > 0f64)
            || (-error2 > self.eps && alpha2 < self.C)
            || (-error2 > self.eps && alpha2 > 0f64)
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
                let flag = self.take_step(i1, i2, error_cache, cache);
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
                let flag = self.take_step(i1, i2, error_cache, cache);
                if flag {
                    return flag;
                }
            }
            //if nothing works, restart
            let random = rng.gen::<usize>();
            return (random..self.alphas.len() + random)
                .any(|x| self.take_step(x % random, i2, error_cache, cache));
        }
        false
    }

    fn binary_fit(&mut self, features: ArrayView2<f64>, labels: ArrayView1<f64>) {
        let (nrows, ncols) = (features.nrows(), features.ncols());
        self.support_vectors = features.to_owned();
        self.support_labels = labels.to_owned();
        self.alphas = Array1::from_elem(nrows, 0f64);
        self.alpha_stars = Array1::from_elem(nrows, 0f64);
        let mut cache = KernelCache::new_from_feature_size(features);
        let mut errors = self.predict(features) - self.support_labels.view();
        let mut count = 0;
        let mut examined = true;
        let mut loop_counter = 0;
        let mut minimum_changed;
        while count > 0 || examined {
            loop_counter += 1;
            count = 0;
            if examined {
                for i2 in 0..nrows {
                    let flag = self.heuristic_2(i2, &mut errors.view_mut(), &mut cache);
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
                    let flag = self.heuristic_2(index, &mut errors.view_mut(), &mut cache);
                    count += flag as i32;
                }
            }
            if loop_counter % 2 == 0 {
                minimum_changed = 1f64.max(0.1 * nrows as f64);
            } else {
                minimum_changed = 1f64;
            }
            if examined {
                examined = false;
            } else if count < minimum_changed as i32 {
                examined = true;
            }

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
            self.support_vectors =
                Array2::from_shape_fn((support_indices.len(), ncols), |(x, y)| {
                    self.support_vectors[(support_indices[x], y)]
                });
            self.alphas =
                Array1::from_shape_fn(support_indices.len(), |x| self.alphas[support_indices[x]]);
            self.alpha_stars = Array1::from_shape_fn(support_indices.len(), |x| {
                self.alpha_stars[support_indices[x]]
            });
        }
    }

    fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let kernel = self.kernel_op_2d(self.support_vectors.view(), data);
        let alphas = self.alphas.to_owned() - self.alpha_stars.view();
        let weights = alphas * self.support_labels.view();
        let res = weights.dot(&kernel.view());
        res - self.bias
    }
}

struct SVRBuilder {
    svr: RawSVR,
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, f64>,
    target: usize,
    scaler: ScalerState,
}

impl SVRBuilder {
    pub fn new() -> Self {
        Self {
            svr: RawSVR::default(),
            strategy: TrainTestSplitStrategy::default(),
            data: TrainTestSplitStrategyData::default(),
            target: 0,
            scaler: ScalerState::default(),
        }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn set_c(self, c: f64) -> Self {
        let prev = self.svr;
        Self {
            svr: RawSVR { C: c, ..prev },
            ..self
        }
    }

    pub fn set_eps(self, eps: f64) -> Self {
        let prev = self.svr;
        Self {
            svr: RawSVR { eps, ..prev },
            ..self
        }
    }

    pub fn kernel(self, kernel: SVMKernel) -> Self {
        let prev = self.svr;
        Self {
            svr: RawSVR { kernel, ..prev },
            ..self
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target = dataset._get_string_index(target);
        self.data =
            TrainTestSplitStrategyData::<f64, f64>::new_r(dataset, self.target, self.strategy);
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

    fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        self.svr.binary_fit(features, labels);
    }

    fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        self.svr.predict(data)
    }

    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
    where
        F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Vec<f64>,
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
    fn test_convergence_svr() {}
}
